import CoreML
import UIKit

/// One detection, with its box normalised to the original radiograph (0-1, x/y/width/height).
struct Detection: Identifiable {
    enum Kind { case fracture, metal, sign }
    let id = UUID()
    let kind: Kind
    let label: String
    let confidence: Double
    let rect: CGRect
}

struct ScreenResult {
    let detections: [Detection]
    let maxFracture: Double
    let maxSign: Double
    let topSign: String?
    let milliseconds: Double

    var fractureFlag: Bool { maxFracture >= Screener.fractureCutoff }
    var signFlag: Bool { maxSign >= Screener.signCutoff }
    var flagged: Bool { fractureFlag || signFlag }
}

/// The two-model screening rule from the paper: flag when the fracture model reaches 0.26 on any view
/// or an injury sign (periosteal reaction, pronator sign, soft-tissue swelling) reaches 0.30.
final class Screener {
    static let fractureCutoff = 0.26
    static let signCutoff = 0.30
    static let signClasses: [Int: String] = [5: "Periosteal reaction", 6: "Pronator sign", 7: "Soft-tissue swelling"]

    private let fractureModel: MLModel
    private let signModel: MLModel
    private let side: Int

    init() throws {
        let cfg = MLModelConfiguration()
        #if targetEnvironment(simulator)
        cfg.computeUnits = .cpuOnly          // the Simulator has no MPSGraph backend
        #else
        cfg.computeUnits = .all
        #endif
        fractureModel = try MLModel(contentsOf: Bundle.main.url(forResource: "fracture_model", withExtension: "mlmodelc")!, configuration: cfg)
        signModel = try MLModel(contentsOf: Bundle.main.url(forResource: "sign_model", withExtension: "mlmodelc")!, configuration: cfg)
        let feat = fractureModel.modelDescription.inputDescriptionsByName["image"]!
        side = feat.imageConstraint!.pixelsWide
    }

    func run(_ image: UIImage) throws -> ScreenResult {
        let (buffer, scale, origin) = letterbox(image)
        let t0 = CFAbsoluteTimeGetCurrent()
        let fr = try predict(fractureModel, buffer)
        let sg = try predict(signModel, buffer)
        let ms = (CFAbsoluteTimeGetCurrent() - t0) * 1000

        var dets: [Detection] = []
        var maxFracture = 0.0, maxSign = 0.0
        var topSign: String? = nil
        for (cls, conf, box) in fr {
            let rect = unletterbox(box, scale: scale, origin: origin, image: image)
            if cls == 0 {
                maxFracture = max(maxFracture, conf)
                if conf >= Screener.fractureCutoff { dets.append(Detection(kind: .fracture, label: "Fracture", confidence: conf, rect: rect)) }
            } else if cls == 1, conf >= 0.5 {
                dets.append(Detection(kind: .metal, label: "Metal", confidence: conf, rect: rect))
            }
        }
        for (cls, conf, box) in sg {
            guard let name = Screener.signClasses[cls] else { continue }
            if conf > maxSign { maxSign = conf; topSign = name }
            if conf >= Screener.signCutoff {
                dets.append(Detection(kind: .sign, label: name, confidence: conf, rect: unletterbox(box, scale: scale, origin: origin, image: image)))
            }
        }
        return ScreenResult(detections: dets, maxFracture: maxFracture, maxSign: maxSign, topSign: topSign, milliseconds: ms)
    }

    // MARK: - Core ML plumbing

    /// Returns (class index, confidence, box) with the box as centre x/y, width, height normalised to the letterboxed input.
    private func predict(_ model: MLModel, _ buffer: CVPixelBuffer) throws -> [(Int, Double, [Double])] {
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "image": MLFeatureValue(pixelBuffer: buffer),
            "confidenceThreshold": MLFeatureValue(double: 0.02),
            "iouThreshold": MLFeatureValue(double: 0.7),
        ])
        let out = try model.prediction(from: input)
        guard let conf = out.featureValue(for: "confidence")?.multiArrayValue,
              let coords = out.featureValue(for: "coordinates")?.multiArrayValue else { return [] }
        let n = conf.shape[0].intValue, nc = conf.shape[1].intValue
        var result: [(Int, Double, [Double])] = []
        for i in 0..<n {
            var best = 0, bestConf = 0.0
            for c in 0..<nc {
                let v = conf[[i, c] as [NSNumber]].doubleValue
                if v > bestConf { bestConf = v; best = c }
            }
            let box = (0..<4).map { coords[[i, $0] as [NSNumber]].doubleValue }
            result.append((best, bestConf, box))
        }
        return result
    }

    /// Square letterbox with the 114-grey padding used in evaluation.
    private func letterbox(_ image: UIImage) -> (CVPixelBuffer, CGFloat, CGPoint) {
        let canvas = CGSize(width: side, height: side)
        let scale = min(canvas.width / image.size.width, canvas.height / image.size.height)
        let w = image.size.width * scale, h = image.size.height * scale
        let origin = CGPoint(x: (canvas.width - w) / 2, y: (canvas.height - h) / 2)
        let fmt = UIGraphicsImageRendererFormat.default(); fmt.scale = 1
        let rendered = UIGraphicsImageRenderer(size: canvas, format: fmt).image { ctx in
            UIColor(white: 114.0 / 255.0, alpha: 1).setFill()
            ctx.fill(CGRect(origin: .zero, size: canvas))
            image.draw(in: CGRect(origin: origin, size: CGSize(width: w, height: h)))
        }
        var pb: CVPixelBuffer?
        let attrs: [String: Any] = [kCVPixelBufferCGImageCompatibilityKey as String: true,
                                    kCVPixelBufferCGBitmapContextCompatibilityKey as String: true]
        CVPixelBufferCreate(kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &pb)
        let buf = pb!
        CVPixelBufferLockBaseAddress(buf, [])
        let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buf), width: side, height: side, bitsPerComponent: 8,
                            bytesPerRow: CVPixelBufferGetBytesPerRow(buf), space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
        ctx.draw(rendered.cgImage!, in: CGRect(origin: .zero, size: canvas))
        CVPixelBufferUnlockBaseAddress(buf, [])
        return (buf, scale, origin)
    }

    private func unletterbox(_ box: [Double], scale: CGFloat, origin: CGPoint, image: UIImage) -> CGRect {
        let s = CGFloat(side)
        let cx = (CGFloat(box[0]) * s - origin.x) / (image.size.width * scale)
        let cy = (CGFloat(box[1]) * s - origin.y) / (image.size.height * scale)
        let w = CGFloat(box[2]) * s / (image.size.width * scale)
        let h = CGFloat(box[3]) * s / (image.size.height * scale)
        return CGRect(x: cx - w / 2, y: cy - h / 2, width: w, height: h)
    }
}
