import PhotosUI
import SwiftUI

struct SampleCase: Identifiable {
    let id: String
    let title: String
    let file: String
}

let sampleCases = [
    SampleCase(id: "complete", title: "AP view", file: "case_complete_ap"),
    SampleCase(id: "buckle", title: "AP view", file: "case_buckle_ap"),
    SampleCase(id: "sh3", title: "Lateral view", file: "case_sh3_lat"),
    SampleCase(id: "normal", title: "AP view", file: "case_normal_ap"),
]

struct ContentView: View {
    @State private var screener: Screener? = try? Screener()
    @State private var image: UIImage?
    @State private var result: ScreenResult?
    @State private var pickerItem: PhotosPickerItem?
    @State private var busy = false
    @State private var sourceName = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    if let image {
                        RadiographCard(image: image, result: result, busy: busy, sourceName: sourceName)
                        if let result { VerdictCard(result: result) }
                    } else {
                        HeroCard()
                    }
                    inputButtons
                    sampleRow
                    footer
                }
                .padding(.horizontal, 16)
                .padding(.bottom, 24)
            }
            .background(Color(red: 0.06, green: 0.07, blue: 0.09).ignoresSafeArea())
            .navigationTitle("Wrist Fracture Screen")
            .navigationBarTitleDisplayMode(image == nil ? .large : .inline)
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task {
                if let data = try? await item.loadTransferable(type: Data.self), let ui = UIImage(data: data) {
                    await analyze(ui, name: "Photo library")
                }
            }
        }
        .onAppear(perform: handleLaunchArguments)
    }

    private var inputButtons: some View {
        HStack(spacing: 12) {
            PhotosPicker(selection: $pickerItem, matching: .images) {
                Label("Choose radiograph", systemImage: "photo.on.rectangle")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .tint(Color(red: 0.13, green: 0.55, blue: 0.62))
            Button {
            } label: {
                Label("Take photo", systemImage: "camera")
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.bordered)
            .tint(.white)
        }
        .font(.headline)
    }

    private var sampleRow: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Sample cases")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
            HStack(spacing: 12) {
                ForEach(sampleCases) { c in
                    Button {
                        if let ui = UIImage(named: c.file) { Task { await analyze(ui, name: "Sample: \(c.title)") } }
                    } label: {
                        VStack(spacing: 6) {
                            Image(uiImage: UIImage(named: c.file) ?? UIImage())
                                .resizable()
                                .scaledToFill()
                                .frame(width: 76, height: 76)
                                .clipped()
                                .clipShape(RoundedRectangle(cornerRadius: 10))
                                .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color.white.opacity(0.15)))
                            Text(c.title)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var footer: some View {
        VStack(spacing: 6) {
            Label("All analysis runs on this device. No image leaves the phone.", systemImage: "lock.shield")
            Text("Flags a radiograph when the fracture model reaches 0.26 or an injury sign reaches 0.30. Research use only, not a medical device.")
                .multilineTextAlignment(.center)
        }
        .font(.caption)
        .foregroundStyle(.secondary)
        .padding(.top, 6)
    }

    @MainActor
    private func analyze(_ ui: UIImage, name: String) async {
        image = ui; result = nil; busy = true; sourceName = name
        let screener = self.screener
        let r = await Task.detached(priority: .userInitiated) { try? screener?.run(ui) }.value
        result = r; busy = false
    }

    private func handleLaunchArguments() {
        let args = ProcessInfo.processInfo.arguments
        guard let i = args.firstIndex(of: "--case"), i + 1 < args.count,
              let c = sampleCases.first(where: { $0.id == args[i + 1] }),
              let ui = UIImage(named: c.file) else { return }
        Task { await analyze(ui, name: "Sample: \(c.title)") }
    }
}

// MARK: - Cards

struct HeroCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Image(systemName: "waveform.path.ecg.rectangle")
                .font(.system(size: 40))
                .foregroundStyle(Color(red: 0.13, green: 0.83, blue: 0.88))
            Text("Two-model fracture screening for pediatric wrist radiographs")
                .font(.title3.weight(.semibold))
            Text("A fracture detector and an injury-sign detector run on every view. Detections are drawn on the image so the reader can accept or reject each one.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(18)
        .background(RoundedRectangle(cornerRadius: 18).fill(Color.white.opacity(0.06)))
    }
}

struct RadiographCard: View {
    let image: UIImage
    let result: ScreenResult?
    let busy: Bool
    let sourceName: String

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            GeometryReader { geo in
                let fit = fitRect(image.size, in: geo.size)
                ZStack(alignment: .topLeading) {
                    Image(uiImage: image)
                        .resizable()
                        .frame(width: fit.width, height: fit.height)
                        .offset(x: fit.minX, y: fit.minY)
                    if let result {
                        ForEach(result.detections) { d in
                            DetectionBox(detection: d, frame: fit)
                        }
                    }
                    if busy {
                        ProgressView("Analyzing…")
                            .padding(12)
                            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
                            .position(x: geo.size.width / 2, y: geo.size.height / 2)
                    }
                }
            }
            .frame(maxWidth: .infinity)
            .frame(height: 430)
            .background(Color.black)
            .clipShape(RoundedRectangle(cornerRadius: 14))
            HStack {
                Text(sourceName).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("\(Int(image.size.width)) × \(Int(image.size.height)) px").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    func fitRect(_ size: CGSize, in box: CGSize) -> CGRect {
        let s = min(box.width / size.width, box.height / size.height)
        let w = size.width * s, h = size.height * s
        return CGRect(x: (box.width - w) / 2, y: (box.height - h) / 2, width: w, height: h)
    }
}

struct DetectionBox: View {
    let detection: Detection
    let frame: CGRect

    var color: Color {
        switch detection.kind {
        case .fracture: return Color(red: 0.13, green: 0.83, blue: 0.88)
        case .metal: return .gray
        case .sign: return Color(red: 0.95, green: 0.54, blue: 0.24)
        }
    }

    var body: some View {
        let r = detection.rect
        let x = frame.minX + r.minX * frame.width, y = frame.minY + r.minY * frame.height
        let w = r.width * frame.width, h = r.height * frame.height
        ZStack(alignment: .topLeading) {
            RoundedRectangle(cornerRadius: 3)
                .stroke(color, lineWidth: 2.5)
                .frame(width: w, height: h)
                .offset(x: x, y: y)
            Text("\(detection.label) \(String(format: "%.2f", detection.confidence))")
                .font(.caption.weight(.bold))
                .foregroundStyle(.black)
                .padding(.horizontal, 6).padding(.vertical, 3)
                .background(color, in: RoundedRectangle(cornerRadius: 4))
                .offset(x: x, y: max(0, y - 22))
        }
    }
}

struct VerdictCard: View {
    let result: ScreenResult

    var body: some View {
        let (title, subtitle, icon, tint) = verdict
        HStack(alignment: .top, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 30))
                .foregroundStyle(tint)
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline)
                Text(subtitle).font(.subheadline).foregroundStyle(.secondary)
                HStack(spacing: 14) {
                    stat("Fracture", result.maxFracture, cutoff: Screener.fractureCutoff)
                    stat(result.topSign ?? "Injury sign", result.maxSign, cutoff: Screener.signCutoff)
                    Spacer()
                    Text(String(format: "%.0f ms", result.milliseconds)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
                }
                .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(RoundedRectangle(cornerRadius: 16).fill(tint.opacity(0.16)))
        .overlay(RoundedRectangle(cornerRadius: 16).stroke(tint.opacity(0.6), lineWidth: 1))
    }

    private func stat(_ name: String, _ value: Double, cutoff: Double) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(name).font(.caption2).foregroundStyle(.secondary)
            Text(String(format: "%.2f", value))
                .font(.subheadline.monospacedDigit().weight(value >= cutoff ? .bold : .regular))
        }
    }

    private var verdict: (String, String, String, Color) {
        if result.fractureFlag {
            return ("Fracture suspected", "A fracture box reached the screening cutoff. Review the highlighted region.", "exclamationmark.triangle.fill", Color(red: 0.95, green: 0.35, blue: 0.3))
        } else if result.signFlag {
            return ("Indirect sign of injury", "No fracture box reached its cutoff, but \(result.topSign?.lowercased() ?? "an injury sign") was detected. Consider a careful look at the physis and carpus.", "eye.trianglebadge.exclamationmark.fill", Color(red: 0.95, green: 0.6, blue: 0.24))
        } else {
            return ("No fracture detected", "Neither model reached its cutoff on this view. Screen every view of the examination.", "checkmark.seal.fill", Color(red: 0.3, green: 0.8, blue: 0.5))
        }
    }
}
