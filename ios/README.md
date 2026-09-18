# iOS application

`WristScreen` is the on-device iOS application described in the paper: both trained models run with Core ML on
the phone, the fracture and injury-sign boxes are drawn on the radiograph, and the screening rule
(fracture >= 0.26 or injury sign >= 0.30) is applied per view. No image leaves the device.

## Build

1. Export the released weights to Core ML (input 1,024 px, non-maximum suppression included):

   ```
   pip install "ultralytics>=8.3" "coremltools>=8" "numpy<2"
   python -c "from ultralytics import YOLO; YOLO('weights/fracture_model.pt').export(format='coreml', imgsz=1024, nms=True)"
   python -c "from ultralytics import YOLO; YOLO('weights/signs_model.pt').export(format='coreml', imgsz=1024, nms=True)"
   ```

   or download `coreml_models.zip` from the release. Place `fracture_model.mlpackage` and `sign_model.mlpackage`
   in `ios/WristScreen/Models/`.

2. Generate the Xcode project and build (Xcode 26, iOS 17 or later):

   ```
   brew install xcodegen
   cd ios/WristScreen && xcodegen generate
   xcodebuild -project WristScreen.xcodeproj -scheme WristScreen \
       -destination 'platform=iOS Simulator,name=iPhone 17 Pro' CODE_SIGNING_ALLOWED=NO build
   ```

`Samples/` holds four 8-bit copies of GRAZPEDWRI-DX test radiographs used as in-app sample cases (CC BY 4.0).
Launching with `--case complete|buckle|sh3|normal` opens a sample directly, which is how the paper's Figure 3
screenshots were captured in the iOS Simulator.
