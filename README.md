# Real-Time Spatial AR HUD & Vision Intelligence System

An Apple Silicon-optimized perception engine integrating concurrent YOLOv8 inference, 21-point MediaPipe hand landmark tracking, and continuous spatial raycasting for touchless interaction.

![Demo](assets/demo.gif)

## Key Features
- **Heterogeneous Pipeline:** GPU-accelerated YOLOv8m inference via Apple Metal (`mps`) paired with CPU-based MediaPipe hand tracking.
- **Raycast Target Acquisition:** Mathematical hit-testing mapping 2D/3D hand vectors to YOLO bounding boxes with Kalman-stabilized ID retention.
- **Air-Writing Canvas:** Dual-hand spatial sketching using continuous NumPy array blending.
- **Biometric ROI Telemetry:** Automated facial landmark isolation upsampled into an augmented Picture-in-Picture (PiP) feed.
