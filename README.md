# Real-Time Spatial AR HUD & Vision Intelligence System

An Apple Silicon-optimized perception engine integrating concurrent YOLOv8 inference, 21-point MediaPipe hand landmark tracking, and continuous spatial raycasting for touchless interaction.

![Demo](assets/demo.gif)

## Key Features
- **Heterogeneous Pipeline:** GPU-accelerated YOLOv8m inference via Apple Metal (`mps`) paired with CPU-based MediaPipe hand tracking.
- **Raycast Target Acquisition:** Mathematical hit-testing mapping 2D/3D hand vectors to YOLO bounding boxes with Kalman-stabilized ID retention.
- **Air-Writing Canvas:** Dual-hand spatial sketching using continuous NumPy array blending.
- **Biometric ROI Telemetry:** Automated facial landmark isolation upsampled into an augmented Picture-in-Picture (PiP) feed.

## Quick Start
```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)<your-username>/spatial-ar-hud-vision.git
cd spatial-ar-hud-vision

# Setup environment
python3 -m venv ai_env
source ai_env/bin/activate
pip install -r requirements.txt

# Run
python src/main.py
