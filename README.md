cat << 'EOF' > README.md
# Real-Time Spatial AR HUD & Vision Intelligence System

An Apple Silicon-optimized perception engine integrating concurrent YOLOv8 inference, 21-point MediaPipe hand landmark tracking, and continuous spatial raycasting for touchless interaction.

## Key Features
- **Heterogeneous Multi-Model Pipeline:** Accelerated YOLOv8m inference on Apple Silicon GPU (`mps`) paired with CPU-based MediaPipe hand landmark tracking.
- **Spatial Target Acquisition ("Force Grab"):** Mathematical hit-testing mapping 2D/3D hand coordinates into YOLO bounding volumes with temporal hold-to-lock confirmation.
- **Persistent Target Tracking:** Uses multi-object tracking IDs to maintain lock telemetry even across moving subjects.
- **Biometric ROI Telemetry:** Automated head/torso region extraction upsampled into an augmented Picture-in-Picture (PiP) surveillance uplink.
- **Non-Euclidean Air-Writing:** Dual-hand spatial sketching utilizing 360-degree angle-independent pose estimation and persistent NumPy array blending.

## Quick Start

```bash
# Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/spatial-ar-hud-vision.git
cd spatial-ar-hud-vision

# Setup environment
python3.12 -m venv ai_env
source ai_env/bin/activate
pip install -r requirements.txt

# Run
python src/main.py
