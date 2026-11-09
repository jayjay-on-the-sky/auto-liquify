# auto-liquify

Minimal, high-end automated and interactive **Face-Aware Liquify** in Python, reverse-engineered directly from **Photoshop 2025's Face-Aware Liquify** (`Liquify.plugin` vector distortion mesh + nozzle profiles + interactive guide trackers).

Built with `ponytail` principles (~150 lines, zero bloat, pure NumPy + OpenCV + ONNXRuntime).

---

## 1. Photoshop Guide Control & Interactive UI (`face_liquify_gui.py`)

Reverse-engineered from Adobe Photoshop's `CLFaceToolLinearTracker`, `CLFaceToolDisplacementTracker`, and `CLAdvancedControlsPanel`:
- **Direct Canvas Dragging (Nắm kéo trực tiếp)**: Hover and drag on-canvas guide handles (eyes, nose, mouth corners, lips, chin, jawline, forehead) to morph features in real-time.
- **Adobe-Style Properties Panel**: Live side panel with sliders across 4 core anatomical groups:
  - **EYES**: Eye Size, Eye Height, Eye Width, Eye Tilt.
  - **NOSE**: Nose Height, Nose Width.
  - **MOUTH**: Smile, Upper Lip, Lower Lip, Mouth Width, Mouth Height.
  - **FACE SHAPE**: Forehead, Chin Height, Jawline, Face Width.
- **Pure Texture Preservation**: Real-time Thin Plate Spline (TPS) RBF vector deformation (`K = r^2 * ln(r)`) with Lanczos4 remapping preserves 100% skin micro-textures and sharpness.
- **Static Anchor Ring**: Locks background, hair, neck, and shoulders at `0px` displacement.

```bash
python3 face_liquify_gui.py portrait.jpg
```

**Shortcuts in GUI Window**:
- **Left-Click + Drag**: Click on any face handle to drag & distort, or drag sliders on the side panel.
- **[S]**: Save the current warped photo to `liquified_result.jpg`.
- **[R]**: Reset all parameters to zero.
- **[Q] / [ESC]**: Exit.

---

## 2. Automated Douyin Beautification (`auto_liquify.py`)

Batch/headless 1-liner function applying the optimized Douyin aesthetic profile:

```python
import cv2
from auto_liquify import auto_douyin_liquify

img = cv2.imread("portrait.jpg")

beautified = auto_douyin_liquify(
    img,
    detector_path="face_detection_yunet_2023mar.onnx",
    mesh_path="face_landmarks_468.onnx",
    v_line=0.7,     # Jawline slimming (0.0 to 1.0)
    chin_lift=0.5,  # Chin point sharpening
    big_eye=0.5,    # Eye enlargement
    nose_slim=0.4,  # Nose contour reduction
)

cv2.imwrite("beautified.jpg", beautified)
```

---

## Prerequisites

```bash
pip install numpy opencv-python onnxruntime
```

Required ONNX models:
- [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (227 KB)
- [face_landmarks_468.onnx](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/112_face_mesh) (~4.9 MB MediaPipe Face Mesh)
