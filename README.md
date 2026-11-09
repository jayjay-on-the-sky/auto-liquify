# auto-liquify

Minimal, high-end automated **Douyin-style Face Liquify** in Python, reverse-engineered from **Photoshop Face-Aware Liquify** (`Liquify.plugin` vector distortion mesh + nozzle profiles).

Built with `ponytail` principles (~130 lines, zero bloat, pure NumPy + OpenCV + ONNXRuntime).

---

## Features

- **468-Landmark Thin Plate Spline (TPS) RBF**: Smooth, natural 2D displacement field avoiding plastic AI-hallucination or blurring. Skin pores, highlights, and micro-textures remain 100% crisp.
- **Dynamic Scale Normalization**: Automatically scales jawline, chin, and eye offset vectors according to face pixel size (`face_w / 200px`), preventing pinching/hollowing on distant or small faces.
- **Background & Hair Lock**: A ring of static TPS anchor points placed outside the face boundary guarantees **zero background warping** or shoulder bending.
- **Douyin Retouching Profiles**:
  - `v_line`: Natural jawline narrowing.
  - `chin_lift`: Chin point elevation & refinement.
  - `nose_slim`: Subtle nasal ala compression.
  - `big_eye`: Radial expansion preserving iris roundness.

---

## Quickstart

### Prerequisites

```bash
pip install numpy opencv-python onnxruntime
```

Download the required ONNX models:
- [face_detection_yunet_2023mar.onnx](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (227 KB)
- [face_landmarks_468.onnx](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/112_face_mesh) (~4.9 MB MediaPipe Face Mesh)

### Usage

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

## How It Works (Photoshop Reverse-Engineering)

1. **Detection & Alignment**: YuNet detects face bounding boxes, cropped with $1.4\times$ margin and fed into MediaPipe FaceMesh to extract 468 3D landmarks.
2. **Displacement Vectors**: Control points along the jawline, nose alae, and eyes are displaced along physiological contour normals.
3. **Radial Basis Function (Thin Plate Spline)**: Computes a global minimum-curvature mapping kernel:
   $$K(r) = r^2 \ln(r + \epsilon)$$
4. **Remapping**: Uses `cv2.remap` with `INTER_LANCZOS4` over the predicted deformation grid, updating only pixels within the face ROI.
