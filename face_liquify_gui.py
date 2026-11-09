"""
Interactive Photoshop Face-Aware Liquify GUI:
- Direct click-and-drag (nắm kéo) on face guide handles (eyes, nose, mouth, chin, jawline, forehead)
- Interactive Photoshop Properties Panel on the right (sliders update in real-time)
- Thin Plate Spline (TPS) RBF vector deformation
"""
import sys
import math
import cv2
import numpy as np
import onnxruntime as ort


# Photoshop FaceFeature default ranges (-1.0 to 1.0)
FACE_FEATURES = {
    "EyeSize": 0.0,
    "EyeHeight": 0.0,
    "EyeWidth": 0.0,
    "EyeTilt": 0.0,
    "NoseHeight": 0.0,
    "NoseWidth": 0.0,
    "MouthSmile": 0.0,
    "UpperLip": 0.0,
    "LowerLip": 0.0,
    "MouthWidth": 0.0,
    "MouthHeight": 0.0,
    "Forehead": 0.0,
    "ChinHeight": 0.0,
    "Jawline": 0.0,
    "FaceWidth": 0.0,
}


class FaceLiquifyEngine:
    def __init__(self, detector_path: str, mesh_path: str, providers: list[str] | None = None):
        self.detector_path = detector_path
        self.mesh_path = mesh_path
        self.session = ort.InferenceSession(mesh_path, providers=providers or ["CPUExecutionProvider"])
        self.in_name = self.session.get_inputs()[0].name
        self.detector = None
        self.last_dims = (0, 0)

    def detect_and_mesh(self, img_bgr: np.ndarray):
        h, w = img_bgr.shape[:2]
        if self.detector is None or self.last_dims != (w, h):
            self.detector = cv2.FaceDetectorYN.create(self.detector_path, "", (w, h), 0.5)
            self.last_dims = (w, h)
        _, faces = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return []

        face_data = []
        for face in faces:
            fx, fy, fw, fh = [int(round(v)) for v in face[:4]]
            cx, cy, half = fx + fw // 2, fy + fh // 2, int(max(fw, fh) * 0.75)
            x0, y0 = max(0, cx - half), max(0, cy - half)
            x1, y1 = min(w, cx + half), min(h, cy + half)
            patch = img_bgr[y0:y1, x0:x1]
            ph, pw = patch.shape[:2]
            if ph < 32 or pw < 32:
                continue

            patch_256 = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) * (1.0 / 255.0)
            res = self.session.run(None, {self.in_name: patch_256[None]})[0]
            pts = res.reshape(-1, 3)[:468, :2].copy()
            pts[:, 0] = pts[:, 0] * (pw / 256.0) + x0
            pts[:, 1] = pts[:, 1] * (ph / 256.0) + y0

            face_w = float(np.linalg.norm(pts[454] - pts[234]))
            sr = max(0.1, face_w / 200.0)

            face_data.append({
                "bbox": (x0, y0, x1, y1),
                "pts": pts,
                "sr": sr,
                "face_w": face_w,
            })
        return face_data

    @staticmethod
    def get_guide_handles(pts: np.ndarray, sr: float):
        l_eye = (pts[33] + pts[133]) * 0.5
        r_eye = (pts[362] + pts[263]) * 0.5
        l_eye_top = pts[159]
        r_eye_top = pts[386]
        l_eye_outer = pts[33]
        r_eye_outer = pts[263]

        nose_tip = pts[1]
        nose_l = pts[48]
        nose_r = pts[278]

        m_corner_l = pts[61]
        m_corner_r = pts[291]
        m_upper = pts[0]
        m_lower = pts[17]

        chin = pts[152]
        jaw_l = pts[172]
        jaw_r = pts[397]
        cheek_l = pts[234]
        cheek_r = pts[454]
        forehead = pts[10]

        handles = {
            "left_eye_size": {"pt": l_eye_top, "name": "Eye Size", "axis": (0, -1), "feature": "EyeSize", "scale": 20.0 * sr},
            "right_eye_size": {"pt": r_eye_top, "name": "Eye Size", "axis": (0, -1), "feature": "EyeSize", "scale": 20.0 * sr},
            "left_eye_width": {"pt": l_eye_outer, "name": "Eye Width", "axis": (-1, 0), "feature": "EyeWidth", "scale": 20.0 * sr},
            "right_eye_width": {"pt": r_eye_outer, "name": "Eye Width", "axis": (1, 0), "feature": "EyeWidth", "scale": 20.0 * sr},
            "nose_width_l": {"pt": nose_l, "name": "Nose Width", "axis": (-1, 0), "feature": "NoseWidth", "scale": 15.0 * sr},
            "nose_width_r": {"pt": nose_r, "name": "Nose Width", "axis": (1, 0), "feature": "NoseWidth", "scale": 15.0 * sr},
            "nose_height": {"pt": nose_tip, "name": "Nose Height", "axis": (0, -1), "feature": "NoseHeight", "scale": 20.0 * sr},
            "mouth_smile_l": {"pt": m_corner_l, "name": "Smile", "axis": (0, -1), "feature": "MouthSmile", "scale": 15.0 * sr},
            "mouth_smile_r": {"pt": m_corner_r, "name": "Smile", "axis": (0, -1), "feature": "MouthSmile", "scale": 15.0 * sr},
            "mouth_width_l": {"pt": m_corner_l, "name": "Mouth Width", "axis": (-1, 0), "feature": "MouthWidth", "scale": 20.0 * sr},
            "mouth_width_r": {"pt": m_corner_r, "name": "Mouth Width", "axis": (1, 0), "feature": "MouthWidth", "scale": 20.0 * sr},
            "upper_lip": {"pt": m_upper, "name": "Upper Lip", "axis": (0, -1), "feature": "UpperLip", "scale": 15.0 * sr},
            "lower_lip": {"pt": m_lower, "name": "Lower Lip", "axis": (0, 1), "feature": "LowerLip", "scale": 15.0 * sr},
            "chin_height": {"pt": chin, "name": "Chin Height", "axis": (0, -1), "feature": "ChinHeight", "scale": 25.0 * sr},
            "jawline_l": {"pt": jaw_l, "name": "Jawline", "axis": (1, -0.3), "feature": "Jawline", "scale": 20.0 * sr},
            "jawline_r": {"pt": jaw_r, "name": "Jawline", "axis": (-1, -0.3), "feature": "Jawline", "scale": 20.0 * sr},
            "face_width_l": {"pt": cheek_l, "name": "Face Width", "axis": (-1, 0), "feature": "FaceWidth", "scale": 25.0 * sr},
            "face_width_r": {"pt": cheek_r, "name": "Face Width", "axis": (1, 0), "feature": "FaceWidth", "scale": 25.0 * sr},
            "forehead": {"pt": forehead, "name": "Forehead", "axis": (0, -1), "feature": "Forehead", "scale": 20.0 * sr},
        }
        return handles

    @staticmethod
    def apply_deformation(img_bgr: np.ndarray, face_info: dict, params: dict) -> np.ndarray:
        pts = face_info["pts"]
        sr = face_info["sr"]
        x0, y0, x1, y1 = face_info["bbox"]
        h, w = img_bgr.shape[:2]

        src_pts, dst_pts = [], []

        # 1. Eyes
        l_eye = (pts[33] + pts[133]) * 0.5
        r_eye = (pts[362] + pts[263]) * 0.5
        eye_sz = params.get("EyeSize", 0.0) * 0.08
        eye_w = params.get("EyeWidth", 0.0) * 3.0 * sr
        eye_h = params.get("EyeHeight", 0.0) * 3.0 * sr
        eye_tilt = params.get("EyeTilt", 0.0) * 0.05

        for idx in [33, 160, 158, 133, 153, 144, 159, 145]:
            dp = pts[idx] - l_eye
            p_dst = pts[idx] + dp * eye_sz
            p_dst[0] += (1.0 if dp[0] > 0 else -1.0) * eye_w
            p_dst[1] += (1.0 if dp[1] > 0 else -1.0) * eye_h
            p_dst[1] += -dp[0] * eye_tilt
            src_pts.append(pts[idx])
            dst_pts.append(p_dst)

        for idx in [362, 385, 387, 263, 373, 380, 386, 374]:
            dp = pts[idx] - r_eye
            p_dst = pts[idx] + dp * eye_sz
            p_dst[0] += (1.0 if dp[0] > 0 else -1.0) * eye_w
            p_dst[1] += (1.0 if dp[1] > 0 else -1.0) * eye_h
            p_dst[1] += dp[0] * eye_tilt
            src_pts.append(pts[idx])
            dst_pts.append(p_dst)

        # 2. Nose
        n_w = params.get("NoseWidth", 0.0) * 3.5 * sr
        n_h = params.get("NoseHeight", 0.0) * 4.0 * sr
        src_pts.extend([pts[48], pts[278], pts[1]])
        dst_pts.extend([
            [pts[48][0] + n_w, pts[48][1]],
            [pts[278][0] - n_w, pts[278][1]],
            [pts[1][0], pts[1][1] - n_h]
        ])

        # 3. Mouth
        smile = params.get("MouthSmile", 0.0) * 4.5 * sr
        m_w = params.get("MouthWidth", 0.0) * 4.0 * sr
        m_h = params.get("MouthHeight", 0.0) * 3.0 * sr
        u_lip = params.get("UpperLip", 0.0) * 3.0 * sr
        l_lip = params.get("LowerLip", 0.0) * 3.0 * sr

        src_pts.extend([pts[61], pts[291], pts[0], pts[17]])
        dst_pts.extend([
            [pts[61][0] - m_w, pts[61][1] - smile],
            [pts[291][0] + m_w, pts[291][1] - smile],
            [pts[0][0], pts[0][1] - u_lip - m_h * 0.5],
            [pts[17][0], pts[17][1] + l_lip + m_h * 0.5]
        ])

        # 4. Face Shape
        chin_v = params.get("ChinHeight", 0.0) * 5.0 * sr
        jaw_v = params.get("Jawline", 0.0) * 5.0 * sr
        face_w_v = params.get("FaceWidth", 0.0) * 6.0 * sr
        fore_v = params.get("Forehead", 0.0) * 5.0 * sr

        src_pts.append(pts[152])
        dst_pts.append([pts[152][0], pts[152][1] - chin_v])

        for idx in [172, 136, 150]:
            src_pts.append(pts[idx])
            dst_pts.append([pts[idx][0] + jaw_v, pts[idx][1] - 0.2 * chin_v])
        for idx in [397, 365, 379]:
            src_pts.append(pts[idx])
            dst_pts.append([pts[idx][0] - jaw_v, pts[idx][1] - 0.2 * chin_v])

        for idx in [234, 93, 132]:
            src_pts.append(pts[idx])
            dst_pts.append([pts[idx][0] - face_w_v, pts[idx][1]])
        for idx in [454, 323, 361]:
            src_pts.append(pts[idx])
            dst_pts.append([pts[idx][0] + face_w_v, pts[idx][1]])

        for idx in [10, 67, 297]:
            src_pts.append(pts[idx])
            dst_pts.append([pts[idx][0], pts[idx][1] - fore_v])

        # Static anchor boundary
        pad_a = int(35 * sr + 15)
        for ax in np.linspace(x0 - pad_a, x1 + pad_a, 7):
            for ay in [y0 - pad_a, y1 + pad_a]:
                src_pts.append([ax, ay])
                dst_pts.append([ax, ay])
        for ay in np.linspace(y0 - pad_a, y1 + pad_a, 7):
            for ax in [x0 - pad_a, x1 + pad_a]:
                src_pts.append([ax, ay])
                dst_pts.append([ax, ay])

        src_p = np.array(src_pts, dtype=np.float32)
        dst_p = np.array(dst_pts, dtype=np.float32)

        rx0, ry0 = max(0, x0 - pad_a - 5), max(0, y0 - pad_a - 5)
        rx1, ry1 = min(w, x1 + pad_a + 5), min(h, y1 + pad_a + 5)
        roi_h, roi_w = ry1 - ry0, rx1 - rx0

        N = len(dst_p)
        diff = dst_p[:, None, :] - dst_p[None, :, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        K = (r**2) * np.log(np.maximum(r, 1e-4))
        np.fill_diagonal(K, 30.0)

        P = np.column_stack([np.ones(N, dtype=np.float32), dst_p])
        L = np.block([[K, P], [P.T, np.zeros((3, 3), dtype=np.float32)]])
        Y = np.vstack([src_p, np.zeros((3, 2), dtype=np.float32)])
        weights = np.linalg.solve(L, Y)
        W_mat = weights[:N]
        A_mat = weights[N:]

        grid_y, grid_x = np.indices((roi_h, roi_w), dtype=np.float32)
        pts_eval = np.column_stack([(grid_x + rx0).ravel(), (grid_y + ry0).ravel()])

        mapped = np.zeros_like(pts_eval)
        chunk_size = 50000
        for i in range(0, len(pts_eval), chunk_size):
            chunk = pts_eval[i : i + chunk_size]
            d_chunk = chunk[:, None, :] - dst_p[None, :, :]
            r_chunk = np.sqrt(np.sum(d_chunk**2, axis=-1))
            K_chunk = (r_chunk**2) * np.log(np.maximum(r_chunk, 1e-4))
            P_chunk = np.column_stack([np.ones(len(chunk), dtype=np.float32), chunk])
            mapped[i : i + chunk_size] = K_chunk @ W_mat + P_chunk @ A_mat

        map_x = mapped[:, 0].reshape(roi_h, roi_w).astype(np.float32)
        map_y = mapped[:, 1].reshape(roi_h, roi_w).astype(np.float32)

        out = img_bgr.copy()
        out[ry0:ry1, rx0:rx1] = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
        return out


def draw_photoshop_overlay(canvas: np.ndarray, face_info: dict, handles: dict, active_handle: str | None = None) -> np.ndarray:
    disp = canvas.copy()
    pts = face_info["pts"]

    # Face contour
    oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 152,
                    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    cv2.polylines(disp, [np.int32(pts[oval_indices])], False, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.polylines(disp, [np.int32(pts[[33, 160, 158, 133, 153, 144, 33]])], True, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(disp, [np.int32(pts[[362, 385, 387, 263, 373, 380, 362]])], True, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(disp, [np.int32(pts[[48, 1, 278]])], False, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.polylines(disp, [np.int32(pts[[61, 0, 291, 17, 61]])], True, (200, 200, 200), 1, cv2.LINE_AA)

    for hid, hdata in handles.items():
        px, py = int(round(hdata["pt"][0])), int(round(hdata["pt"][1]))
        is_active = (hid == active_handle)
        color = (0, 220, 255) if is_active else (255, 255, 255)
        radius = 5 if is_active else 3
        cv2.circle(disp, (px, py), radius + 2, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.circle(disp, (px, py), radius, color, -1, cv2.LINE_AA)

        if is_active:
            label = hdata["name"]
            cv2.putText(disp, label, (px + 10, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(disp, label, (px + 10, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 240, 255), 1, cv2.LINE_AA)

    return disp


def render_photoshop_panel(params: dict, active_item: str | None = None, panel_w: int = 280, panel_h: int = 700) -> tuple[np.ndarray, dict]:
    p = np.full((panel_h, panel_w, 3), 36, dtype=np.uint8)

    cv2.rectangle(p, (0, 0), (panel_w, 42), (48, 48, 48), -1)
    cv2.putText(p, "Face-Aware Liquify", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.line(p, (0, 42), (panel_w, 42), (24, 24, 24), 1)

    sections = [
        ("EYES", [("Eye Size", "EyeSize"), ("Eye Height", "EyeHeight"), ("Eye Width", "EyeWidth"), ("Eye Tilt", "EyeTilt")]),
        ("NOSE", [("Nose Height", "NoseHeight"), ("Nose Width", "NoseWidth")]),
        ("MOUTH", [("Smile", "MouthSmile"), ("Upper Lip", "UpperLip"), ("Lower Lip", "LowerLip"), ("Mouth Width", "MouthWidth"), ("Mouth Height", "MouthHeight")]),
        ("FACE SHAPE", [("Forehead", "Forehead"), ("Chin Height", "ChinHeight"), ("Jawline", "Jawline"), ("Face Width", "FaceWidth")]),
    ]

    y_cur = 60
    slider_rects = {}

    for sec_title, items in sections:
        cv2.putText(p, sec_title, (14, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        y_cur += 12
        cv2.line(p, (14, y_cur), (panel_w - 14, y_cur), (55, 55, 55), 1)
        y_cur += 18

        for label, fkey in items:
            val = params.get(fkey, 0.0)
            val_pct = int(round(val * 100))
            is_active = (fkey == active_item)

            l_color = (0, 220, 255) if is_active else (210, 210, 210)
            cv2.putText(p, label, (18, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.38, l_color, 1, cv2.LINE_AA)
            cv2.putText(p, f"{val_pct:+d}", (panel_w - 55, y_cur), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)

            s_x0, s_y = 18, y_cur + 8
            s_w = panel_w - 36
            cv2.line(p, (s_x0, s_y), (s_x0 + s_w, s_y), (60, 60, 60), 2)
            cv2.circle(p, (s_x0 + s_w // 2, s_y), 2, (90, 90, 90), -1)

            thumb_x = int(s_x0 + s_w * 0.5 + (val * 0.5) * s_w)
            thumb_color = (0, 220, 255) if is_active else (180, 180, 180)
            cv2.circle(p, (thumb_x, s_y), 4, thumb_color, -1)

            slider_rects[fkey] = (s_x0, s_y - 8, s_w, 16)
            y_cur += 30
        y_cur += 8

    # Reset button
    cv2.rectangle(p, (20, panel_h - 40), (panel_w - 20, panel_h - 15), (60, 60, 60), -1)
    cv2.putText(p, "Reset All", (panel_w // 2 - 28, panel_h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    slider_rects["__RESET__"] = (20, panel_h - 40, panel_w - 40, 25)

    return p, slider_rects


def run_interactive_liquify(
    image_path: str,
    detector_path: str,
    mesh_path: str,
):
    """Launches full interactive OpenCV GUI for Photoshop-style Face Liquify."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        return

    engine = FaceLiquifyEngine(detector_path, mesh_path)
    faces = engine.detect_and_mesh(img)
    if not faces:
        print("No faces detected.")
        return

    face = faces[0]  # default to primary face
    handles = engine.get_guide_handles(face["pts"], face["sr"])
    params = FACE_FEATURES.copy()

    # GUI State
    state = {
        "dragging_handle": None,
        "dragging_slider": None,
        "drag_start_pos": (0, 0),
        "param_start_val": 0.0,
        "active_handle": None,
        "active_feature": None,
        "dirty": True,
        "current_deformed": img.copy(),
    }

    win_name = "Photoshop Face-Aware Liquify"
    cv2.namedWindow(win_name)

    panel_w = 280

    def on_mouse(event, x, y, flags, userdata):
        canvas_w = img.shape[1]

        if event == cv2.EVENT_LBUTTONDOWN:
            if x >= canvas_w:
                # Click in panel area
                px = x - canvas_w
                py = y
                for fkey, (rx, ry, rw, rh) in slider_rects.items():
                    if rx <= px <= rx + rw and ry <= py <= ry + rh:
                        if fkey == "__RESET__":
                            for k in params:
                                params[k] = 0.0
                            state["dirty"] = True
                        else:
                            state["dragging_slider"] = fkey
                            # Set value immediately
                            val = ((px - rx) / float(rw) - 0.5) * 2.0
                            params[fkey] = max(-1.0, min(1.0, val))
                            state["active_feature"] = fkey
                            state["dirty"] = True
                        break
            else:
                # Click on canvas handles
                for hid, hdata in handles.items():
                    hx, hy = hdata["pt"]
                    dist = math.hypot(x - hx, y - hy)
                    if dist <= 12:
                        state["dragging_handle"] = hid
                        state["drag_start_pos"] = (x, y)
                        state["param_start_val"] = params[hdata["feature"]]
                        state["active_handle"] = hid
                        state["active_feature"] = hdata["feature"]
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if state["dragging_slider"]:
                px = x - canvas_w
                fkey = state["dragging_slider"]
                rx, ry, rw, rh = slider_rects[fkey]
                val = ((px - rx) / float(rw) - 0.5) * 2.0
                params[fkey] = max(-1.0, min(1.0, val))
                state["dirty"] = True

            elif state["dragging_handle"]:
                hid = state["dragging_handle"]
                hdata = handles[hid]
                dx = x - state["drag_start_pos"][0]
                dy = y - state["drag_start_pos"][1]
                ax, ay = hdata["axis"]
                proj = dx * ax + dy * ay
                delta = proj / hdata["scale"]
                new_val = max(-1.0, min(1.0, state["param_start_val"] + delta))
                params[hdata["feature"]] = new_val
                state["dirty"] = True

            else:
                # Hover detection
                if x < canvas_w:
                    hovered = None
                    for hid, hdata in handles.items():
                        hx, hy = hdata["pt"]
                        if math.hypot(x - hx, y - hy) <= 12:
                            hovered = hid
                            break
                    state["active_handle"] = hovered
                    if hovered:
                        state["active_feature"] = handles[hovered]["feature"]

        elif event == cv2.EVENT_LBUTTONUP:
            state["dragging_handle"] = None
            state["dragging_slider"] = None

    cv2.setMouseCallback(win_name, on_mouse)

    print("\n--- Photoshop Face-Aware Liquify Controls ---")
    print("• Click & Drag on face handles to warp in real-time.")
    print("• Adjust sliders on the right panel.")
    print("• Press [S] to Save output to 'liquified_result.jpg'.")
    print("• Press [R] to Reset all sliders to zero.")
    print("• Press [Q] or [ESC] to Exit.\n")

    while True:
        if state["dirty"]:
            state["current_deformed"] = engine.apply_deformation(img, face, params)
            state["dirty"] = False

        # Draw overlay on current deformed image
        canvas_view = draw_photoshop_overlay(state["current_deformed"], face, handles, state["active_handle"])
        panel_view, slider_rects = render_photoshop_panel(params, state["active_feature"], panel_w=panel_w, panel_h=canvas_view.shape[0])

        combined = np.hstack([canvas_view, panel_view])
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(15) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        elif key in (ord('r'), ord('R')):
            for k in params:
                params[k] = 0.0
            state["dirty"] = True
        elif key in (ord('s'), ord('S')):
            cv2.imwrite("liquified_result.jpg", state["current_deformed"])
            print(">> Saved current result to 'liquified_result.jpg'!")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 face_liquify_gui.py <image_path>")
        sys.exit(1)
    run_interactive_liquify(
        sys.argv[1],
        detector_path="face_detection_yunet_2023mar.onnx",
        mesh_path="face_landmarks_468.onnx",
    )
