#!/usr/bin/env python3
"""
Real-time camera face distortion effects for psychological horror.

Uses MediaPipe face mesh + OpenCV for CPU-only processing at 30+ FPS.
Effects scale with stress level 0-5: subtle warping, desaturation,
shadow edges, eye darkening, peripheral figures, and glitch effects.

Includes a gaze detection trick: distortions fade when the player
looks directly at the camera and reappear in peripheral vision.

Usage:
  python scripts/generate_camera_fx.py --demo
  python scripts/generate_camera_fx.py --demo --level 3
"""

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import mediapipe as mp
except ImportError:
    mp = None


def load_config(config_path="configs/fusion_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class CameraEffectsGenerator:
    def __init__(self, config_path="configs/fusion_config.yaml"):
        cfg = load_config(config_path)
        self.fx_cfg = cfg["camera_fx"]
        self.effects = self.fx_cfg["effects_per_level"]
        self.gaze_cfg = self.fx_cfg["gaze_trick"]

        self._face_mesh = None
        if mp is not None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.fx_cfg["face_mesh_confidence"],
                min_tracking_confidence=self.fx_cfg["face_mesh_confidence"],
            )

        self._gaze_blend = 1.0
        self._last_gaze_direct = False
        self._last_time = time.time()
        self._flicker_counter = 0

    def _get_face_landmarks(self, frame):
        if self._face_mesh is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def _detect_gaze_direct(self, landmarks, frame_w, frame_h):
        """Check if subject is looking directly at camera using iris landmarks."""
        if landmarks is None:
            return False

        LEFT_IRIS = [468, 469, 470, 471, 472]
        RIGHT_IRIS = [473, 474, 475, 476, 477]
        LEFT_EYE_INNER = 133
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_INNER = 362
        RIGHT_EYE_OUTER = 263

        try:
            l_iris_x = np.mean([landmarks.landmark[i].x for i in LEFT_IRIS])
            l_inner = landmarks.landmark[LEFT_EYE_INNER].x
            l_outer = landmarks.landmark[LEFT_EYE_OUTER].x
            l_ratio = (l_iris_x - l_outer) / (l_inner - l_outer + 1e-6)

            r_iris_x = np.mean([landmarks.landmark[i].x for i in RIGHT_IRIS])
            r_inner = landmarks.landmark[RIGHT_EYE_INNER].x
            r_outer = landmarks.landmark[RIGHT_EYE_OUTER].x
            r_ratio = (r_iris_x - r_inner) / (r_outer - r_inner + 1e-6)

            avg_ratio = (l_ratio + r_ratio) / 2.0
            return 0.35 < avg_ratio < 0.65
        except (IndexError, AttributeError):
            return False

    def _update_gaze_blend(self, is_direct):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now

        if is_direct:
            fade_rate = 1000.0 / max(self.gaze_cfg.get("fade_out_on_direct_gaze_ms", 500), 1)
            self._gaze_blend = max(0.0, self._gaze_blend - fade_rate * dt)
        else:
            fade_rate = 1000.0 / max(self.gaze_cfg.get("fade_in_on_peripheral_ms", 1000), 1)
            self._gaze_blend = min(1.0, self._gaze_blend + fade_rate * dt)

    def _apply_warp(self, frame, landmarks, strength):
        """Subtle face mesh warping via landmark displacement."""
        if landmarks is None or strength < 0.01:
            return frame
        h, w = frame.shape[:2]
        result = frame.copy()

        key_indices = [1, 33, 61, 133, 159, 263, 291, 362, 386, 10, 152]
        src_pts = []
        dst_pts = []

        for idx in key_indices:
            try:
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                dx = int(random.gauss(0, strength * w * 0.02))
                dy = int(random.gauss(0, strength * h * 0.02))
                src_pts.append([x, y])
                dst_pts.append([x + dx, y + dy])
            except (IndexError, AttributeError):
                continue

        if len(src_pts) < 3:
            return frame

        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)

        # Build displacement map using thin-plate-style interpolation
        map_x = np.zeros((h, w), dtype=np.float32)
        map_y = np.zeros((h, w), dtype=np.float32)
        for y_coord in range(h):
            map_y[y_coord, :] = y_coord
        for x_coord in range(w):
            map_x[:, x_coord] = x_coord

        for i in range(len(src_pts)):
            sx, sy = src_pts[i]
            dx_off = dst_pts[i][0] - src_pts[i][0]
            dy_off = dst_pts[i][1] - src_pts[i][1]
            sigma = w * 0.08

            for y_coord in range(max(0, int(sy - 3 * sigma)), min(h, int(sy + 3 * sigma))):
                for x_coord in range(max(0, int(sx - 3 * sigma)), min(w, int(sx + 3 * sigma))):
                    dist_sq = (x_coord - sx) ** 2 + (y_coord - sy) ** 2
                    weight = np.exp(-dist_sq / (2 * sigma * sigma))
                    map_x[y_coord, x_coord] += dx_off * weight
                    map_y[y_coord, x_coord] += dy_off * weight

        result = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)
        return result

    def _apply_desaturation(self, frame, amount):
        if amount < 0.01:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 1.0 - amount, gray_bgr, amount, 0)

    def _apply_vignette(self, frame, strength):
        if strength < 0.01:
            return frame
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_radius = np.sqrt(cx ** 2 + cy ** 2)
        mask = 1.0 - strength * (radius / max_radius) ** 2
        mask = np.clip(mask, 0, 1).astype(np.float32)
        return (frame * mask[:, :, np.newaxis]).astype(np.uint8)

    def _apply_shadow_edges(self, frame, strength):
        """Inject dark edge shadows for peripheral dread."""
        if strength < 0.01:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.GaussianBlur(edges, (7, 7), 0)
        shadow = (edges.astype(np.float32) / 255.0) * strength
        result = frame.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= (1.0 - shadow * 0.5)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_eye_darkening(self, frame, landmarks, strength):
        """Darken eye regions for hollow-eyed effect."""
        if landmarks is None or strength < 0.01:
            return frame
        h, w = frame.shape[:2]
        result = frame.copy()

        LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        for eye_indices in [LEFT_EYE, RIGHT_EYE]:
            pts = []
            for idx in eye_indices:
                try:
                    lm = landmarks.landmark[idx]
                    pts.append([int(lm.x * w), int(lm.y * h)])
                except (IndexError, AttributeError):
                    continue
            if len(pts) < 4:
                continue

            pts = np.array(pts, dtype=np.int32)
            margin = int(w * 0.02)
            pts_expanded = pts.copy()
            center = pts.mean(axis=0)
            for i in range(len(pts_expanded)):
                direction = pts_expanded[i] - center
                norm = np.linalg.norm(direction)
                if norm > 0:
                    pts_expanded[i] = (center + direction * (1 + margin / norm)).astype(int)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [pts_expanded], 255)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            mask_f = mask.astype(np.float32) / 255.0

            darkened = result.astype(np.float32)
            for c in range(3):
                darkened[:, :, c] *= (1.0 - mask_f * strength * 0.7)
            result = np.clip(darkened, 0, 255).astype(np.uint8)

        return result

    def _apply_color_shift(self, frame, hue_shift):
        """Shift hue for unease (negative = cooler/bluer)."""
        if abs(hue_shift) < 1:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _apply_flicker(self, frame, intensity):
        """Random brightness flicker."""
        if intensity < 0.01:
            return frame
        self._flicker_counter += 1
        if random.random() > intensity:
            return frame
        factor = 1.0 + random.uniform(-0.3, -0.1) * intensity * 10
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _apply_glitch(self, frame, intensity):
        """Random horizontal slice displacement."""
        if intensity < 0.01 or random.random() > intensity:
            return frame
        h, w = frame.shape[:2]
        result = frame.copy()
        num_slices = random.randint(1, max(1, int(intensity * 10)))
        for _ in range(num_slices):
            y = random.randint(0, h - 1)
            slice_h = random.randint(1, max(1, int(h * 0.05)))
            shift = random.randint(-int(w * 0.1), int(w * 0.1))
            y_end = min(y + slice_h, h)
            result[y:y_end] = np.roll(result[y:y_end], shift, axis=1)
        return result

    def _apply_peripheral_figures(self, frame, intensity):
        """Draw faint shadow figures at screen edges."""
        if intensity < 0.01 or random.random() > intensity * 0.3:
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()

        x = random.choice([random.randint(0, int(w * 0.15)), random.randint(int(w * 0.85), w - 1)])
        y_base = random.randint(int(h * 0.2), int(h * 0.8))
        fig_h = random.randint(int(h * 0.15), int(h * 0.35))
        fig_w = random.randint(int(w * 0.03), int(w * 0.06))

        # Dark elliptical silhouette
        cv2.ellipse(overlay, (x, y_base), (fig_w, fig_h), 0, 0, 360, (5, 5, 5), -1)
        # Head
        head_r = fig_w
        cv2.circle(overlay, (x, y_base - fig_h), head_r, (5, 5, 5), -1)

        alpha = 0.1 + intensity * 0.15
        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    def apply(self, frame, stress_level, landmarks=None):
        """Apply all effects for the given stress level.

        Args:
            frame: BGR numpy array
            stress_level: int 0-5
            landmarks: MediaPipe face landmarks (optional, detected if None)

        Returns:
            Processed BGR frame
        """
        level_key = stress_level
        params = self.effects.get(level_key, self.effects.get(str(level_key), {}))

        if not params or not params.get("enabled", True):
            return frame

        if landmarks is None:
            landmarks = self._get_face_landmarks(frame)

        if self.gaze_cfg.get("enabled", False):
            is_direct = self._detect_gaze_direct(landmarks, frame.shape[1], frame.shape[0])
            self._update_gaze_blend(is_direct)
        else:
            self._gaze_blend = 1.0

        blend = self._gaze_blend
        result = frame.copy()

        if "desaturation" in params:
            result = self._apply_desaturation(result, params["desaturation"] * blend)

        if "color_shift_hue" in params:
            result = self._apply_color_shift(result, int(params["color_shift_hue"] * blend))

        if "vignette" in params:
            result = self._apply_vignette(result, params["vignette"] * blend)

        if "shadow_edges" in params:
            result = self._apply_shadow_edges(result, params["shadow_edges"] * blend)

        if "warp" in params and landmarks is not None:
            result = self._apply_warp(result, landmarks, params["warp"] * blend)

        if "eye_dark" in params and landmarks is not None:
            result = self._apply_eye_darkening(result, landmarks, params["eye_dark"] * blend)

        if "flicker" in params:
            result = self._apply_flicker(result, params["flicker"] * blend)

        if "peripheral_figures" in params:
            result = self._apply_peripheral_figures(result, params["peripheral_figures"] * blend)

        if "glitch" in params:
            result = self._apply_glitch(result, params["glitch"] * blend)

        return result


def run_demo(initial_level=0):
    """Interactive demo: webcam feed with keyboard-controlled stress level."""
    print("=" * 60)
    print("  Camera FX Demo — Psychological Horror Effects")
    print("=" * 60)
    print("  Keys: 0-5 = set stress level, Q = quit")
    print("  Gaze trick: look directly at camera to fade effects")
    print("=" * 60)

    fx = CameraEffectsGenerator()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Running in headless mode.")
        print("Generating sample frames to assets/textures/cached/camera_fx_samples/")

        sample_dir = Path("assets/textures/cached/camera_fx_samples")
        sample_dir.mkdir(parents=True, exist_ok=True)

        dummy = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        for level in range(6):
            result = fx.apply(dummy, level)
            path = sample_dir / f"level_{level}.png"
            cv2.imwrite(str(path), result)
            print(f"  Saved {path}")

        print("Done. Use a machine with a webcam for the full interactive demo.")
        return

    level = initial_level
    labels = ["Calm", "Wary", "Nervous", "Anxious", "Frightened", "Terrified"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = fx.apply(frame, level)

        display = np.hstack([frame, processed])
        h, w = display.shape[:2]

        label = f"Level {level}: {labels[level]}"
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display, "Original", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, "Distorted", (w // 2 + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        gaze_status = "DIRECT" if fx._gaze_blend < 0.3 else "PERIPHERAL"
        cv2.putText(display, f"Gaze: {gaze_status} ({fx._gaze_blend:.1f})",
                     (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

        cv2.imshow("Horror Camera FX", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif ord("0") <= key <= ord("5"):
            level = key - ord("0")
            print(f"  Stress level → {level} ({labels[level]})")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera face distortion effects")
    parser.add_argument("--demo", action="store_true", help="Run interactive webcam demo")
    parser.add_argument("--level", type=int, default=0, help="Initial stress level (0-5)")
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()

    if args.demo:
        run_demo(initial_level=args.level)
    else:
        print("Use --demo to run the interactive webcam demo.")
        print("Import CameraEffectsGenerator class for pipeline integration.")
