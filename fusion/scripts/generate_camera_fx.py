#!/usr/bin/env python3
"""
Real-time camera face distortion for psychological horror.

Takes a stress level (0-5) and applies effects to webcam/image input:
  0 = clean, 5 = heavy warping + glitch + peripheral figures.

Uses MediaPipe face mesh + OpenCV — CPU only, 30+ FPS, zero VRAM.
Includes gaze detection: distortions fade when player looks directly
at the feed (gaslighting effect from proposal).

Demo mode: saves distorted sample frames to disk for review.

Usage:
  python scripts/generate_camera_fx.py --demo             # webcam or fallback images
  python scripts/generate_camera_fx.py --demo --level 4   # specific level
  python scripts/generate_camera_fx.py --demo --all-levels # side-by-side grid
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


def load_config(config_path="configs/generation_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


STRESS_LABELS = {0: "Calm", 1: "Wary", 2: "Nervous", 3: "Anxious", 4: "Frightened", 5: "Terrified"}


class CameraEffectsGenerator:
    def __init__(self, config_path="configs/generation_config.yaml"):
        cfg = load_config(config_path)
        self.fx_cfg = cfg["camera_fx"]
        self.effects = {}
        for k, v in self.fx_cfg["effects_per_level"].items():
            self.effects[int(k)] = v
        self.gaze_cfg = self.fx_cfg["gaze_trick"]

        self._face_mesh = None
        if mp is not None:
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                min_detection_confidence=self.fx_cfg["face_mesh_confidence"],
                min_tracking_confidence=self.fx_cfg["face_mesh_confidence"],
            )

        self._gaze_blend = 1.0
        self._last_time = time.time()

    def _get_landmarks(self, frame):
        if self._face_mesh is None:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]

    def _detect_direct_gaze(self, landmarks, w, h):
        if landmarks is None:
            return False
        LEFT_IRIS = [468, 469, 470, 471, 472]
        RIGHT_IRIS = [473, 474, 475, 476, 477]
        try:
            l_x = np.mean([landmarks.landmark[i].x for i in LEFT_IRIS])
            l_in, l_out = landmarks.landmark[133].x, landmarks.landmark[33].x
            l_ratio = (l_x - l_out) / (l_in - l_out + 1e-6)

            r_x = np.mean([landmarks.landmark[i].x for i in RIGHT_IRIS])
            r_in, r_out = landmarks.landmark[362].x, landmarks.landmark[263].x
            r_ratio = (r_x - r_in) / (r_out - r_in + 1e-6)

            return 0.35 < (l_ratio + r_ratio) / 2.0 < 0.65
        except (IndexError, AttributeError):
            return False

    def _update_gaze_blend(self, is_direct):
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        if is_direct:
            rate = 1000.0 / max(self.gaze_cfg.get("fade_out_on_direct_gaze_ms", 500), 1)
            self._gaze_blend = max(0.0, self._gaze_blend - rate * dt)
        else:
            rate = 1000.0 / max(self.gaze_cfg.get("fade_in_on_peripheral_ms", 1000), 1)
            self._gaze_blend = min(1.0, self._gaze_blend + rate * dt)

    def apply(self, frame, stress_level, landmarks=None):
        """Apply horror effects to a BGR frame at stress level 0-5."""
        params = self.effects.get(stress_level, {})
        if not params or (stress_level == 0):
            return frame.copy()

        if landmarks is None:
            landmarks = self._get_landmarks(frame)

        if self.gaze_cfg.get("enabled", False):
            self._update_gaze_blend(self._detect_direct_gaze(landmarks, frame.shape[1], frame.shape[0]))
        else:
            self._gaze_blend = 1.0

        b = self._gaze_blend
        result = frame.copy()

        if "desaturation" in params:
            result = self._desaturate(result, params["desaturation"] * b)
        if "color_shift_hue" in params:
            result = self._color_shift(result, int(params["color_shift_hue"] * b))
        if "vignette" in params:
            result = self._vignette(result, params["vignette"] * b)
        if "shadow_edges" in params:
            result = self._shadow_edges(result, params["shadow_edges"] * b)
        if "warp" in params and landmarks is not None:
            result = self._warp(result, landmarks, params["warp"] * b)
        if "eye_dark" in params and landmarks is not None:
            result = self._eye_dark(result, landmarks, params["eye_dark"] * b)
        if "flicker" in params:
            result = self._flicker(result, params["flicker"] * b)
        if "peripheral_figures" in params:
            result = self._peripheral_figures(result, params["peripheral_figures"] * b)
        if "glitch" in params:
            result = self._glitch(result, params["glitch"] * b)

        return result

    def _desaturate(self, frame, amount):
        if amount < 0.01:
            return frame
        gray = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(frame, 1.0 - amount, gray, amount, 0)

    def _color_shift(self, frame, hue_shift):
        if abs(hue_shift) < 1:
            return frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    def _vignette(self, frame, strength):
        if strength < 0.01:
            return frame
        h, w = frame.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        mask = np.clip(1.0 - strength * (r / r_max) ** 2, 0, 1).astype(np.float32)
        return (frame * mask[:, :, np.newaxis]).astype(np.uint8)

    def _shadow_edges(self, frame, strength):
        if strength < 0.01:
            return frame
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edges = cv2.GaussianBlur(edges, (7, 7), 0)
        shadow = (edges.astype(np.float32) / 255.0) * strength
        result = frame.astype(np.float32)
        for c in range(3):
            result[:, :, c] *= (1.0 - shadow * 0.5)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _warp(self, frame, landmarks, strength):
        if strength < 0.01:
            return frame
        h, w = frame.shape[:2]
        key_idx = [1, 33, 61, 133, 159, 263, 291, 362, 386, 10, 152]
        map_x = np.tile(np.arange(w, dtype=np.float32), (h, 1))
        map_y = np.tile(np.arange(h, dtype=np.float32).reshape(-1, 1), (1, w))

        for idx in key_idx:
            try:
                lm = landmarks.landmark[idx]
                sx, sy = lm.x * w, lm.y * h
                dx = random.gauss(0, strength * w * 0.02)
                dy = random.gauss(0, strength * h * 0.02)
                sigma = w * 0.08
                y_lo, y_hi = max(0, int(sy - 3 * sigma)), min(h, int(sy + 3 * sigma))
                x_lo, x_hi = max(0, int(sx - 3 * sigma)), min(w, int(sx + 3 * sigma))
                yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
                dist_sq = (xx - sx) ** 2 + (yy - sy) ** 2
                weight = np.exp(-dist_sq / (2 * sigma * sigma))
                map_x[y_lo:y_hi, x_lo:x_hi] += (dx * weight).astype(np.float32)
                map_y[y_lo:y_hi, x_lo:x_hi] += (dy * weight).astype(np.float32)
            except (IndexError, AttributeError):
                continue

        return cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

    def _eye_dark(self, frame, landmarks, strength):
        if strength < 0.01:
            return frame
        h, w = frame.shape[:2]
        result = frame.astype(np.float32)
        L_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        R_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

        for eye in [L_EYE, R_EYE]:
            pts = []
            for idx in eye:
                try:
                    lm = landmarks.landmark[idx]
                    pts.append([int(lm.x * w), int(lm.y * h)])
                except (IndexError, AttributeError):
                    continue
            if len(pts) < 4:
                continue
            pts = np.array(pts, dtype=np.int32)
            center = pts.mean(axis=0)
            expanded = (center + (pts - center) * 1.4).astype(np.int32)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [expanded], 255)
            mask = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
            for c in range(3):
                result[:, :, c] *= (1.0 - mask * strength * 0.7)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _flicker(self, frame, intensity):
        if intensity < 0.01 or random.random() > intensity:
            return frame
        factor = 1.0 + random.uniform(-0.3, -0.1) * intensity * 10
        return np.clip(frame.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    def _peripheral_figures(self, frame, intensity):
        if intensity < 0.01 or random.random() > intensity * 0.3:
            return frame
        h, w = frame.shape[:2]
        overlay = frame.copy()
        x = random.choice([random.randint(0, int(w * 0.15)), random.randint(int(w * 0.85), w - 1)])
        y_base = random.randint(int(h * 0.2), int(h * 0.8))
        fig_h = random.randint(int(h * 0.15), int(h * 0.35))
        fig_w = random.randint(int(w * 0.03), int(w * 0.06))
        cv2.ellipse(overlay, (x, y_base), (fig_w, fig_h), 0, 0, 360, (5, 5, 5), -1)
        cv2.circle(overlay, (x, y_base - fig_h), fig_w, (5, 5, 5), -1)
        alpha = 0.1 + intensity * 0.15
        return cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)

    def _glitch(self, frame, intensity):
        if intensity < 0.01 or random.random() > intensity:
            return frame
        h, w = frame.shape[:2]
        result = frame.copy()
        for _ in range(random.randint(1, max(1, int(intensity * 10)))):
            y = random.randint(0, h - 1)
            sh = random.randint(1, max(1, int(h * 0.05)))
            shift = random.randint(-int(w * 0.1), int(w * 0.1))
            y_end = min(y + sh, h)
            result[y:y_end] = np.roll(result[y:y_end], shift, axis=1)
        return result

    def run_demo(self, levels=None, output_dir=None):
        """Generate demo outputs: webcam or static fallback images."""
        if levels is None:
            levels = list(range(6))
        if output_dir is None:
            output_dir = Path("assets/textures/cached/camera_fx_demo")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Camera FX Demo — Psychological Horror Distortion")
        print(f"{'='*60}")

        cap = cv2.VideoCapture(0)
        use_webcam = cap.isOpened()

        if use_webcam:
            print("  Webcam detected. Keys: 0-5 = stress level, Q = quit")
            print("  Gaze trick active: look directly at camera to fade effects")
            self._run_webcam_demo(cap, levels)
        else:
            print("  No webcam — generating static demo frames")
            self._run_static_demo(levels, output_dir)

        cap.release()
        cv2.destroyAllWindows()

    def _run_webcam_demo(self, cap, levels):
        level = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            distorted = self.apply(frame, level)
            display = np.hstack([frame, distorted])
            h, w = display.shape[:2]
            label = f"Level {level}: {STRESS_LABELS[level]}"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, "Original", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, "Distorted", (w // 2 + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            gaze = "DIRECT" if self._gaze_blend < 0.3 else "PERIPHERAL"
            cv2.putText(display, f"Gaze: {gaze} ({self._gaze_blend:.1f})",
                        (w - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)
            cv2.imshow("Horror Camera FX", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif ord("0") <= key <= ord("5"):
                level = key - ord("0")

    def _run_static_demo(self, levels, output_dir):
        # Create a test face-like image
        size = 480
        face = np.full((size, int(size * 1.33), 3), 60, dtype=np.uint8)
        w = face.shape[1]
        # Simple face shape
        cv2.ellipse(face, (w // 2, size // 2), (120, 160), 0, 0, 360, (140, 130, 120), -1)
        # Eyes
        cv2.ellipse(face, (w // 2 - 45, size // 2 - 30), (20, 10), 0, 0, 360, (60, 60, 60), -1)
        cv2.ellipse(face, (w // 2 + 45, size // 2 - 30), (20, 10), 0, 0, 360, (60, 60, 60), -1)
        cv2.circle(face, (w // 2 - 45, size // 2 - 30), 6, (20, 20, 20), -1)
        cv2.circle(face, (w // 2 + 45, size // 2 - 30), 6, (20, 20, 20), -1)
        # Nose + mouth
        cv2.line(face, (w // 2, size // 2 - 10), (w // 2, size // 2 + 20), (100, 90, 85), 2)
        cv2.ellipse(face, (w // 2, size // 2 + 50), (30, 10), 0, 0, 180, (90, 70, 70), 2)

        cell_w, cell_h = face.shape[1], face.shape[0]
        grid = np.zeros((cell_h + 40, cell_w * len(levels), 3), dtype=np.uint8)

        for i, level in enumerate(levels):
            # Force gaze_blend to 1 for static demo
            self._gaze_blend = 1.0
            distorted = self.apply(face, level)

            x = i * cell_w
            grid[40:40 + cell_h, x:x + cell_w] = distorted

            label = f"{level}: {STRESS_LABELS[level]}"
            cv2.putText(grid, label, (x + 5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Save individual
            path = output_dir / f"fx_level{level}.png"
            cv2.imwrite(str(path), distorted)
            print(f"  Level {level} ({STRESS_LABELS[level]}) -> {path}")

        grid_path = output_dir / "summary_camera_fx.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"\n  Summary grid -> {grid_path}")
        print(f"  All outputs in {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Camera face distortion effects")
    parser.add_argument("--demo", action="store_true", help="Run demo (webcam or static)")
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4, 5], help="Initial/only stress level")
    parser.add_argument("--all-levels", action="store_true", help="Generate all levels (static mode)")
    parser.add_argument("--config", default="configs/generation_config.yaml")
    args = parser.parse_args()

    if args.demo or args.all_levels:
        fx = CameraEffectsGenerator(config_path=args.config)
        levels = [args.level] if args.level is not None and not args.all_levels else list(range(6))
        fx.run_demo(levels=levels)
    else:
        print("Usage:")
        print("  python scripts/generate_camera_fx.py --demo")
        print("  python scripts/generate_camera_fx.py --demo --level 4")
        print("  python scripts/generate_camera_fx.py --demo --all-levels")


if __name__ == "__main__":
    main()
