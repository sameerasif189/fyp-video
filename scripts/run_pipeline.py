#!/usr/bin/env python3
"""
End-to-end demo pipeline: fusion + generation + camera effects.

Simulates Unity integration by combining all modules:
  1. Webcam/simulated → video model → 4-class probs
  2. Simulated audio model → stress 0-5
  3. Simulated heart rate → stress 0-5
  4. Fusion → unified stress level
  5. Background gen thread (texture + audio, alternating VRAM)
  6. Real-time camera FX on webcam feed
  7. Display: original | distorted | stress gauge

Usage:
  python scripts/run_pipeline.py --simulate                 # fake model outputs
  python scripts/run_pipeline.py --simulate --stress-override 4  # force level 4
  python scripts/run_pipeline.py --simulate --scenario ramp  # stress ramp scenario
"""

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from fusion import StressFusion, STRESS_LABELS
from generate_camera_fx import CameraEffectsGenerator

try:
    from generate_audio import AudioGenerator
except ImportError:
    AudioGenerator = None

try:
    from generate_visual import VisualGenerator
except ImportError:
    VisualGenerator = None


class StressSimulator:
    """Simulates multi-modal stress inputs for demo purposes."""

    def __init__(self, scenario="calm"):
        self.scenario = scenario
        self.t = 0
        self._scenarios = {
            "calm": self._calm,
            "ramp": self._ramp,
            "spike": self._spike,
            "oscillate": self._oscillate,
            "random": self._random,
        }

    def step(self):
        self.t += 1
        fn = self._scenarios.get(self.scenario, self._calm)
        return fn()

    def _calm(self):
        return {
            "video": [0.7, 0.15, 0.1, 0.05],
            "audio": 0,
            "heartrate": 0,
        }

    def _ramp(self):
        progress = min(self.t / 300.0, 1.0)
        level = int(progress * 5)

        video_probs = [0.0] * 4
        if level <= 1:
            video_probs[0] = 1.0 - progress
            video_probs[1] = progress
        elif level <= 3:
            video_probs[1] = 1.0 - (progress - 0.4)
            video_probs[2] = progress - 0.4
        else:
            video_probs[2] = 1.0 - (progress - 0.7)
            video_probs[3] = progress - 0.7

        total = sum(video_probs)
        if total > 0:
            video_probs = [p / total for p in video_probs]
        else:
            video_probs = [1.0, 0.0, 0.0, 0.0]

        return {
            "video": video_probs,
            "audio": min(level, 5),
            "heartrate": min(level, 5),
        }

    def _spike(self):
        if 100 < self.t < 130:
            return {"video": [0.0, 0.0, 0.1, 0.9], "audio": 5, "heartrate": 4}
        if 200 < self.t < 220:
            return {"video": [0.0, 0.0, 0.1, 0.9], "audio": 5, "heartrate": 5}
        return {"video": [0.6, 0.2, 0.15, 0.05], "audio": 1, "heartrate": 1}

    def _oscillate(self):
        phase = math.sin(self.t * 0.02) * 0.5 + 0.5
        level = int(phase * 5)
        video_probs = [0.0] * 4
        video_idx = min(level // 2, 3)
        video_probs[video_idx] = 0.8
        remaining = 0.2 / 3
        for i in range(4):
            if i != video_idx:
                video_probs[i] = remaining
        return {"video": video_probs, "audio": level, "heartrate": level}

    def _random(self):
        level = np.random.randint(0, 6)
        video_probs = np.random.dirichlet([1, 1, 1, 1]).tolist()
        return {
            "video": video_probs,
            "audio": level,
            "heartrate": np.random.randint(max(0, level - 1), min(6, level + 2)),
        }


def draw_stress_gauge(frame, level, score, label, x=10, y=10):
    """Draw a stress level gauge on the frame."""
    gauge_w = 200
    gauge_h = 25
    padding = 5

    cv2.rectangle(frame, (x, y), (x + gauge_w + 2 * padding, y + gauge_h + 2 * padding + 25),
                  (30, 30, 30), -1)

    cv2.putText(frame, f"Stress: {label} ({score:.1f})",
                (x + padding, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    bar_y = y + 25
    fill = int((score / 5.0) * gauge_w)
    colors = [
        (0, 200, 0),    # green
        (0, 200, 200),  # yellow-green
        (0, 200, 255),  # yellow
        (0, 140, 255),  # orange
        (0, 60, 255),   # red-orange
        (0, 0, 255),    # red
    ]
    color = colors[min(level, 5)]

    cv2.rectangle(frame, (x + padding, bar_y), (x + padding + gauge_w, bar_y + gauge_h),
                  (60, 60, 60), -1)
    if fill > 0:
        cv2.rectangle(frame, (x + padding, bar_y), (x + padding + fill, bar_y + gauge_h),
                      color, -1)

    for i in range(1, 6):
        tick_x = x + padding + int(i * gauge_w / 6)
        cv2.line(frame, (tick_x, bar_y), (tick_x, bar_y + gauge_h), (100, 100, 100), 1)


def draw_modality_panel(frame, state, x=10, y=80):
    """Draw per-modality stress breakdown."""
    cv2.rectangle(frame, (x, y), (x + 210, y + 80), (30, 30, 30), -1)

    mod_labels = {"video": "VID", "audio": "AUD", "heartrate": "HR"}
    row = 0
    for mod, data in state.per_modality.items():
        label = mod_labels.get(mod, mod[:3].upper())
        score = data["score"]
        text = f"{label}: {score:.1f}"
        cv2.putText(frame, text, (x + 5, y + 18 + row * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        row += 1


def run_pipeline(scenario="calm", stress_override=None, enable_audio=False, enable_visual=False):
    """Run the full demo pipeline."""
    print("=" * 60)
    print("  Adaptive Horror Pipeline — End-to-End Demo")
    print("=" * 60)
    print(f"  Scenario: {scenario}")
    if stress_override is not None:
        print(f"  Stress override: {stress_override}")
    print(f"  Audio gen: {'ON' if enable_audio else 'OFF (use --enable-audio)'}")
    print(f"  Visual gen: {'ON' if enable_visual else 'OFF (use --enable-visual)'}")
    print("  Keys: 0-5 = override stress, R = release override")
    print("         S = cycle scenario, Q = quit")
    print("=" * 60)

    fusion = StressFusion()
    camera_fx = CameraEffectsGenerator()
    simulator = StressSimulator(scenario)

    audio_gen = None
    visual_gen = None

    if enable_audio and AudioGenerator is not None:
        audio_gen = AudioGenerator()
        audio_gen.start_background_thread()

    if enable_visual and VisualGenerator is not None:
        visual_gen = VisualGenerator()
        visual_gen.start_background_thread()

    cap = cv2.VideoCapture(0)
    use_webcam = cap.isOpened()

    if not use_webcam:
        print("No webcam detected. Using synthetic frames.")

    current_override = stress_override
    scenarios = ["calm", "ramp", "spike", "oscillate", "random"]
    scenario_idx = scenarios.index(scenario) if scenario in scenarios else 0

    frame_count = 0
    fps_timer = time.time()
    fps = 0.0

    prev_level = -1

    while True:
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            frame = np.random.randint(40, 80, (480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "No webcam — synthetic frame",
                        (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

        inputs = simulator.step()

        if current_override is not None:
            state = StressState_override(current_override)
        else:
            state = fusion.update(
                video=inputs["video"],
                audio=inputs["audio"],
                heartrate=inputs["heartrate"],
            )

        if state.level != prev_level:
            prev_level = state.level
            if audio_gen:
                audio_gen.request_buffer_refill(state.level)
            if visual_gen:
                visual_gen.request_generation({
                    "type": "surface",
                    "level": state.level,
                    "clean_path": "assets/textures/clean/wallpaper.png",
                })
                if state.level >= 3:
                    visual_gen.request_generation({
                        "type": "entity",
                        "level": state.level,
                        "name": "shadow_figure",
                    })

        distorted = camera_fx.apply(frame, state.level)

        display = np.hstack([frame, distorted])
        h, w = display.shape[:2]

        draw_stress_gauge(display, state.level, state.raw_score, state.label,
                          x=w // 2 - 100, y=h - 70)
        draw_modality_panel(display, state, x=w - 220, y=10)

        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_timer = time.time()

        cv2.putText(display, f"FPS: {fps:.0f}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display, f"Scenario: {simulator.scenario}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if current_override is not None:
            cv2.putText(display, f"OVERRIDE: {current_override}", (w // 2 - 60, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(display, "Original", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display, "Distorted", (w // 2 + 10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Adaptive Horror Pipeline", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif ord("0") <= key <= ord("5"):
            current_override = key - ord("0")
            fusion.reset()
            print(f"  Override → level {current_override} ({STRESS_LABELS[current_override]})")
        elif key == ord("r"):
            current_override = None
            fusion.reset()
            print("  Override released — back to simulation")
        elif key == ord("s"):
            scenario_idx = (scenario_idx + 1) % len(scenarios)
            simulator = StressSimulator(scenarios[scenario_idx])
            fusion.reset()
            print(f"  Scenario → {scenarios[scenario_idx]}")

    cap.release()
    cv2.destroyAllWindows()

    if audio_gen:
        audio_gen.stop_background_thread()
        audio_gen.unload()
    if visual_gen:
        visual_gen.stop_background_thread()
        visual_gen.unload()

    print("Pipeline stopped.")


def StressState_override(level):
    """Create a StressState for manual override mode."""
    from fusion import StressState
    return StressState(
        level=level,
        raw_score=float(level),
        confidence=1.0,
        per_modality={"override": {"score": float(level)}},
        label=STRESS_LABELS[level],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive horror pipeline demo")
    parser.add_argument("--simulate", action="store_true", help="Use simulated model outputs")
    parser.add_argument("--stress-override", type=int, help="Force a specific stress level (0-5)")
    parser.add_argument("--scenario", default="calm",
                        choices=["calm", "ramp", "spike", "oscillate", "random"],
                        help="Simulation scenario")
    parser.add_argument("--enable-audio", action="store_true", help="Enable AudioLDM generation")
    parser.add_argument("--enable-visual", action="store_true", help="Enable SD texture generation")
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()

    if args.simulate or args.stress_override is not None:
        run_pipeline(
            scenario=args.scenario,
            stress_override=args.stress_override,
            enable_audio=args.enable_audio,
            enable_visual=args.enable_visual,
        )
    else:
        print("Use --simulate to run with simulated model outputs.")
        print("Use --stress-override N to force a stress level.")
        print("\nExamples:")
        print("  python scripts/run_pipeline.py --simulate")
        print("  python scripts/run_pipeline.py --simulate --scenario ramp")
        print("  python scripts/run_pipeline.py --simulate --stress-override 4")
        print("  python scripts/run_pipeline.py --simulate --enable-audio --enable-visual")
