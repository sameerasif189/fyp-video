#!/usr/bin/env python3
"""
Multi-modal fusion: merge video + audio + heart rate → unified stress level 0-5.

Video model outputs 4 classes (calm/stressed/fearful/panic) → mapped to 6-level scale.
Audio and heart rate models output 6 levels directly (Calm..Terrified).

Usage:
  python scripts/fusion.py --test
  python scripts/fusion.py --test --verbose
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml


def load_config(config_path="configs/fusion_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


STRESS_LABELS = ["Calm", "Wary", "Nervous", "Anxious", "Frightened", "Terrified"]


@dataclass
class StressState:
    level: int = 0
    raw_score: float = 0.0
    confidence: float = 0.0
    per_modality: dict = field(default_factory=dict)
    label: str = "Calm"


class StressFusion:
    def __init__(self, config_path="configs/fusion_config.yaml"):
        cfg = load_config(config_path)
        self.fusion_cfg = cfg["fusion"]

        self.weights = self.fusion_cfg["weights"]
        self.video_mapping = self.fusion_cfg["video_class_to_stress"]
        self.alpha = self.fusion_cfg["smoothing"]["alpha"]
        self.max_jump = self.fusion_cfg["smoothing"]["max_jump"]

        self._prev_score = None
        self._prev_level = None

    def _video_to_stress(self, video_probs):
        """Convert 4-class video probabilities to a stress score 0-5.

        video_probs: list of 4 floats [calm, stressed, fearful, panic]
        Returns weighted stress score.
        """
        class_names = ["calm", "stressed", "fearful", "panic"]
        score = 0.0
        for i, name in enumerate(class_names):
            score += video_probs[i] * self.video_mapping[name]
        return score

    def _reweight(self, available):
        """Redistribute weights among available modalities."""
        total = sum(self.weights[m] for m in available)
        if total == 0:
            return {m: 1.0 / len(available) for m in available}
        return {m: self.weights[m] / total for m in available}

    def _ema_smooth(self, raw_score):
        """Exponential moving average with max jump clamping."""
        if self._prev_score is None:
            self._prev_score = raw_score
            return raw_score

        smoothed = self.alpha * raw_score + (1.0 - self.alpha) * self._prev_score

        if self._prev_level is not None:
            new_level = int(round(smoothed))
            jump = new_level - self._prev_level
            if abs(jump) > self.max_jump:
                clamped_level = self._prev_level + self.max_jump * (1 if jump > 0 else -1)
                smoothed = float(clamped_level)

        self._prev_score = smoothed
        return smoothed

    def update(self, video=None, audio=None, heartrate=None):
        """Process new readings from any combination of modalities.

        Args:
            video: list of 4 floats (class probabilities) or None
            audio: int 0-5 (stress level) or None
            heartrate: int 0-5 (stress level) or None

        Returns:
            StressState with fused stress level and metadata.
        """
        available = {}
        per_modality = {}

        if video is not None:
            score = self._video_to_stress(video)
            available["video"] = score
            per_modality["video"] = {"score": score, "raw": video}

        if audio is not None:
            available["audio"] = float(audio)
            per_modality["audio"] = {"score": float(audio)}

        if heartrate is not None:
            available["heartrate"] = float(heartrate)
            per_modality["heartrate"] = {"score": float(heartrate)}

        if not available:
            return StressState()

        weights = self._reweight(list(available.keys()))
        raw_score = sum(weights[m] * available[m] for m in available)
        confidence = sum(self.weights[m] for m in available)

        smoothed = self._ema_smooth(raw_score)
        level = int(round(max(0, min(5, smoothed))))
        self._prev_level = level

        return StressState(
            level=level,
            raw_score=smoothed,
            confidence=confidence,
            per_modality=per_modality,
            label=STRESS_LABELS[level],
        )

    def reset(self):
        self._prev_score = None
        self._prev_level = None


def run_test(verbose=False):
    """Run synthetic test scenarios to verify fusion logic."""
    print("=" * 60)
    print("  Multi-Modal Fusion — Test Mode")
    print("=" * 60)

    fusion = StressFusion()

    scenarios = [
        {
            "name": "All calm",
            "video": [0.9, 0.05, 0.03, 0.02],
            "audio": 0,
            "heartrate": 0,
            "expected_range": (0, 1),
        },
        {
            "name": "Video fearful, audio nervous, HR wary",
            "video": [0.1, 0.1, 0.7, 0.1],
            "audio": 2,
            "heartrate": 1,
            "expected_range": (2, 3),
        },
        {
            "name": "All panic/terrified",
            "video": [0.0, 0.0, 0.1, 0.9],
            "audio": 5,
            "heartrate": 5,
            "expected_range": (4, 5),
        },
        {
            "name": "Video only (audio + HR missing)",
            "video": [0.0, 0.0, 0.0, 1.0],
            "audio": None,
            "heartrate": None,
            "expected_range": (4, 5),
        },
        {
            "name": "Audio only",
            "video": None,
            "audio": 3,
            "heartrate": None,
            "expected_range": (2, 4),
        },
        {
            "name": "No inputs (all missing)",
            "video": None,
            "audio": None,
            "heartrate": None,
            "expected_range": (0, 0),
        },
    ]

    passed = 0
    failed = 0

    for s in scenarios:
        fusion.reset()
        state = fusion.update(
            video=s["video"], audio=s["audio"], heartrate=s["heartrate"]
        )
        lo, hi = s["expected_range"]
        ok = lo <= state.level <= hi

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] {s['name']}")
        print(f"  Level: {state.level} ({state.label}), Score: {state.raw_score:.2f}, Confidence: {state.confidence:.2f}")

        if verbose:
            for mod, data in state.per_modality.items():
                print(f"  {mod}: score={data['score']:.2f}")

        if not ok:
            print(f"  Expected level in [{lo}, {hi}], got {state.level}")

    # EMA smoothing test
    print(f"\n{'=' * 60}")
    print("  EMA Smoothing Test — rapid transitions")
    print("=" * 60)
    fusion.reset()

    sequence = [
        ([0.9, 0.05, 0.03, 0.02], 0, 0),  # calm
        ([0.9, 0.05, 0.03, 0.02], 0, 0),  # calm
        ([0.0, 0.0, 0.1, 0.9], 5, 5),     # sudden panic
        ([0.0, 0.0, 0.1, 0.9], 5, 5),     # stay panic
        ([0.0, 0.0, 0.1, 0.9], 5, 5),     # stay panic
        ([0.9, 0.05, 0.03, 0.02], 0, 0),  # sudden calm
    ]

    prev_level = None
    ema_ok = True
    for i, (v, a, h) in enumerate(sequence):
        state = fusion.update(video=v, audio=a, heartrate=h)
        jump = abs(state.level - prev_level) if prev_level is not None else 0
        jump_ok = jump <= fusion.max_jump or prev_level is None

        if not jump_ok:
            ema_ok = False

        marker = "" if jump_ok else " ← JUMP VIOLATION"
        print(f"  t={i}: level={state.level} ({state.label}), score={state.raw_score:.2f}, jump={jump}{marker}")
        prev_level = state.level

    if ema_ok:
        passed += 1
        print("[PASS] Max jump constraint respected")
    else:
        failed += 1
        print("[FAIL] Max jump constraint violated")

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-modal stress fusion")
    parser.add_argument("--test", action="store_true", help="Run test scenarios")
    parser.add_argument("--verbose", action="store_true", help="Show per-modality details")
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()

    if args.test:
        success = run_test(verbose=args.verbose)
        sys.exit(0 if success else 1)
    else:
        print("Use --test to run synthetic test scenarios.")
        print("Import StressFusion class for pipeline integration.")
