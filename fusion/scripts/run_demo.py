#!/usr/bin/env python3
"""
End-to-end generation demo — produces all horror assets for stress levels 0-5.

No game environment needed. Generates and saves:
  - Corrupted surface textures (walls, floors, etc.) per level
  - Entity UV skins (BadShade) for levels 3-5
  - Ambient horror audio clips per level
  - Camera distortion effect samples per level
  - Summary grids for visual review

Usage:
  python scripts/run_demo.py                          # full demo, all generators
  python scripts/run_demo.py --level 4                # single level across all generators
  python scripts/run_demo.py --generators visual audio # specific generators
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml

STRESS_LABELS = {0: "Calm", 1: "Wary", 2: "Nervous", 3: "Anxious", 4: "Frightened", 5: "Terrified"}


def run_visual_demo(levels, config):
    from generate_visual import VisualGenerator
    print(f"\n{'#'*60}")
    print(f"  VISUAL GENERATION — Surface Textures + Entity UV Skins")
    print(f"{'#'*60}")
    gen = VisualGenerator(config_path=config)
    gen.run_demo(levels=levels)
    gen.unload()


def run_audio_demo(levels, config):
    from generate_audio import AudioGenerator
    print(f"\n{'#'*60}")
    print(f"  AUDIO GENERATION — Domestic Horror Ambient")
    print(f"{'#'*60}")
    gen = AudioGenerator(config_path=config)
    gen.run_demo(levels=levels, play=False)
    gen.unload()


def run_camera_fx_demo(levels, config):
    from generate_camera_fx import CameraEffectsGenerator
    print(f"\n{'#'*60}")
    print(f"  CAMERA FX — Face Distortion Effects")
    print(f"{'#'*60}")
    fx = CameraEffectsGenerator(config_path=config)
    fx.run_demo(levels=levels)


def main():
    parser = argparse.ArgumentParser(description="Full generation pipeline demo")
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4, 5], help="Single stress level")
    parser.add_argument("--generators", nargs="+",
                        choices=["visual", "audio", "camera"],
                        default=["visual", "audio", "camera"],
                        help="Which generators to run")
    parser.add_argument("--config", default="configs/generation_config.yaml")
    args = parser.parse_args()

    levels = [args.level] if args.level is not None else list(range(6))

    print("=" * 60)
    print("  Adaptive Psychological Horror — Generation Demo")
    print("=" * 60)
    print(f"  Stress levels: {levels}")
    print(f"  Generators:    {args.generators}")
    print(f"  Stress scale:  0=Calm, 1=Wary, 2=Nervous, 3=Anxious, 4=Frightened, 5=Terrified")
    print("=" * 60)

    start = time.time()

    if "visual" in args.generators:
        run_visual_demo(levels, args.config)

    if "audio" in args.generators:
        run_audio_demo(levels, args.config)

    if "camera" in args.generators:
        run_camera_fx_demo(levels, args.config)

    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"  Demo Complete — {elapsed:.1f}s total")
    print(f"{'='*60}")
    print(f"  Outputs:")
    print(f"    Textures:     assets/textures/cached/")
    print(f"    Entity skins: assets/entities/skins/")
    print(f"    Audio clips:  assets/audio/cached/")
    print(f"    Camera FX:    assets/textures/cached/camera_fx_demo/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
