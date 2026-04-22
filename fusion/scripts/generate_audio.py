#!/usr/bin/env python3
"""
Domestic horror ambient audio generation using AudioLDM.

Takes a stress level (0-5) and generates ambient audio clips.
Level 0 = silence/clock ticking, Level 5 = screaming/chaos.

Demo mode generates and saves clips for all levels to disk.

Usage:
  python scripts/generate_audio.py --demo                # all levels, save + play
  python scripts/generate_audio.py --level 3             # single level
  python scripts/generate_audio.py --demo --no-play      # save only, no playback
"""

import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np
import yaml

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import AudioLDMPipeline
except ImportError:
    AudioLDMPipeline = None


def load_config(config_path="configs/generation_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


STRESS_LABELS = {0: "Calm", 1: "Wary", 2: "Nervous", 3: "Anxious", 4: "Frightened", 5: "Terrified"}


class AudioGenerator:
    def __init__(self, config_path="configs/generation_config.yaml"):
        cfg = load_config(config_path)
        self.audio_cfg = cfg["audio_generation"]

        self.scene_context = self.audio_cfg["scene_context"]
        self.prompts = {}
        for level, template in self.audio_cfg["prompts_per_level"].items():
            self.prompts[int(level)] = template.replace("{scene_context}", self.scene_context)

        self.duration = self.audio_cfg["duration_sec"]
        self.sample_rate = self.audio_cfg["sample_rate"]
        self.steps = self.audio_cfg["inference_steps"]
        self.crossfade_sec = self.audio_cfg.get("crossfade_sec", 2.0)
        self.cache_dir = Path(self.audio_cfg.get("cache_dir", "assets/audio/cached"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = None
        self._loaded = False

    def _load_pipeline(self):
        if self._loaded:
            return
        if torch is None or AudioLDMPipeline is None:
            print("NOTE: torch/diffusers not installed. Using algorithmic fallback.")
            return

        print(f"Loading AudioLDM: {self.audio_cfg['model']}...")
        dtype = torch.float16 if self.audio_cfg.get("use_fp16", True) else torch.float32
        self._pipeline = AudioLDMPipeline.from_pretrained(
            self.audio_cfg["model"], torch_dtype=dtype
        )
        if self.audio_cfg.get("cpu_offload", True):
            self._pipeline.enable_model_cpu_offload()
        else:
            self._pipeline = self._pipeline.to("cuda")
        self._loaded = True
        print("AudioLDM loaded.")

    def unload(self):
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._loaded = False
            if torch is not None:
                torch.cuda.empty_cache()

    def generate_clip(self, level):
        """Generate a single audio clip at stress level 0-5. Saves to cache_dir.

        Returns: path to saved .wav file.
        """
        prompt = self.prompts.get(level, self.prompts.get(0))

        if self._pipeline is not None:
            audio = self._pipeline(
                prompt,
                num_inference_steps=self.steps,
                audio_length_in_s=self.duration,
            ).audios[0]
        else:
            audio = self._fallback_audio(level)

        filename = f"level{level}_{int(time.time())}.wav"
        path = self.cache_dir / filename
        self._save_wav(str(path), audio)
        return str(path)

    def _fallback_audio(self, level):
        """Procedural audio fallback when AudioLDM is unavailable."""
        samples = int(self.duration * self.sample_rate)
        t = np.linspace(0, self.duration, samples, dtype=np.float32)
        audio = np.zeros(samples, dtype=np.float32)

        if level == 0:
            # Quiet room tone + clock tick
            audio += np.random.randn(samples) * 0.003
            tick_interval = int(self.sample_rate)
            for i in range(0, samples, tick_interval):
                end = min(i + 200, samples)
                audio[i:end] += np.sin(np.linspace(0, 20 * np.pi, end - i)) * 0.08

        elif level == 1:
            # Ambient + faint wind
            audio += np.random.randn(samples) * 0.005
            wind = np.sin(2 * np.pi * 0.3 * t) * np.random.randn(samples) * 0.01
            audio += wind

        elif level == 2:
            # Creaking + pipe sounds
            audio += np.random.randn(samples) * 0.008
            for _ in range(4):
                pos = np.random.randint(0, samples - self.sample_rate // 2)
                length = np.random.randint(self.sample_rate // 8, self.sample_rate // 3)
                end = min(pos + length, samples)
                freq = np.random.uniform(30, 80)
                creak = np.sin(2 * np.pi * freq * np.linspace(0, length / self.sample_rate, end - pos))
                env = np.linspace(0.05, 0, end - pos)
                audio[pos:end] += creak * env

        elif level == 3:
            # Whispers + dripping
            audio += np.random.randn(samples) * 0.01
            # Dripping
            for _ in range(8):
                pos = np.random.randint(0, samples - 1000)
                drip = np.sin(np.linspace(0, 30 * np.pi, 800)) * np.exp(-np.linspace(0, 5, 800))
                end = min(pos + 800, samples)
                audio[pos:end] += drip[:end - pos] * 0.06
            # Whisper-like noise bursts
            for _ in range(3):
                pos = np.random.randint(0, samples - self.sample_rate)
                length = np.random.randint(self.sample_rate // 2, self.sample_rate)
                end = min(pos + length, samples)
                whisper = np.random.randn(end - pos) * 0.02
                # Bandpass-ish: multiply by sinusoid
                freq = np.random.uniform(200, 400)
                carrier = np.sin(2 * np.pi * freq * np.linspace(0, (end - pos) / self.sample_rate, end - pos))
                env = np.sin(np.linspace(0, np.pi, end - pos))
                audio[pos:end] += whisper * carrier * env

        elif level == 4:
            # Doors, footsteps, sobbing
            audio += np.random.randn(samples) * 0.015
            # Door slam
            for _ in range(2):
                pos = np.random.randint(0, samples - 5000)
                slam = np.random.randn(3000) * 0.15 * np.exp(-np.linspace(0, 8, 3000))
                end = min(pos + 3000, samples)
                audio[pos:end] += slam[:end - pos]
            # Footsteps
            step_interval = int(self.sample_rate * 0.6)
            start = np.random.randint(0, samples // 2)
            for i in range(start, min(start + step_interval * 8, samples), step_interval):
                end = min(i + 1500, samples)
                step = np.random.randn(end - i) * 0.04 * np.exp(-np.linspace(0, 6, end - i))
                audio[i:end] += step
            # Low moan
            moan_pos = np.random.randint(samples // 3, 2 * samples // 3)
            moan_len = min(self.sample_rate * 2, samples - moan_pos)
            moan_t = np.linspace(0, 2, moan_len)
            moan = np.sin(2 * np.pi * 120 * moan_t) * np.sin(np.linspace(0, np.pi, moan_len)) * 0.03
            audio[moan_pos:moan_pos + moan_len] += moan

        elif level == 5:
            # Screaming, pounding, chaos
            audio += np.random.randn(samples) * 0.03
            # Scream
            scream_pos = np.random.randint(0, samples // 2)
            scream_len = min(self.sample_rate * 3, samples - scream_pos)
            scream_t = np.linspace(0, 3, scream_len)
            scream = np.sin(2 * np.pi * 800 * scream_t + 200 * np.sin(2 * np.pi * 5 * scream_t))
            scream *= np.sin(np.linspace(0, np.pi, scream_len)) * 0.12
            audio[scream_pos:scream_pos + scream_len] += scream
            # Pounding
            for _ in range(6):
                pos = np.random.randint(0, samples - 2000)
                pound = np.random.randn(2000) * 0.2 * np.exp(-np.linspace(0, 10, 2000))
                end = min(pos + 2000, samples)
                audio[pos:end] += pound[:end - pos]
            # Glass shatter
            shatter_pos = np.random.randint(samples // 3, 2 * samples // 3)
            shatter_len = min(self.sample_rate, samples - shatter_pos)
            shatter = np.random.randn(shatter_len) * 0.1 * np.exp(-np.linspace(0, 6, shatter_len))
            audio[shatter_pos:shatter_pos + shatter_len] += shatter

        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def _save_wav(self, path, audio):
        if sf is not None:
            sf.write(path, audio, self.sample_rate)
        else:
            with wave.open(path, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.sample_rate)
                audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
                wf.writeframes(audio_int.tobytes())

    def crossfade(self, audio_a, audio_b):
        fade_samples = int(self.crossfade_sec * self.sample_rate)
        fade_samples = min(fade_samples, len(audio_a), len(audio_b))
        result = np.zeros(len(audio_a) + len(audio_b) - fade_samples, dtype=np.float32)
        result[:len(audio_a)] = audio_a
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        overlap = len(audio_a) - fade_samples
        result[overlap:len(audio_a)] *= fade_out
        result[overlap:len(audio_a)] += audio_b[:fade_samples] * fade_in
        result[len(audio_a):] = audio_b[fade_samples:]
        return result

    def run_demo(self, levels=None, play=True):
        """Generate + optionally play one clip per level."""
        if levels is None:
            levels = list(range(6))

        self._load_pipeline()

        print(f"\n{'='*60}")
        print(f"  Audio Generation Demo — Domestic Horror Ambient")
        print(f"{'='*60}")

        paths = {}
        for level in levels:
            label = STRESS_LABELS[level]
            print(f"\n  Level {level} ({label})")
            print(f"  Prompt: {self.prompts[level]}")
            print(f"  Generating {self.duration}s clip...", end=" ", flush=True)
            path = self.generate_clip(level)
            paths[level] = path
            print(f"-> {path}")

            if play:
                try:
                    import sounddevice as sd
                    if sf is not None:
                        data, sr = sf.read(path)
                    else:
                        with wave.open(path, "r") as wf:
                            sr = wf.getframerate()
                            frames = wf.readframes(wf.getnframes())
                            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
                    print(f"  Playing ({len(data)/sr:.1f}s)...")
                    sd.play(data, sr)
                    sd.wait()
                except Exception:
                    print("  (sounddevice unavailable — saved but not played)")

        self.unload()

        print(f"\n{'='*60}")
        print(f"  All clips saved to {self.cache_dir}/")
        print(f"{'='*60}")
        return paths


def main():
    parser = argparse.ArgumentParser(description="Domestic horror audio generation")
    parser.add_argument("--demo", action="store_true", help="Generate all levels")
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4, 5], help="Single stress level")
    parser.add_argument("--no-play", action="store_true", help="Don't play audio, just save")
    parser.add_argument("--config", default="configs/generation_config.yaml")
    args = parser.parse_args()

    if args.demo:
        gen = AudioGenerator(config_path=args.config)
        gen.run_demo(play=not args.no_play)
    elif args.level is not None:
        gen = AudioGenerator(config_path=args.config)
        gen._load_pipeline()
        path = gen.generate_clip(args.level)
        print(f"Level {args.level} ({STRESS_LABELS[args.level]}) -> {path}")
        gen.unload()
    else:
        print("Usage:")
        print("  python scripts/generate_audio.py --demo")
        print("  python scripts/generate_audio.py --level 3")
        print("  python scripts/generate_audio.py --demo --no-play")


if __name__ == "__main__":
    main()
