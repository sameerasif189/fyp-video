#!/usr/bin/env python3
"""
Domestic horror ambient audio generation using AudioLDM.

Generates text-to-audio clips per stress level with a lookahead buffer pattern:
maintains pre-generated clips so playback never stops during gameplay.

AudioLDM time-shares VRAM with Stable Diffusion (can't both fit in 8GB).

Usage:
  python scripts/generate_audio.py --preview              # generate + play 1 clip per level
  python scripts/generate_audio.py --precache              # pre-generate buffer clips
  python scripts/generate_audio.py --precache --levels 3 4 5
"""

import argparse
import os
import queue
import sys
import threading
import time
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


def load_config(config_path="configs/fusion_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class AudioGenerator:
    def __init__(self, config_path="configs/fusion_config.yaml"):
        cfg = load_config(config_path)
        self.audio_cfg = cfg["audio_generation"]
        self.vram_cfg = cfg.get("vram", {})

        self.scene_context = self.audio_cfg["scene_context"]
        self.prompts = {}
        for level, prompt_template in self.audio_cfg["prompts_per_level"].items():
            self.prompts[int(level)] = prompt_template.replace(
                "{scene_context}", self.scene_context
            )

        self.duration = self.audio_cfg["duration_sec"]
        self.sample_rate = self.audio_cfg["sample_rate"]
        self.steps = self.audio_cfg["inference_steps"]
        self.buffer_size = self.audio_cfg["lookahead_buffer"]
        self.crossfade_sec = self.audio_cfg.get("crossfade_sec", 2.0)
        self.cache_dir = Path(self.audio_cfg.get("cache_dir", "assets/audio/cached"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._pipeline = None
        self._loaded = False
        self._buffer = {i: [] for i in range(6)}
        self._gen_queue = queue.Queue()
        self._gen_thread = None
        self._running = False
        self._vram_lock = threading.Lock()

    def _load_pipeline(self):
        if self._loaded:
            return
        if torch is None or AudioLDMPipeline is None:
            print("WARNING: torch or diffusers not installed. Audio gen will use dummy output.")
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
            print("AudioLDM unloaded from VRAM.")

    def generate_clip(self, level, save=True):
        """Generate a single audio clip for the given stress level.

        Returns: numpy array (samples,) at self.sample_rate, or path if saved.
        """
        prompt = self.prompts.get(level, self.prompts.get(0, "ambient room tone"))

        if self._pipeline is None:
            self._load_pipeline()

        if self._pipeline is not None:
            with self._vram_lock:
                audio = self._pipeline(
                    prompt,
                    num_inference_steps=self.steps,
                    audio_length_in_s=self.duration,
                ).audios[0]
        else:
            # Dummy: generate ambient noise with level-scaled intensity
            samples = int(self.duration * self.sample_rate)
            intensity = 0.01 + level * 0.03
            audio = np.random.randn(samples).astype(np.float32) * intensity

        if save:
            timestamp = int(time.time() * 1000)
            path = self.cache_dir / f"level_{level}_{timestamp}.wav"
            if sf is not None:
                sf.write(str(path), audio, self.sample_rate)
            else:
                self._save_wav_raw(str(path), audio)
            return str(path)

        return audio

    def _save_wav_raw(self, path, audio):
        """Fallback WAV writer without soundfile."""
        import struct
        import wave
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            audio_int = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
            wf.writeframes(audio_int.tobytes())

    def precache(self, levels=None, clips_per_level=None):
        """Pre-generate buffer clips for specified levels."""
        if levels is None:
            levels = list(range(6))
        if clips_per_level is None:
            clips_per_level = self.buffer_size

        total = len(levels) * clips_per_level
        print(f"Pre-caching {total} audio clips ({clips_per_level} per level)...")

        for level in levels:
            for i in range(clips_per_level):
                print(f"  Level {level}, clip {i + 1}/{clips_per_level}...", end=" ", flush=True)
                path = self.generate_clip(level, save=True)
                self._buffer[level].append(path)
                print(f"saved → {path}")

        print(f"Pre-cache complete. {total} clips generated.")

    def get_cached_clip(self, level):
        """Get a cached clip path for the given level, or generate one."""
        # Check for existing cached files
        if not self._buffer[level]:
            cached = sorted(self.cache_dir.glob(f"level_{level}_*.wav"))
            self._buffer[level] = [str(p) for p in cached]

        if self._buffer[level]:
            return self._buffer[level].pop(0)

        return self.generate_clip(level, save=True)

    def crossfade(self, audio_a, audio_b):
        """Crossfade between two audio arrays."""
        fade_samples = int(self.crossfade_sec * self.sample_rate)
        fade_samples = min(fade_samples, len(audio_a), len(audio_b))

        result = np.zeros(len(audio_a) + len(audio_b) - fade_samples, dtype=np.float32)
        result[:len(audio_a)] = audio_a

        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

        overlap_start = len(audio_a) - fade_samples
        result[overlap_start:len(audio_a)] *= fade_out
        result[overlap_start:len(audio_a)] += audio_b[:fade_samples] * fade_in
        result[len(audio_a):] = audio_b[fade_samples:]

        return result

    def start_background_thread(self):
        """Start background generation thread for runtime buffering."""
        self._running = True
        self._gen_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._gen_thread.start()
        print("Audio background gen thread started.")

    def stop_background_thread(self):
        self._running = False
        if self._gen_thread:
            self._gen_queue.put(None)
            self._gen_thread.join(timeout=10)
        print("Audio background gen thread stopped.")

    def request_buffer_refill(self, level):
        """Queue a buffer refill request for the given level."""
        self._gen_queue.put(level)

    def _background_worker(self):
        while self._running:
            try:
                level = self._gen_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if level is None:
                break

            while len(self._buffer[level]) < self.buffer_size:
                if not self._running:
                    break
                try:
                    path = self.generate_clip(level, save=True)
                    self._buffer[level].append(path)
                except Exception as e:
                    print(f"Background audio gen error (level {level}): {e}")
                    break


def run_preview(levels=None):
    """Generate and play one clip per level for preview."""
    print("=" * 60)
    print("  Audio Generation Preview — Domestic Horror Ambient")
    print("=" * 60)

    gen = AudioGenerator()

    if levels is None:
        levels = list(range(6))

    labels = ["Calm", "Wary", "Nervous", "Anxious", "Frightened", "Terrified"]

    for level in levels:
        print(f"\n--- Level {level}: {labels[level]} ---")
        print(f"  Prompt: {gen.prompts[level]}")
        print(f"  Generating {gen.duration}s clip...", flush=True)

        path = gen.generate_clip(level, save=True)
        print(f"  Saved → {path}")

        try:
            import sounddevice as sd
            data, sr = sf.read(path)
            print(f"  Playing... ({len(data)/sr:.1f}s)")
            sd.play(data, sr)
            sd.wait()
        except Exception:
            print("  (sounddevice not available — clip saved but not played)")

    gen.unload()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domestic horror audio generation")
    parser.add_argument("--preview", action="store_true", help="Generate + play 1 clip per level")
    parser.add_argument("--precache", action="store_true", help="Pre-generate buffer clips")
    parser.add_argument("--levels", nargs="+", type=int, help="Specific levels to generate (0-5)")
    parser.add_argument("--clips", type=int, default=None, help="Clips per level for precache")
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()

    if args.preview:
        run_preview(levels=args.levels)
    elif args.precache:
        gen = AudioGenerator(config_path=args.config)
        gen.precache(levels=args.levels, clips_per_level=args.clips)
        gen.unload()
    else:
        print("Use --preview to generate + play clips, or --precache to pre-generate.")
        print("Import AudioGenerator class for pipeline integration.")
