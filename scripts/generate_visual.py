#!/usr/bin/env python3
"""
Domestic horror visual generation: surface textures + entity UV skins.

Mode A: Surface textures — img2img with ControlNet to corrupt clean household
        textures (walls, floors, furniture) at each stress level.
Mode B: Entity skins — text2img to generate UV texture maps for 3D Blender
        meshes at stress levels 3-5 (entities only appear at high stress).

Uses SD 1.5 + optional LoRA + ControlNet Canny.
Time-shares VRAM with AudioLDM via background thread.

Usage:
  python scripts/generate_visual.py --precache --surface-textures assets/textures/clean/
  python scripts/generate_visual.py --precache --entity-skins
  python scripts/generate_visual.py --precache --levels 3 4 5
"""

import argparse
import os
import queue
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import (
        StableDiffusionImg2ImgPipeline,
        StableDiffusionControlNetPipeline,
        ControlNetModel,
    )
except ImportError:
    StableDiffusionImg2ImgPipeline = None
    StableDiffusionControlNetPipeline = None
    ControlNetModel = None

try:
    from PIL import Image
except ImportError:
    Image = None


def load_config(config_path="configs/fusion_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class LRUCache:
    """Simple LRU cache for generated textures."""

    def __init__(self, maxsize=50):
        self._cache = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)


class VisualGenerator:
    def __init__(self, config_path="configs/fusion_config.yaml"):
        cfg = load_config(config_path)
        self.vis_cfg = cfg["visual_generation"]
        self.vram_cfg = cfg.get("vram", {})

        self.scene_context = self.vis_cfg["scene_context"]
        self.surface_prompts = self.vis_cfg["surface_prompts"]
        self.entity_prompts = self.vis_cfg.get("entity_prompts", {})
        self.negative_prompt = self.vis_cfg["negative_prompt"]
        self.strength_per_level = self.vis_cfg["strength_per_level"]
        self.texture_size = self.vis_cfg["texture_size"]
        self.steps = self.vis_cfg["inference_steps"]

        self.cache_dir = Path(self.vis_cfg.get("cache_dir", "assets/textures/cached"))
        self.entity_skin_dir = Path(self.vis_cfg.get("entity_skin_dir", "assets/entities/skins"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.entity_skin_dir.mkdir(parents=True, exist_ok=True)

        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._loaded = False
        self._lru = LRUCache(self.vis_cfg.get("max_cache_items", 50))
        self._gen_queue = queue.Queue()
        self._gen_thread = None
        self._running = False
        self._vram_lock = threading.Lock()

    def _load_pipeline(self, mode="img2img"):
        if self._loaded:
            return
        if torch is None:
            print("WARNING: torch not installed. Visual gen will use dummy output.")
            return

        dtype = torch.float16 if self.vis_cfg.get("use_fp16", True) else torch.float32
        model_id = self.vis_cfg["model"]

        if mode == "controlnet" and ControlNetModel is not None:
            print(f"Loading ControlNet: {self.vis_cfg['controlnet']}...")
            controlnet = ControlNetModel.from_pretrained(
                self.vis_cfg["controlnet"], torch_dtype=dtype
            )
            self._controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id, controlnet=controlnet, torch_dtype=dtype
            )
            if self.vis_cfg.get("cpu_offload", True):
                self._controlnet_pipe.enable_model_cpu_offload()
            else:
                self._controlnet_pipe = self._controlnet_pipe.to("cuda")

            lora_path = self.vis_cfg.get("lora_path")
            if lora_path and Path(lora_path).exists():
                print(f"Loading LoRA: {lora_path}")
                self._controlnet_pipe.load_lora_weights(lora_path)

            self._loaded = True
            print("SD + ControlNet loaded.")

        elif StableDiffusionImg2ImgPipeline is not None:
            print(f"Loading SD img2img: {model_id}...")
            self._img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                model_id, torch_dtype=dtype
            )
            if self.vis_cfg.get("cpu_offload", True):
                self._img2img_pipe.enable_model_cpu_offload()
            else:
                self._img2img_pipe = self._img2img_pipe.to("cuda")

            lora_path = self.vis_cfg.get("lora_path")
            if lora_path and Path(lora_path).exists():
                print(f"Loading LoRA: {lora_path}")
                self._img2img_pipe.load_lora_weights(lora_path)

            self._loaded = True
            print("SD img2img loaded.")

    def unload(self):
        if self._img2img_pipe is not None:
            del self._img2img_pipe
            self._img2img_pipe = None
        if self._controlnet_pipe is not None:
            del self._controlnet_pipe
            self._controlnet_pipe = None
        self._loaded = False
        if torch is not None:
            torch.cuda.empty_cache()
        print("SD pipelines unloaded from VRAM.")

    def _build_prompt(self, level, mode="surface"):
        """Build full prompt from scene context + level-specific description."""
        if mode == "entity":
            level_prompt = self.entity_prompts.get(level, self.entity_prompts.get(str(level)))
            if level_prompt is None:
                return None
            return f"{level_prompt}, {self.scene_context}"
        else:
            level_prompt = self.surface_prompts.get(level, self.surface_prompts.get(str(level)))
            if level_prompt is None:
                return None
            return f"{self.scene_context}, {level_prompt}"

    def generate_surface_texture(self, clean_image_path, level, save=True):
        """Corrupt a clean household texture at the given stress level.

        Args:
            clean_image_path: path to clean texture image
            level: stress level 0-5
            save: whether to save to cache_dir

        Returns:
            numpy array (BGR) or saved file path
        """
        if level == 0:
            img = cv2.imread(str(clean_image_path))
            return str(clean_image_path) if save else img

        cache_key = f"surface_{Path(clean_image_path).stem}_level_{level}"
        cached = self._lru.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(level, mode="surface")
        if prompt is None:
            img = cv2.imread(str(clean_image_path))
            return str(clean_image_path) if save else img

        strength = self.strength_per_level[min(level, len(self.strength_per_level) - 1)]

        clean_img = cv2.imread(str(clean_image_path))
        clean_img = cv2.resize(clean_img, (self.texture_size, self.texture_size))

        if self._controlnet_pipe is not None and Image is not None:
            edges = cv2.Canny(clean_img, 50, 150)
            control_image = Image.fromarray(edges)
            init_image = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))

            with self._vram_lock:
                result = self._controlnet_pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    image=control_image,
                    num_inference_steps=self.steps,
                    controlnet_conditioning_scale=strength,
                ).images[0]

            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        elif self._img2img_pipe is not None and Image is not None:
            init_image = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
            init_image = init_image.resize((self.texture_size, self.texture_size))

            with self._vram_lock:
                result = self._img2img_pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    image=init_image,
                    strength=strength,
                    num_inference_steps=self.steps,
                ).images[0]

            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        else:
            # Dummy: progressively add noise and darken
            result_np = clean_img.astype(np.float32)
            noise = np.random.randn(*result_np.shape).astype(np.float32) * (level * 15)
            result_np = result_np * (1.0 - strength * 0.3) + noise
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)

        if save:
            out_path = self.cache_dir / f"{cache_key}.png"
            cv2.imwrite(str(out_path), result_np)
            self._lru.put(cache_key, str(out_path))
            return str(out_path)

        self._lru.put(cache_key, result_np)
        return result_np

    def generate_entity_skin(self, level, entity_name="shadow_figure", save=True):
        """Generate a UV texture map for a 3D entity at the given stress level.

        Entities only appear at levels 3-5.

        Args:
            level: stress level (3, 4, or 5)
            entity_name: name for the entity mesh
            save: whether to save to entity_skin_dir

        Returns:
            numpy array (BGR) or saved file path
        """
        if level < 3:
            return None

        cache_key = f"entity_{entity_name}_level_{level}"
        cached = self._lru.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(level, mode="entity")
        if prompt is None:
            return None

        if self._controlnet_pipe is not None and Image is not None:
            blank = Image.new("RGB", (self.texture_size, self.texture_size), (128, 128, 128))

            with self._vram_lock:
                result = self._controlnet_pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    image=blank,
                    num_inference_steps=self.steps,
                ).images[0]

            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        elif self._img2img_pipe is not None and Image is not None:
            blank = Image.new("RGB", (self.texture_size, self.texture_size), (128, 128, 128))

            with self._vram_lock:
                result = self._img2img_pipe(
                    prompt=prompt,
                    negative_prompt=self.negative_prompt,
                    image=blank,
                    strength=0.9,
                    num_inference_steps=self.steps,
                ).images[0]

            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        else:
            # Dummy: dark texture with noise scaled by level
            base_dark = 20 + (level - 3) * 10
            result_np = np.full(
                (self.texture_size, self.texture_size, 3), base_dark, dtype=np.uint8
            )
            noise = np.random.randint(0, 30, result_np.shape, dtype=np.uint8)
            result_np = cv2.add(result_np, noise)

        if save:
            out_path = self.entity_skin_dir / f"{cache_key}.png"
            cv2.imwrite(str(out_path), result_np)
            self._lru.put(cache_key, str(out_path))
            return str(out_path)

        self._lru.put(cache_key, result_np)
        return result_np

    def precache_surfaces(self, clean_dir, levels=None):
        """Pre-generate corrupted textures for all clean images in a directory."""
        clean_dir = Path(clean_dir)
        if not clean_dir.exists():
            print(f"Clean texture directory not found: {clean_dir}")
            print("Place clean household textures (walls, floors, etc.) there first.")
            return

        images = list(clean_dir.glob("*.png")) + list(clean_dir.glob("*.jpg"))
        if not images:
            print(f"No images found in {clean_dir}")
            print("Generating sample clean textures for demo...")
            images = self._generate_sample_clean_textures(clean_dir)

        if levels is None:
            levels = list(range(1, 6))

        self._load_pipeline(mode="controlnet")

        total = len(images) * len(levels)
        count = 0
        for img_path in images:
            for level in levels:
                count += 1
                print(f"  [{count}/{total}] {img_path.name} → level {level}...", end=" ", flush=True)
                path = self.generate_surface_texture(img_path, level, save=True)
                print(f"→ {path}")

        print(f"Surface precache complete. {count} textures generated.")

    def _generate_sample_clean_textures(self, clean_dir):
        """Generate placeholder clean textures for demo purposes."""
        samples = {
            "wallpaper": (180, 170, 150),
            "wooden_floor": (120, 100, 70),
            "bathroom_tile": (200, 200, 210),
            "kitchen_counter": (160, 155, 145),
            "carpet": (140, 120, 100),
        }

        paths = []
        for name, base_color in samples.items():
            img = np.full((self.texture_size, self.texture_size, 3), base_color, dtype=np.uint8)
            noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Add subtle pattern
            for y in range(0, self.texture_size, 32):
                cv2.line(img, (0, y), (self.texture_size, y), tuple(int(c * 0.95) for c in base_color), 1)

            path = clean_dir / f"{name}.png"
            cv2.imwrite(str(path), img)
            paths.append(path)
            print(f"  Created sample: {path}")

        return paths

    def precache_entity_skins(self, entity_names=None, levels=None):
        """Pre-generate entity UV skins for stress levels 3-5."""
        if entity_names is None:
            entity_names = ["shadow_figure", "child_silhouette", "tall_figure"]
        if levels is None:
            levels = [3, 4, 5]

        self._load_pipeline(mode="img2img")

        total = len(entity_names) * len(levels)
        count = 0
        for name in entity_names:
            for level in levels:
                count += 1
                print(f"  [{count}/{total}] {name} → level {level}...", end=" ", flush=True)
                path = self.generate_entity_skin(level, entity_name=name, save=True)
                if path:
                    print(f"→ {path}")
                else:
                    print("skipped (level too low)")

        print(f"Entity skin precache complete. {count} skins generated.")

    def start_background_thread(self):
        """Start background generation thread for runtime texture updates."""
        self._running = True
        self._gen_thread = threading.Thread(target=self._background_worker, daemon=True)
        self._gen_thread.start()
        print("Visual background gen thread started.")

    def stop_background_thread(self):
        self._running = False
        if self._gen_thread:
            self._gen_queue.put(None)
            self._gen_thread.join(timeout=10)
        print("Visual background gen thread stopped.")

    def request_generation(self, task):
        """Queue a generation task: dict with type, level, and source info."""
        self._gen_queue.put(task)

    def _background_worker(self):
        while self._running:
            try:
                task = self._gen_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if task is None:
                break

            try:
                if task.get("type") == "surface":
                    self.generate_surface_texture(
                        task["clean_path"], task["level"], save=True
                    )
                elif task.get("type") == "entity":
                    self.generate_entity_skin(
                        task["level"], entity_name=task.get("name", "shadow_figure"), save=True
                    )
            except Exception as e:
                print(f"Background visual gen error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Domestic horror visual generation")
    parser.add_argument("--precache", action="store_true", help="Pre-generate assets")
    parser.add_argument("--surface-textures", type=str, help="Clean texture directory")
    parser.add_argument("--entity-skins", action="store_true", help="Generate entity UV skins")
    parser.add_argument("--levels", nargs="+", type=int, help="Specific levels (0-5)")
    parser.add_argument("--config", default="configs/fusion_config.yaml")
    args = parser.parse_args()

    if args.precache:
        gen = VisualGenerator(config_path=args.config)

        if args.surface_textures:
            gen.precache_surfaces(args.surface_textures, levels=args.levels)

        if args.entity_skins:
            gen.precache_entity_skins(levels=args.levels)

        if not args.surface_textures and not args.entity_skins:
            print("Generating both surface textures and entity skins...")
            gen.precache_surfaces("assets/textures/clean", levels=args.levels)
            gen.precache_entity_skins(levels=args.levels)

        gen.unload()
    else:
        print("Use --precache to pre-generate assets.")
        print("  --surface-textures <dir>  Corrupt clean textures at each level")
        print("  --entity-skins            Generate entity UV texture maps")
        print("  --levels 3 4 5            Specific stress levels")
        print("\nImport VisualGenerator class for pipeline integration.")


if __name__ == "__main__":
    main()
