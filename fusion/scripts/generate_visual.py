#!/usr/bin/env python3
"""
Domestic horror visual generation: surface textures + entity UV skins.

Takes a stress level (0-5) and generates:
  Surface textures: Corrupted household textures (walls, floors, furniture)
                    via SD 1.5 img2img + ControlNet Canny
  Entity UV skins:  Texture maps for 3D Blender meshes (levels 3-5 only)
                    via SD 1.5 text2img — BadShade class

Demo mode generates and saves all outputs to disk for review.

Usage:
  python scripts/generate_visual.py --demo                     # all levels, saves to disk
  python scripts/generate_visual.py --level 3                  # single level
  python scripts/generate_visual.py --entity-skins --level 4   # entity skins only
"""

import argparse
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


def load_config(config_path="configs/generation_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


STRESS_LABELS = {0: "Calm", 1: "Wary", 2: "Nervous", 3: "Anxious", 4: "Frightened", 5: "Terrified"}


class LRUCache:
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
    def __init__(self, config_path="configs/generation_config.yaml"):
        cfg = load_config(config_path)
        self.vis_cfg = cfg["visual_generation"]

        self.scene_context = self.vis_cfg["scene_context"]
        self.surface_prompts = {int(k): v for k, v in self.vis_cfg["surface_prompts"].items()}
        self.entity_prompts = {int(k): v for k, v in self.vis_cfg.get("entity_prompts", {}).items()}
        self.negative_prompt = self.vis_cfg["negative_prompt"]
        self.strength_map = {int(k): v for k, v in self.vis_cfg["strength_per_level"].items()}
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
        self._vram_lock = threading.Lock()

    def _load_pipeline(self, mode="img2img"):
        if self._loaded:
            return
        if torch is None:
            print("NOTE: torch not installed. Using algorithmic fallback (no SD).")
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
                self._img2img_pipe.load_lora_weights(lora_path)

            self._loaded = True
            print("SD img2img loaded.")

    def unload(self):
        for pipe in [self._img2img_pipe, self._controlnet_pipe]:
            if pipe is not None:
                del pipe
        self._img2img_pipe = None
        self._controlnet_pipe = None
        self._loaded = False
        if torch is not None:
            torch.cuda.empty_cache()

    def _build_prompt(self, level, mode="surface"):
        if mode == "entity":
            p = self.entity_prompts.get(level)
            return f"{p}, {self.scene_context}" if p else None
        else:
            p = self.surface_prompts.get(level)
            return f"{self.scene_context}, {p}" if p else None

    def generate_surface_texture(self, clean_image_path, level):
        """Corrupt a clean household texture at stress level 0-5. Saves to cache_dir."""
        if level == 0:
            return str(clean_image_path)

        cache_key = f"surface_{Path(clean_image_path).stem}_level{level}"
        cached = self._lru.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(level, mode="surface")
        strength = self.strength_map.get(level, 0.3)

        clean_img = cv2.imread(str(clean_image_path))
        if clean_img is None:
            print(f"  ERROR: Cannot read {clean_image_path}")
            return None
        clean_img = cv2.resize(clean_img, (self.texture_size, self.texture_size))

        if self._controlnet_pipe is not None and Image is not None:
            edges = cv2.Canny(clean_img, 50, 150)
            control_image = Image.fromarray(edges)
            with self._vram_lock:
                result = self._controlnet_pipe(
                    prompt=prompt, negative_prompt=self.negative_prompt,
                    image=control_image, num_inference_steps=self.steps,
                    controlnet_conditioning_scale=strength,
                ).images[0]
            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        elif self._img2img_pipe is not None and Image is not None:
            init_image = Image.fromarray(cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB))
            with self._vram_lock:
                result = self._img2img_pipe(
                    prompt=prompt, negative_prompt=self.negative_prompt,
                    image=init_image, strength=strength,
                    num_inference_steps=self.steps,
                ).images[0]
            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        else:
            result_np = self._fallback_corrupt(clean_img, level, strength)

        out_path = self.cache_dir / f"{cache_key}.png"
        cv2.imwrite(str(out_path), result_np)
        self._lru.put(cache_key, str(out_path))
        return str(out_path)

    def generate_entity_skin(self, level, entity_name="shadow_figure"):
        """Generate a UV texture map for a 3D entity. Levels 3-5 only. Saves to entity_skin_dir."""
        if level < 3:
            return None

        cache_key = f"entity_{entity_name}_level{level}"
        cached = self._lru.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(level, mode="entity")
        if prompt is None:
            return None

        if self._img2img_pipe is not None and Image is not None:
            blank = Image.new("RGB", (self.texture_size, self.texture_size), (128, 128, 128))
            with self._vram_lock:
                result = self._img2img_pipe(
                    prompt=prompt, negative_prompt=self.negative_prompt,
                    image=blank, strength=0.9, num_inference_steps=self.steps,
                ).images[0]
            result_np = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        else:
            result_np = self._fallback_entity_skin(level)

        out_path = self.entity_skin_dir / f"{cache_key}.png"
        cv2.imwrite(str(out_path), result_np)
        self._lru.put(cache_key, str(out_path))
        return str(out_path)

    # ── Algorithmic fallbacks (no GPU / no SD) ──────────────────────────────

    def _fallback_corrupt(self, clean_img, level, strength):
        """Procedural texture corruption when SD is unavailable."""
        result = clean_img.astype(np.float32)
        h, w = result.shape[:2]

        result *= (1.0 - strength * 0.4)

        noise = np.random.randn(h, w, 3).astype(np.float32) * (level * 12)
        result += noise

        if level >= 2:
            shift = np.zeros_like(result)
            shift[:, :, 1] += level * 2
            shift[:, :, 0] -= level * 4
            result += shift

        if level >= 3:
            for _ in range(level * 3):
                x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
                angle = np.random.uniform(0, 2 * np.pi)
                length = np.random.randint(30, 100)
                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))
                color = tuple(int(c) for c in [np.random.randint(10, 40)] * 3)
                thickness = np.random.randint(1, 3)
                result_u8 = np.clip(result, 0, 255).astype(np.uint8)
                cv2.line(result_u8, (x1, y1), (x2, y2), color, thickness)
                result = result_u8.astype(np.float32)

        if level >= 4:
            for _ in range(level * 2):
                cx, cy = np.random.randint(0, w), np.random.randint(0, h)
                radius = np.random.randint(15, 50)
                Y, X = np.ogrid[:h, :w]
                mask = ((X - cx) ** 2 + (Y - cy) ** 2) < radius ** 2
                stain_color = np.array([10, 10, np.random.randint(40, 90)], dtype=np.float32)
                result[mask] = result[mask] * 0.3 + stain_color * 0.7

        if level >= 5:
            for _ in range(5):
                cx, cy = np.random.randint(30, w - 30), np.random.randint(30, h - 30)
                length = np.random.randint(20, 60)
                for finger in range(4):
                    angle = np.random.uniform(-0.5, 0.5) + np.pi * 0.5
                    ex = int(cx + length * np.cos(angle) + finger * 12)
                    ey = int(cy + length * np.sin(angle))
                    result_u8 = np.clip(result, 0, 255).astype(np.uint8)
                    cv2.line(result_u8, (cx + finger * 12, cy), (ex, ey), (15, 15, 70), 3)
                    result = result_u8.astype(np.float32)

        return np.clip(result, 0, 255).astype(np.uint8)

    def _fallback_entity_skin(self, level):
        """Procedural entity UV skin when SD is unavailable."""
        size = self.texture_size
        base_val = max(5, 20 - level * 3)
        skin = np.full((size, size, 3), base_val, dtype=np.float32)

        for scale in [4, 8, 16, 32]:
            small = np.random.randn(size // scale, size // scale, 3).astype(np.float32)
            large = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
            skin += large * (8 / scale) * level

        for _ in range(8 + level * 5):
            pts = []
            x, y = np.random.randint(0, size), np.random.randint(0, size)
            for _ in range(np.random.randint(5, 20)):
                x = max(0, min(size - 1, x + np.random.randint(-15, 15)))
                y = max(0, min(size - 1, y + np.random.randint(-15, 15)))
                pts.append([x, y])
            pts = np.array(pts, dtype=np.int32)
            vein_color = (max(2, 8 - level), max(2, 5 - level), max(2, 30 - level * 5))
            skin_u8 = np.clip(skin, 0, 255).astype(np.uint8)
            cv2.polylines(skin_u8, [pts], False, vein_color, 1, cv2.LINE_AA)
            skin = skin_u8.astype(np.float32)

        if level >= 5:
            for _ in range(np.random.randint(2, 5)):
                cx = np.random.randint(80, size - 80)
                cy = np.random.randint(80, size - 80)
                skin_u8 = np.clip(skin, 0, 255).astype(np.uint8)
                cv2.ellipse(skin_u8, (cx, cy), (12, 7), np.random.randint(0, 180),
                            0, 360, (60, 60, 60), -1)
                cv2.circle(skin_u8, (cx, cy), 4, (5, 5, 5), -1)
                skin = skin_u8.astype(np.float32)

        skin[:, :, 2] += level * 5

        return np.clip(skin, 0, 255).astype(np.uint8)

    # ── Sample clean textures ───────────────────────────────────────────────

    def generate_sample_clean_textures(self, clean_dir=None):
        """Generate placeholder clean household textures for demo."""
        if clean_dir is None:
            clean_dir = Path("assets/textures/clean")
        clean_dir = Path(clean_dir)
        clean_dir.mkdir(parents=True, exist_ok=True)

        samples = {
            "wallpaper": (180, 170, 150),
            "wooden_floor": (120, 100, 70),
            "bathroom_tile": (200, 200, 210),
            "kitchen_counter": (160, 155, 145),
            "carpet": (140, 120, 100),
        }

        paths = []
        for name, base_color in samples.items():
            size = self.texture_size
            img = np.full((size, size, 3), base_color, dtype=np.uint8)
            noise = np.random.randint(-10, 10, img.shape, dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            for y in range(0, size, 32):
                cv2.line(img, (0, y), (size, y), tuple(int(c * 0.93) for c in base_color), 1)
            for x in range(0, size, 32):
                cv2.line(img, (x, 0), (x, size), tuple(int(c * 0.95) for c in base_color), 1)

            path = clean_dir / f"{name}.png"
            cv2.imwrite(str(path), img)
            paths.append(path)
            print(f"  Created clean: {path}")

        return paths

    # ── Demo ────────────────────────────────────────────────────────────────

    def run_demo(self, levels=None, clean_dir=None, entity_names=None):
        """Generate all outputs for demo review (no game env needed)."""
        if levels is None:
            levels = list(range(6))
        if entity_names is None:
            entity_names = ["shadow_figure", "child_silhouette", "tall_figure"]

        if clean_dir is None:
            clean_dir = Path("assets/textures/clean")
        clean_dir = Path(clean_dir)
        images = list(clean_dir.glob("*.png")) + list(clean_dir.glob("*.jpg"))
        if not images:
            print("No clean textures found — generating samples...")
            images = self.generate_sample_clean_textures(clean_dir)

        self._load_pipeline(mode="controlnet")

        # Surface textures
        gen_levels = [l for l in levels if l > 0]
        if gen_levels:
            print(f"\n{'='*60}")
            print(f"  Surface Textures (WeirdFace + Anxiety)")
            print(f"{'='*60}")
            total = len(images) * len(gen_levels)
            count = 0
            for img_path in images:
                for level in gen_levels:
                    count += 1
                    label = STRESS_LABELS[level]
                    print(f"  [{count}/{total}] {img_path.name} @ {level} ({label})...", end=" ", flush=True)
                    path = self.generate_surface_texture(img_path, level)
                    print(f"-> {Path(path).name}")

        # Entity skins
        entity_levels = [l for l in levels if l >= 3]
        if entity_levels:
            print(f"\n{'='*60}")
            print(f"  Entity UV Skins (BadShade)")
            print(f"{'='*60}")
            total = len(entity_names) * len(entity_levels)
            count = 0
            for name in entity_names:
                for level in entity_levels:
                    count += 1
                    label = STRESS_LABELS[level]
                    print(f"  [{count}/{total}] {name} @ {level} ({label})...", end=" ", flush=True)
                    path = self.generate_entity_skin(level, entity_name=name)
                    if path:
                        print(f"-> {Path(path).name}")
                    else:
                        print("skipped")

        # Summary grid
        self._save_summary_grid(images, levels, entity_names, entity_levels)

        print(f"\nOutputs saved to:")
        print(f"  Textures:     {self.cache_dir}/")
        print(f"  Entity skins: {self.entity_skin_dir}/")

    def _save_summary_grid(self, clean_images, levels, entity_names, entity_levels):
        """Save a visual summary grid showing progression across stress levels."""
        # Texture grid: rows = textures, cols = levels
        if clean_images and levels:
            gen_levels = [l for l in levels if l > 0]
            cols = [0] + gen_levels
            cell_size = 128
            grid_w = len(cols) * cell_size
            grid_h = len(clean_images) * cell_size
            grid = np.zeros((grid_h + 40, grid_w, 3), dtype=np.uint8)

            for ci, col_level in enumerate(cols):
                label = STRESS_LABELS[col_level]
                cv2.putText(grid, f"{col_level}:{label}", (ci * cell_size + 5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            for ri, img_path in enumerate(clean_images):
                for ci, col_level in enumerate(cols):
                    if col_level == 0:
                        img = cv2.imread(str(img_path))
                    else:
                        cache_key = f"surface_{img_path.stem}_level{col_level}"
                        cached_path = self.cache_dir / f"{cache_key}.png"
                        img = cv2.imread(str(cached_path)) if cached_path.exists() else None

                    if img is not None:
                        thumb = cv2.resize(img, (cell_size, cell_size))
                        y = ri * cell_size + 40
                        x = ci * cell_size
                        grid[y:y + cell_size, x:x + cell_size] = thumb

            grid_path = self.cache_dir / "summary_textures.png"
            cv2.imwrite(str(grid_path), grid)
            print(f"\n  Texture summary grid -> {grid_path}")

        # Entity grid
        if entity_names and entity_levels:
            cell_size = 128
            grid_w = len(entity_levels) * cell_size
            grid_h = len(entity_names) * cell_size
            grid = np.zeros((grid_h + 40, grid_w, 3), dtype=np.uint8)

            for ci, level in enumerate(entity_levels):
                label = STRESS_LABELS[level]
                cv2.putText(grid, f"{level}:{label}", (ci * cell_size + 5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            for ri, name in enumerate(entity_names):
                for ci, level in enumerate(entity_levels):
                    cache_key = f"entity_{name}_level{level}"
                    skin_path = self.entity_skin_dir / f"{cache_key}.png"
                    img = cv2.imread(str(skin_path)) if skin_path.exists() else None
                    if img is not None:
                        thumb = cv2.resize(img, (cell_size, cell_size))
                        y = ri * cell_size + 40
                        x = ci * cell_size
                        grid[y:y + cell_size, x:x + cell_size] = thumb

            grid_path = self.entity_skin_dir / "summary_entities.png"
            cv2.imwrite(str(grid_path), grid)
            print(f"  Entity summary grid  -> {grid_path}")


def main():
    parser = argparse.ArgumentParser(description="Domestic horror visual generation")
    parser.add_argument("--demo", action="store_true", help="Generate all outputs for review")
    parser.add_argument("--level", type=int, choices=[0, 1, 2, 3, 4, 5], help="Single stress level")
    parser.add_argument("--surface-textures", type=str, default=None, help="Clean texture directory")
    parser.add_argument("--entity-skins", action="store_true", help="Generate entity UV skins only")
    parser.add_argument("--entity-names", nargs="+", default=None)
    parser.add_argument("--config", default="configs/generation_config.yaml")
    args = parser.parse_args()

    gen = VisualGenerator(config_path=args.config)

    if args.demo:
        gen.run_demo(clean_dir=args.surface_textures, entity_names=args.entity_names)
    elif args.level is not None:
        if args.entity_skins:
            gen._load_pipeline(mode="img2img")
            names = args.entity_names or ["shadow_figure", "child_silhouette", "tall_figure"]
            for name in names:
                path = gen.generate_entity_skin(args.level, entity_name=name)
                if path:
                    print(f"  {name} @ level {args.level} -> {path}")
                else:
                    print(f"  {name} @ level {args.level}: entities only at levels 3-5")
        else:
            gen.run_demo(levels=[args.level], clean_dir=args.surface_textures)
    else:
        gen.run_demo(clean_dir=args.surface_textures, entity_names=args.entity_names)

    gen.unload()


if __name__ == "__main__":
    main()
