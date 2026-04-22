# FYP: Generative Content Pipeline
## Adaptive Psychological Horror Gaming — Fusion & Generation

Generative content pipeline for the adaptive horror FYP. Takes a **stress level (0-5)** as input and generates domestic horror assets: corrupted textures, 3D entity UV skins, ambient audio, and real-time camera face distortion.

This pipeline is **separate from the 3 perception models** (video, audio, heart rate). It consumes their unified stress output and drives in-game horror generation.

> **Note:** Currently housed in `fyp-video/fusion/` for convenience. Move to its own repo (`fyp-fusion`) when ready. Run all commands from inside the `fusion/` directory.

### Team
- Sameer Asif (22I-0493) - Video model training, Unity development
- Immad Shah (22I-0395) - Audio model training, Unity development
- Anam Omar (22I-0610) - Dataset normalization, documentation

**Advisor:** Dr. Muhammad Nouman Noor, NUCES Islamabad

### Stress Scale (from FYP Proposal)

| Level | Label | Game Response |
|-------|-------|---------------|
| 0 | Calm | Clean environment, silence |
| 1 | Wary | Subtle wear, faint wind |
| 2 | Nervous | Peeling wallpaper, creaking, pipes |
| 3 | Anxious | Entities appear (BadShade), whispers, scratches |
| 4 | Frightened | Blood stains, sobbing, dark figures, door slams |
| 5 | Terrified | Full horror — screaming, gore, glitch, chaos |

### Repository Structure

```
fyp-fusion/
├── configs/
│   └── generation_config.yaml     # All generation settings & prompts
├── scripts/
│   ├── generate_visual.py         # SD+ControlNet: textures + entity UV skins
│   ├── generate_audio.py          # AudioLDM: domestic horror ambient
│   ├── generate_camera_fx.py      # MediaPipe+OpenCV: face distortion (CPU)
│   └── run_demo.py                # Full demo — all generators
├── assets/
│   ├── textures/clean/            # Clean household textures (input)
│   ├── textures/cached/           # Generated corrupted textures per level
│   ├── entities/base_models/      # .obj/.fbx 3D meshes from Blender
│   ├── entities/skins/            # Generated entity UV texture maps
│   └── audio/cached/              # Generated ambient audio clips
└── requirements.txt
```

### Quick Start

```bash
pip install -r requirements.txt

# Full demo — generates all assets for levels 0-5
python scripts/run_demo.py

# Individual generators
python scripts/generate_visual.py --demo
python scripts/generate_audio.py --demo
python scripts/generate_camera_fx.py --demo

# Single stress level
python scripts/run_demo.py --level 4

# Specific generators only
python scripts/run_demo.py --generators visual audio
```

### Generation Pipeline

| Generator | What it produces | Method | Speed |
|-----------|-----------------|--------|-------|
| `generate_visual.py` | Corrupted wall/floor/furniture textures | SD 1.5 + ControlNet + LoRA | ~2-5s per 512x512 |
| `generate_visual.py` | Entity UV skins (levels 3-5) | SD 1.5 text2img | ~2-5s per 512x512 |
| `generate_audio.py` | Ambient horror audio clips | AudioLDM | ~5-10s per 10s clip |
| `generate_camera_fx.py` | Face distortion effects | MediaPipe + OpenCV (CPU) | Real-time 30+ FPS |

SD and AudioLDM **time-share VRAM** — they alternate on the 8GB GPU, never loaded simultaneously.

### Proposal Class Mapping

| Proposal Class | What changes | Generator |
|---------------|-------------|-----------|
| **WeirdFace** | Object/surface corruption (blood, decay, scratches) | `generate_visual.py` (surfaces) |
| **Anxiety** | Ambient audio, lighting, atmosphere | `generate_audio.py` |
| **BadShade** | Entity appearances at high stress | `generate_visual.py` (entity skins) |
| **Feedback camera** | Player's webcam feed distortion | `generate_camera_fx.py` |

### Hardware
- GPU: NVIDIA RTX 5060 (8GB VRAM)
- RAM: 16GB
- CPU: Intel i7-12700H

### Related Repos
- [`fyp-video`](https://github.com/sameerasif189/fyp-video) — Facial expression recognition (4 classes)
- [`Fyp`](https://github.com/sameerasif189/Fyp) — Audio stress classification (6 levels)
- [`FYP-heartrate`](https://github.com/sameerasif189/FYP-heartrate) — Heart rate stress classification (6 levels)
