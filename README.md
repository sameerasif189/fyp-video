# FYP: Video Face Expression Recognition + Fusion & Generation Pipeline
## Adaptive Psychological Horror Gaming — Video Model

Real-time facial expression recognition model for detecting fear, stress, and emotion in players. This is the **video module** of the FYP — one of three models (video, audio, heart rate) that feed into a unified stress classifier.

The fusion + generation pipeline merges all 3 models into a 6-level stress scale and drives real-time horror content generation: corrupted domestic textures, entity UV skins, ambient audio, and camera face distortion effects.

### Team
- Sameer Asif (22I-0493) - Video model training, Unity development
- Immad Shah (22I-0395) - Audio model training, Unity development
- Anam Omar (22I-0610) - Dataset normalization, documentation

**Advisor:** Dr. Muhammad Nouman Noor, NUCES Islamabad

### Repository Structure

```
fyp-video/
├── DATASETS.md                    # Dataset guide with links & storage plan
├── download_datasets.sh           # Download helper for all 7 datasets
├── configs/
│   ├── video_config.yaml          # Preprocessing config & class mappings
│   └── fusion_config.yaml         # Fusion + generation pipeline config
├── scripts/
│   ├── preprocess_video.py        # Dataset preprocessing pipeline
│   ├── fusion.py                  # Multi-modal fusion (3 models → stress 0-5)
│   ├── generate_visual.py         # SD+LoRA domestic horror textures & entity skins
│   ├── generate_audio.py          # AudioLDM domestic horror ambient audio
│   ├── generate_camera_fx.py      # MediaPipe+OpenCV real-time face distortion
│   └── run_pipeline.py            # End-to-end demo pipeline
├── assets/
│   ├── textures/clean/            # Clean household textures (input)
│   ├── textures/cached/           # Generated corrupted textures per level
│   ├── entities/base_models/      # .obj/.fbx 3D meshes from Blender
│   ├── entities/skins/            # Generated entity UV texture maps
│   └── audio/cached/              # Generated ambient audio clips
├── datasets/
│   └── facial_expression/         # DFEW, FERV39k, FER2013, RAF-DB, CK+, Aff-Wild2, RAVDESS
├── requirements.txt
└── FYP Proposal.docx
```

### Quick Start

```bash
pip install -r requirements.txt

# Preprocess video datasets
python scripts/preprocess_video.py --datasets fer2013 dfew ferv39k

# Test fusion with synthetic inputs
python scripts/fusion.py --test --verbose

# Camera effects demo (CPU only, no GPU needed)
python scripts/generate_camera_fx.py --demo

# Pre-generate horror audio clips
python scripts/generate_audio.py --precache --levels 3 4 5

# Pre-generate corrupted textures + entity skins
python scripts/generate_visual.py --precache

# Full pipeline demo (simulated model outputs)
python scripts/run_pipeline.py --simulate --scenario ramp
```

### Multi-Modal Fusion

Three perception models merge into a unified 6-level stress scale:

| Model | Repo | Output |
|-------|------|--------|
| Video (this repo) | `fyp-video` | 4 classes → mapped to 6 levels |
| Audio | `Fyp` | 6 stress levels directly |
| Heart Rate | `FYP-heartrate` | 6 stress levels directly |

Fusion weights: Video 45%, Audio 30%, Heart Rate 25%.

### 6-Level Stress Scale

| Level | Label | Video Class | Game Response |
|-------|-------|-------------|---------------|
| 0 | Calm | calm | Clean environment |
| 1 | Wary | — | Subtle wear, faded textures |
| 2 | Nervous | stressed | Peeling wallpaper, creaking sounds |
| 3 | Anxious | — | Entities appear, whispers |
| 4 | Frightened | fearful | Blood stains, sobbing, dark figures |
| 5 | Terrified | panic | Full horror, screaming, chaos |

### Generation Pipeline

| Asset Type | Method | Real-Time? |
|-----------|--------|------------|
| Surface textures | SD 1.5 + LoRA + ControlNet | Background (2-5s per texture) |
| Entity UV skins | SD 1.5 text2img | Background (2-5s per skin) |
| Ambient audio | AudioLDM | Background (5-10s per clip, buffered) |
| Camera face FX | MediaPipe + OpenCV | Real-time 30+ FPS (CPU only) |

SD and AudioLDM time-share VRAM (8GB constraint — cannot both be loaded).

### Hardware
- GPU: NVIDIA RTX 5060 (8GB VRAM)
- RAM: 16GB
- Storage: ~100GB dedicated to this model

See [DATASETS.md](DATASETS.md) for full dataset details.
