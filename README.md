# FYP: Video Face Expression Recognition
## Adaptive Psychological Horror Gaming — Video Model

Real-time facial expression recognition model for detecting fear, stress, and emotion in players. This is the **video module** of the FYP — one of three models (video, audio, heart rate) that feed into a unified stress classifier.

### Team
- Sameer Asif (22I-0493) - Video model training, Unity development
- Immad Shah (22I-0395) - Audio model training, Unity development
- Anam Omar (22I-0610) - Dataset normalization, documentation

**Advisor:** Dr. Muhammad Nouman Noor, NUCES Islamabad

### Repository Structure

```
fyp-video/
├── DATASETS.md              # Dataset guide with links & storage plan
├── download_datasets.sh     # Download helper for all 7 datasets
├── configs/
│   └── video_config.yaml    # Preprocessing config & class mappings
├── scripts/
│   └── preprocess_video.py  # Preprocessing pipeline
├── datasets/
│   └── facial_expression/   # DFEW, FERV39k, FER2013, RAF-DB, CK+, Aff-Wild2, RAVDESS
├── requirements.txt
└── FYP Proposal.docx
```

### Quick Start

```bash
pip install -r requirements.txt

# See dataset download instructions
./download_datasets.sh

# After downloading datasets, preprocess them
python scripts/preprocess_video.py

# Process specific datasets only
python scripts/preprocess_video.py --datasets fer2013 dfew ferv39k
```

### 4-Class Emotion Mapping

| Game Class | Stress Level | Emotions |
|------------|-------------|----------|
| calm | 0-1 | neutral, calm, happy |
| stressed | 2-3 | sad, angry, disgust |
| fearful | 3-4 | fear |
| panic | 5 | surprise (startle) |

### Hardware
- GPU: NVIDIA RTX 5060 (8GB VRAM)
- RAM: 16GB
- Storage: ~100GB dedicated to this model

See [DATASETS.md](DATASETS.md) for full dataset details.
