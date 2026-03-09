# FYP: Adaptive Psychological Horror Gaming
## A Multi-Modal Emotion-Responsive Experience

A first-person psychological horror game that adapts to the player's emotional state using real-time biometric monitoring (facial expression, breathing, heart rate) and generative AI.

### Team
- Sameer Asif (22I-0493) - Video model training, Unity development
- Immad Shah (22I-0395) - Audio model training, Unity development
- Anam Omar (22I-0610) - Dataset normalization, documentation

**Advisor:** Dr. Muhammad Nouman Noor, NUCES Islamabad

### Repository Structure

```
fyp-video/
├── DATASETS.md              # Complete dataset guide with links & storage plan
├── download_datasets.sh     # Interactive dataset download helper
├── datasets/                # All datasets (git-ignored)
│   ├── facial_expression/   # DFEW, FERV39k, FER2013, RAF-DB, CK+, Aff-Wild2
│   ├── audio_visual_emotion/# RAVDESS
│   ├── breathing/           # Smarty4COVID, ICBHI
│   ├── stress_detection/    # Multimodal Stress Dataset 2025
│   └── horror_generation/   # Texture models, LoRA weights
└── FYP Proposal.docx        # Full project proposal
```

### Quick Start

```bash
# View dataset download instructions
./download_datasets.sh

# Show all instructions at once
./download_datasets.sh --all
```

### Hardware Requirements
- GPU: NVIDIA RTX 5060 (8GB VRAM)
- RAM: 16GB
- Storage: ~100-120GB (~64GB for datasets, rest for code/models)

See [DATASETS.md](DATASETS.md) for the full dataset guide and storage allocation plan.
