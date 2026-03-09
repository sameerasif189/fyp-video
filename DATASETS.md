# Video Face Expression Datasets for FYP
## Adaptive Psychological Horror Gaming — Video Model

This repo handles the **video/facial expression model** — one of three models in the FYP:
1. **Video (this repo)** — Facial expression recognition (fear, stress, emotion detection)
2. Audio — Separate repo
3. Heart Rate — Separate repo

### Hardware Budget
| Resource | Available |
|----------|-----------|
| GPU | NVIDIA RTX 5060 (8GB VRAM) |
| RAM | 16GB |
| CPU | Intel i7-12700H |
| Storage | ~100-120GB |

---

## Datasets

### 1. DFEW (Dynamic Facial Expression in the Wild) — PRIMARY
- **What:** 16,372 video clips from movies, 7 basic emotions (including fear, surprise, disgust)
- **Why:** Real-world conditions — extreme lighting, occlusions, pose changes. Best for training robust fear detection
- **Size:** ~8-10GB (extracted frames)
- **Access:** Email authors for password → https://dfew-dataset.github.io/
- **Paper:** https://arxiv.org/abs/2008.05924
- **Priority:** HIGH

### 2. FERV39k — PRIMARY
- **What:** 38,935 video clips, 7 emotions, 22 scenes across 4 scenarios
- **Why:** Largest multi-scene video FER dataset (CVPR 2022), great diversity for training
- **Size:** ~5-8GB
- **Access:** Email Fudan University team → https://wangyanckxx.github.io/Proj_CVPR2022_FERV39k.html
- **GitHub:** https://github.com/wangyanckxx/FERV39k
- **Priority:** HIGH

### 3. FER2013 — BASELINE
- **What:** 35,887 grayscale images (48x48), 7 emotions
- **Why:** Tiny size, instant Kaggle download, good for quick prototyping before video datasets arrive
- **Size:** ~300MB
- **Access:** Kaggle (free) → https://www.kaggle.com/datasets/msambare/fer2013
- **Priority:** HIGH (start here)

### 4. RAF-DB (Real-world Affective Faces Database)
- **What:** 29,672 images, 6 basic + compound expressions
- **Why:** Higher quality than FER2013, real-world images, good for pre-training
- **Size:** ~1-2GB
- **Access:** Request from authors → http://www.whdeng.cn/raf/model1.html
- **Priority:** MEDIUM

### 5. CK+ (Extended Cohn-Kanade)
- **What:** 5,876 images as sequences, 123 subjects, 8 emotions, lab-controlled
- **Why:** Gold standard benchmark, clean labels, good for validation
- **Size:** ~1GB
- **Access:** Request from University of Pittsburgh
- **Priority:** MEDIUM

### 6. Aff-Wild2
- **What:** 564 videos (~2.8M frames), 554 subjects, per-frame annotations for 7 expressions + valence/arousal
- **Why:** Largest in-the-wild video emotion database, continuous annotations
- **Size:** Very large — download fear/surprise/disgust subset only (~15GB)
- **Access:** Email d.kollias@qmul.ac.uk with signed EULA
- **Source:** https://ibug.doc.ic.ac.uk/resources/aff-wild2/
- **Priority:** MEDIUM

### 7. RAVDESS Video
- **What:** 7,356 files, 24 actors, 8 emotions (includes fearful, surprised)
- **Why:** Clean lab data with video, good for validation
- **Size:** ~8GB (video subset)
- **Access:** FREE → https://zenodo.org/record/1188976
- **Priority:** MEDIUM

---

## Storage Budget (~45GB total)

| Dataset | Est. Size | Access |
|---------|-----------|--------|
| FER2013 | ~300MB | Kaggle (instant) |
| RAF-DB | ~2GB | Request |
| CK+ | ~1GB | Request |
| DFEW | ~10GB | Request |
| FERV39k | ~8GB | Request |
| RAVDESS video | ~8GB | Free (Zenodo) |
| Aff-Wild2 subset | ~15GB | Request (EULA) |
| **Total** | **~45GB** | |

Leaves ~55-75GB for processed data, models, and training outputs.

---

## Emotion → Game Stress Class Mapping

All 7 dataset emotion labels are mapped to 4 game classes:

| Game Class | Stress Level | Source Emotions |
|------------|-------------|-----------------|
| **calm** | 0-1 | neutral, calm, happy |
| **stressed** | 2-3 | sad, angry, disgust, contempt |
| **fearful** | 3-4 | fear |
| **panic** | 5 | surprise (startle response) |

---

## Download Order

### Phase 1 — Start now (~8GB)
1. FER2013 from Kaggle (~300MB) — prototype immediately
2. RAVDESS video from Zenodo (~8GB) — free, no approval needed

### Phase 2 — Request access (~20GB)
3. DFEW (~10GB) — email for password
4. FERV39k (~8GB) — email Fudan
5. RAF-DB (~2GB) — request
6. CK+ (~1GB) — request

### Phase 3 — Optional (~15GB)
7. Aff-Wild2 fear subset (~15GB) — email with EULA

---

## Training Tips for RTX 5060 (8GB VRAM)

- Use **mixed precision** (fp16) — halves VRAM usage
- Batch size: **8** for 224x224 video clips (16 frames each)
- Use **gradient accumulation** for effective batch 32-64
- Pre-trained backbone: **ResNet18** or **EfficientNet-B0** (fits in 8GB)
- For video temporal modeling: check [Former-DFER](https://github.com/zengqunzhao/Former-DFER)
