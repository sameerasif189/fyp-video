# Video & Multi-Modal Datasets for FYP
## "Adaptive Psychological Horror Gaming: A Multi-Modal Emotion-Responsive Experience"

### Hardware Budget
| Resource | Available |
|----------|-----------|
| GPU | NVIDIA RTX 5060 (8GB VRAM) |
| RAM | 16GB |
| CPU | Intel i7-12700H |
| Storage | ~100-120GB |

### Storage Allocation Plan (~95GB used, ~25GB headroom)

| Category | Datasets | Est. Size |
|----------|----------|-----------|
| Facial Expression (Video) | DFEW + FERV39k | ~15GB |
| Facial Expression (Image) | FER2013 + RAF-DB + CK+ | ~3GB |
| Audio-Visual Emotion | RAVDESS (audio-video subset) | ~8GB |
| In-the-Wild Emotion Video | Aff-Wild2 (subset) | ~15GB |
| Stress Detection | Multimodal Stress Dataset (2025) | ~5GB |
| Breathing/Respiratory | Smarty4COVID + ICBHI | ~3GB |
| Horror Textures/LoRA | Civitai Texture Hell + Horror LoRA weights | ~5GB |
| Pre-trained Models | SD ControlNet, MiniXception, MediaPipe | ~10GB |
| **Total Estimated** | | **~64GB** |

---

## Module 1: Facial Expression Recognition (Fear/Stress Detection)

### 1.1 Video-Based Datasets (PRIMARY - for dynamic expression)

#### DFEW (Dynamic Facial Expression in the Wild)
- **What:** 16,372 video clips from movies, 7 basic emotions (including fear, surprise, disgust)
- **Why:** Real-world conditions - extreme lighting, occlusions, pose changes
- **Size:** ~8-10GB (extracted frames)
- **Access:** Email authors for password → https://dfew-dataset.github.io/
- **Paper:** https://arxiv.org/abs/2008.05924
- **Priority:** HIGH

#### FERV39k
- **What:** 38,935 video clips, 7 emotions, 22 scenes across 4 scenarios
- **Why:** Largest multi-scene video FER dataset (CVPR 2022), great diversity
- **Size:** ~5-8GB
- **Access:** Email Fudan University team → https://wangyanckxx.github.io/Proj_CVPR2022_FERV39k.html
- **GitHub:** https://github.com/wangyanckxx/FERV39k
- **Priority:** HIGH

### 1.2 Image-Based Datasets (SUPPLEMENTARY - for pre-training)

#### FER2013
- **What:** 35,887 grayscale images (48x48), 7 emotions
- **Why:** Standard benchmark, tiny size, good for quick prototyping
- **Size:** ~90-300MB
- **Access:** Kaggle (free) → https://www.kaggle.com/datasets/msambare/fer2013
- **Priority:** HIGH (start here for prototyping)

#### RAF-DB (Real-world Affective Faces Database)
- **What:** 29,672 images, 6 basic + compound expressions
- **Why:** Higher quality than FER2013, real-world images
- **Size:** ~1-2GB
- **Access:** Request from authors → http://www.whdeng.cn/raf/model1.html
- **Priority:** MEDIUM

#### CK+ (Extended Cohn-Kanade)
- **What:** 5,876 images, 123 subjects, 7 emotions, lab-controlled
- **Why:** Gold standard benchmark, clean labels, good for validation
- **Size:** ~1GB
- **Access:** Request from University of Pittsburgh
- **Priority:** MEDIUM

### 1.3 In-the-Wild Video Emotion

#### Aff-Wild2
- **What:** 564 videos (~2.8M frames), 554 subjects, per-frame annotations
- **Why:** Largest in-the-wild video database with valence/arousal + expression + action unit labels
- **Size:** Full dataset is very large; USE A SUBSET (~15GB)
- **Access:** Email d.kollias@qmul.ac.uk with signed EULA
- **Source:** https://ibug.doc.ic.ac.uk/resources/aff-wild2/
- **Priority:** MEDIUM (use fear/surprise/disgust subset only)

#### RAVDESS (Ryerson Audio-Visual Database)
- **What:** 7,356 files, 24 actors, audio + video, 8 emotions (includes fearful)
- **Why:** Multi-modal (audio + video), clean lab data, great for biometric fusion training
- **Size:** ~24.8GB full; USE AUDIO-VIDEO SUBSET (~8GB)
- **Access:** FREE → https://zenodo.org/record/1188976
- **Priority:** HIGH (free, multi-modal, includes fear)

---

## Module 2: Breathing & Respiratory Pattern Detection

#### Smarty4COVID
- **What:** 4,665 breathing recordings + 4,676 cough recordings via mobile devices
- **Why:** Breathing pattern classification (inhalation/exhalation), respiratory rate extraction
- **Size:** ~1-2GB
- **Access:** Free → https://www.nature.com/articles/s41597-023-02646-6
- **Priority:** HIGH

#### ICBHI Respiratory Sound Database
- **What:** 920 recordings from 126 patients, 5.5 hours of respiratory sounds
- **Why:** Annotated breathing patterns (crackles, wheezes, clean breathing)
- **Size:** ~1GB
- **Access:** Kaggle → https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
- **Priority:** MEDIUM

#### Google AudioSet - Breathing Ontology
- **What:** Breathing-related audio segments from YouTube
- **Why:** Large-scale breathing audio for pre-training
- **Size:** Variable (download specific breathing segments)
- **Access:** https://research.google.com/audioset/ontology/breathing_1.html
- **Priority:** LOW (use as augmentation)

#### Thermal Breathing Dataset (Prof. Youngjun Cho)
- **What:** Thermal imaging video of breathing at 3 speeds (10/15/30 bpm)
- **Why:** Video-based breathing detection (non-contact)
- **Size:** ~1GB
- **Access:** https://youngjuncho.com/datasets/
- **Priority:** LOW (if you add thermal camera support)

---

## Module 3: Stress Detection (Multi-Modal)

#### Multimodal Stress Detection Dataset (2025)
- **What:** Facial expressions + physiological signals for stress classification
- **Why:** Directly relevant - combines facial expression with stress labels
- **Size:** ~3-5GB
- **Access:** https://www.nature.com/articles/s41597-025-05812-0
- **Priority:** HIGH

#### Emognition Dataset
- **What:** 10 emotions including fear, surprise, anger + physiological signals
- **Why:** Multi-modal with dimensional emotion scales
- **Size:** ~2-3GB
- **Access:** https://www.nature.com/articles/s41598-024-65276-x
- **Priority:** MEDIUM

---

## Module 4: Horror Content Generation (Textures, Audio, Visual)

#### Texture Hell (Stable Diffusion Checkpoint)
- **What:** SD model trained for texture generation (PBR pipeline)
- **Why:** Generate horror game textures (walls, floors, corrupted objects)
- **Size:** ~2-4GB (model weights)
- **Access:** https://civitai.com/models/43468/texture-hell
- **Priority:** MEDIUM

#### Horror Concept Art LoRA
- **What:** LoRA fine-tuned for horror character/environment concept art
- **Why:** Generate WeirdFace/BadShade entity textures
- **Size:** ~100-500MB
- **Access:** https://dataloop.ai/library/model/glif-loradex-trainer_001_horror-concept-art_nocaption/
- **Priority:** MEDIUM

#### Gore Diffusion LoRA (Research Reference)
- **What:** Research on fine-tuning SD for horror/gore content
- **Why:** Reference for training methodology
- **Access:** https://arxiv.org/html/2403.08812v1
- **Priority:** LOW (reference only)

---

## Pre-trained Models to Download

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| MiniXception | Facial expression classification | ~10MB | keras/tensorflow |
| MediaPipe Face Mesh | Face landmark detection | ~5MB | Google MediaPipe |
| OpenCV Haar Cascades | Face detection | ~1MB | OpenCV |
| SD 1.5 + ControlNet | Horror texture generation | ~5-8GB | HuggingFace |
| AudioLDM | Horror audio generation | ~2-3GB | HuggingFace |

---

## Recommended Download Order (by priority)

### Phase 1 - Start Immediately (~15GB)
1. FER2013 (Kaggle, instant download, ~300MB)
2. RAVDESS audio-video subset (Zenodo, free, ~8GB)
3. Smarty4COVID breathing data (~2GB)
4. ICBHI respiratory sounds (Kaggle, ~1GB)
5. Multimodal Stress Dataset 2025 (~3-5GB)

### Phase 2 - Request Access (~25GB)
6. DFEW (email for password, ~10GB)
7. FERV39k (email Fudan, ~5-8GB)
8. RAF-DB (request, ~2GB)
9. CK+ (request, ~1GB)

### Phase 3 - Selective Download (~20GB)
10. Aff-Wild2 fear/surprise subset (~15GB)
11. Horror texture models from Civitai (~5GB)

### Phase 4 - Generation Models (~10GB)
12. Stable Diffusion 1.5 + ControlNet (~8GB)
13. AudioLDM (~2-3GB)

---

## Training Tips for Your Hardware

### RTX 5060 (8GB VRAM)
- Use **mixed precision training** (fp16) to halve VRAM usage
- Batch size: 8-16 for image models, 4-8 for video models
- Use **gradient accumulation** (effective batch = 32-64)
- For Stable Diffusion fine-tuning: use **LoRA** (not full fine-tune)
- Video models: process 8-16 frames per clip max

### Storage Management
- Keep raw datasets on external drive if possible
- Store only preprocessed/extracted frames on SSD
- Use `.gitignore` for all dataset folders (already configured)

### Recommended Frameworks
- **PyTorch** with `torch.cuda.amp` for mixed precision
- **timm** for pre-trained vision backbones
- **Former-DFER** for video expression recognition (uses DFEW/FERV39k)
  - GitHub: https://github.com/zengqunzhao/Former-DFER
