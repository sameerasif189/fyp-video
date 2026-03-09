#!/bin/bash
# ============================================================
# FYP Video Dataset Download Script
# Adaptive Psychological Horror Gaming
# ============================================================
# Hardware: RTX 5060 (8GB), 16GB RAM, ~100-120GB storage
# This script downloads freely available datasets.
# For gated datasets (DFEW, FERV39k, Aff-Wild2, RAF-DB, CK+),
# you must request access first - see DATASETS.md
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${BASE_DIR}/datasets"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_storage() {
    local available
    available=$(df -BG "$BASE_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    log "Available storage: ${available}GB"
    if [ "$available" -lt 20 ]; then
        error "Less than 20GB available. Free up space before downloading."
        exit 1
    fi
}

# ============================================================
# PHASE 1: Freely Available Datasets (~15GB)
# ============================================================

download_fer2013() {
    local dir="${DATA_DIR}/facial_expression/fer2013"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        warn "FER2013 already exists, skipping."
        return
    fi
    mkdir -p "$dir"
    log "Downloading FER2013..."
    echo "=== FER2013 ==="
    echo "Download manually from: https://www.kaggle.com/datasets/msambare/fer2013"
    echo "You need a Kaggle account. Alternatively use the Kaggle CLI:"
    echo ""
    echo "  pip install kaggle"
    echo "  kaggle datasets download -d msambare/fer2013 -p ${dir}"
    echo "  cd ${dir} && unzip fer2013.zip && rm fer2013.zip"
    echo ""
    echo "Size: ~300MB"
    echo "Place files in: ${dir}/"
    echo ""
}

download_ravdess() {
    local dir="${DATA_DIR}/audio_visual_emotion/ravdess"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        warn "RAVDESS already exists, skipping."
        return
    fi
    mkdir -p "$dir"
    log "Downloading RAVDESS (audio-video subset)..."

    # RAVDESS is freely available on Zenodo
    # Full dataset is ~24.8GB, we download specific parts
    echo "=== RAVDESS ==="
    echo "Free download from Zenodo:"
    echo ""
    echo "  # Audio-Visual Speech (fearful, surprised, etc.)"
    echo "  wget -P ${dir}/ https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    echo "  wget -P ${dir}/ https://zenodo.org/record/1188976/files/Video_Speech_Actor_01.zip"
    echo ""
    echo "  # Download all 24 actors for video:"
    for i in $(seq -w 1 24); do
        echo "  wget -P ${dir}/ https://zenodo.org/record/1188976/files/Video_Speech_Actor_${i}.zip"
    done
    echo ""
    echo "  # Then unzip all:"
    echo "  cd ${dir} && for f in *.zip; do unzip \"\$f\"; done"
    echo ""
    echo "Size: ~8GB (audio-video speech subset)"
    echo ""
}

download_smarty4covid() {
    local dir="${DATA_DIR}/breathing/smarty4covid"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        warn "Smarty4COVID already exists, skipping."
        return
    fi
    mkdir -p "$dir"
    log "Smarty4COVID breathing dataset..."
    echo "=== Smarty4COVID ==="
    echo "Paper: https://www.nature.com/articles/s41597-023-02646-6"
    echo "Contains: 4,665 breathing + 4,676 cough recordings"
    echo "Check the paper's Data Availability section for download link."
    echo "Place files in: ${dir}/"
    echo ""
}

download_icbhi() {
    local dir="${DATA_DIR}/breathing/icbhi_respiratory"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        warn "ICBHI Respiratory already exists, skipping."
        return
    fi
    mkdir -p "$dir"
    log "ICBHI Respiratory Sound Database..."
    echo "=== ICBHI Respiratory Sound Database ==="
    echo "Download from Kaggle:"
    echo ""
    echo "  kaggle datasets download -d vbookshelf/respiratory-sound-database -p ${dir}"
    echo "  cd ${dir} && unzip respiratory-sound-database.zip && rm respiratory-sound-database.zip"
    echo ""
    echo "Size: ~1GB"
    echo ""
}

download_stress_dataset() {
    local dir="${DATA_DIR}/stress_detection/multimodal_stress_2025"
    if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
        warn "Multimodal Stress Dataset already exists, skipping."
        return
    fi
    mkdir -p "$dir"
    log "Multimodal Stress Detection Dataset (2025)..."
    echo "=== Multimodal Stress Detection Dataset ==="
    echo "Paper: https://www.nature.com/articles/s41597-025-05812-0"
    echo "Contains: facial expressions + physiological signals for stress"
    echo "Check the paper's Data Availability section for download link."
    echo "Place files in: ${dir}/"
    echo ""
}

# ============================================================
# PHASE 2: Gated Datasets (Request Access First)
# ============================================================

download_dfew() {
    local dir="${DATA_DIR}/facial_expression/dfew"
    mkdir -p "$dir"
    echo "=== DFEW (Dynamic Facial Expression in the Wild) ==="
    echo "16,372 video clips, 7 emotions"
    echo "REQUEST ACCESS: Email authors at https://dfew-dataset.github.io/"
    echo "Size: ~8-10GB"
    echo "Place files in: ${dir}/"
    echo ""
}

download_ferv39k() {
    local dir="${DATA_DIR}/facial_expression/ferv39k"
    mkdir -p "$dir"
    echo "=== FERV39k ==="
    echo "38,935 video clips, 7 emotions, 22 scenes"
    echo "REQUEST ACCESS: https://wangyanckxx.github.io/Proj_CVPR2022_FERV39k.html"
    echo "Size: ~5-8GB"
    echo "Place files in: ${dir}/"
    echo ""
}

download_rafdb() {
    local dir="${DATA_DIR}/facial_expression/raf_db"
    mkdir -p "$dir"
    echo "=== RAF-DB ==="
    echo "29,672 images, 6 basic + compound expressions"
    echo "REQUEST ACCESS: http://www.whdeng.cn/raf/model1.html"
    echo "Size: ~1-2GB"
    echo "Place files in: ${dir}/"
    echo ""
}

download_ckplus() {
    local dir="${DATA_DIR}/facial_expression/ck_plus"
    mkdir -p "$dir"
    echo "=== CK+ (Extended Cohn-Kanade) ==="
    echo "5,876 images, 123 subjects, 7 emotions"
    echo "REQUEST ACCESS: University of Pittsburgh"
    echo "Size: ~1GB"
    echo "Place files in: ${dir}/"
    echo ""
}

download_affwild2() {
    local dir="${DATA_DIR}/facial_expression/aff_wild2"
    mkdir -p "$dir"
    echo "=== Aff-Wild2 ==="
    echo "564 videos, 2.8M frames, per-frame emotion annotations"
    echo "REQUEST ACCESS: Email d.kollias@qmul.ac.uk with signed EULA"
    echo "Website: https://ibug.doc.ic.ac.uk/resources/aff-wild2/"
    echo "TIP: Request only fear/surprise/disgust subset to save storage"
    echo "Size: ~15GB (subset)"
    echo "Place files in: ${dir}/"
    echo ""
}

# ============================================================
# PHASE 3: Horror Generation Models
# ============================================================

download_horror_models() {
    local dir="${DATA_DIR}/horror_generation"
    mkdir -p "$dir/texture_models"
    mkdir -p "$dir/lora_weights"
    echo "=== Horror Content Generation Models ==="
    echo ""
    echo "1. Texture Hell SD Checkpoint:"
    echo "   https://civitai.com/models/43468/texture-hell"
    echo "   Place in: ${dir}/texture_models/"
    echo ""
    echo "2. Horror Concept Art LoRA:"
    echo "   https://dataloop.ai/library/model/glif-loradex-trainer_001_horror-concept-art_nocaption/"
    echo "   Place in: ${dir}/lora_weights/"
    echo ""
    echo "3. Stable Diffusion 1.5 + ControlNet (for real-time texture gen):"
    echo "   pip install diffusers transformers accelerate"
    echo "   python -c \"from diffusers import StableDiffusionControlNetPipeline; print('Ready')\""
    echo ""
}

# ============================================================
# Main
# ============================================================

show_menu() {
    echo ""
    echo "============================================"
    echo "  FYP Horror Game - Dataset Downloader"
    echo "============================================"
    echo ""
    echo "PHASE 1 - Free Datasets (~15GB):"
    echo "  1) FER2013            (~300MB)  [Kaggle]"
    echo "  2) RAVDESS            (~8GB)    [Zenodo - FREE]"
    echo "  3) Smarty4COVID       (~2GB)    [Nature]"
    echo "  4) ICBHI Respiratory  (~1GB)    [Kaggle]"
    echo "  5) Stress Dataset     (~3-5GB)  [Nature 2025]"
    echo ""
    echo "PHASE 2 - Gated (need to request access):"
    echo "  6) DFEW               (~10GB)"
    echo "  7) FERV39k            (~5-8GB)"
    echo "  8) RAF-DB             (~2GB)"
    echo "  9) CK+               (~1GB)"
    echo "  10) Aff-Wild2 subset  (~15GB)"
    echo ""
    echo "PHASE 3 - Generation Models:"
    echo "  11) Horror texture/LoRA models"
    echo ""
    echo "  a) Show ALL instructions"
    echo "  q) Quit"
    echo ""
    read -rp "Select option: " choice

    case $choice in
        1)  download_fer2013 ;;
        2)  download_ravdess ;;
        3)  download_smarty4covid ;;
        4)  download_icbhi ;;
        5)  download_stress_dataset ;;
        6)  download_dfew ;;
        7)  download_ferv39k ;;
        8)  download_rafdb ;;
        9)  download_ckplus ;;
        10) download_affwild2 ;;
        11) download_horror_models ;;
        a|A)
            check_storage
            echo ""
            echo "========== PHASE 1: FREE DATASETS =========="
            download_fer2013
            download_ravdess
            download_smarty4covid
            download_icbhi
            download_stress_dataset
            echo "========== PHASE 2: GATED DATASETS =========="
            download_dfew
            download_ferv39k
            download_rafdb
            download_ckplus
            download_affwild2
            echo "========== PHASE 3: GENERATION MODELS =========="
            download_horror_models
            ;;
        q|Q) exit 0 ;;
        *)  error "Invalid option" ;;
    esac
}

# Run
if [ "$1" = "--all" ]; then
    check_storage
    download_fer2013
    download_ravdess
    download_smarty4covid
    download_icbhi
    download_stress_dataset
    download_dfew
    download_ferv39k
    download_rafdb
    download_ckplus
    download_affwild2
    download_horror_models
else
    while true; do
        show_menu
    done
fi
