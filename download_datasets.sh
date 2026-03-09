#!/bin/bash
# ============================================================
# FYP Video Face Expression Dataset Download Helper
# Adaptive Psychological Horror Gaming — Video Model
# ============================================================
# Downloads/guides for facial expression recognition datasets
# used to train the fear/emotion detection model
# ============================================================

set -e

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${BASE_DIR}/datasets/facial_expression"

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
    if [ "$available" -lt 10 ]; then
        error "Less than 10GB available. Free up space before downloading."
        exit 1
    fi
}

# ============================================================
# Phase 1: Free / Instant Access (~8GB)
# ============================================================

download_fer2013() {
    local dir="${DATA_DIR}/fer2013"
    if [ -d "$dir" ] && [ "$(find "$dir" -name '*.jpg' -o -name '*.png' -o -name '*.csv' 2>/dev/null | head -1)" ]; then
        warn "FER2013 already exists, skipping."
        return
    fi
    mkdir -p "$dir"
    echo "=== FER2013 (~300MB) ==="
    echo "35,887 grayscale face images, 7 emotions"
    echo ""
    echo "Option A — Kaggle CLI:"
    echo "  pip install kaggle"
    echo "  kaggle datasets download -d msambare/fer2013 -p ${dir}"
    echo "  cd ${dir} && unzip fer2013.zip && rm fer2013.zip"
    echo ""
    echo "Option B — Manual download:"
    echo "  https://www.kaggle.com/datasets/msambare/fer2013"
    echo "  Extract to: ${dir}/"
    echo ""
}

download_ravdess_video() {
    local dir="${DATA_DIR}/ravdess"
    if [ -d "$dir" ] && [ "$(find "$dir" -name '*.mp4' 2>/dev/null | head -1)" ]; then
        warn "RAVDESS Video already exists, skipping."
        return
    fi
    mkdir -p "$dir"
    echo "=== RAVDESS Video (~8GB) — FREE ==="
    echo "7,356 files, 24 actors, 8 emotions (includes fearful)"
    echo ""
    echo "Download video files from Zenodo:"
    for i in $(seq -w 1 24); do
        echo "  wget -P ${dir}/ https://zenodo.org/record/1188976/files/Video_Speech_Actor_${i}.zip"
    done
    echo ""
    echo "Then unzip:"
    echo "  cd ${dir} && for f in *.zip; do unzip \"\$f\"; done && rm *.zip"
    echo ""
}

# ============================================================
# Phase 2: Request Access (~20GB)
# ============================================================

download_dfew() {
    local dir="${DATA_DIR}/dfew"
    mkdir -p "$dir"
    echo "=== DFEW (~10GB) — REQUEST ACCESS ==="
    echo "16,372 video clips from movies, 7 emotions"
    echo "Best dataset for real-world fear detection"
    echo ""
    echo "1. Visit: https://dfew-dataset.github.io/"
    echo "2. Email authors for download password"
    echo "3. Extract to: ${dir}/"
    echo ""
}

download_ferv39k() {
    local dir="${DATA_DIR}/ferv39k"
    mkdir -p "$dir"
    echo "=== FERV39k (~8GB) — REQUEST ACCESS ==="
    echo "38,935 video clips, 7 emotions, 22 scenes (CVPR 2022)"
    echo ""
    echo "1. Visit: https://wangyanckxx.github.io/Proj_CVPR2022_FERV39k.html"
    echo "2. Email Fudan University team for Baidu Drive link"
    echo "3. Extract to: ${dir}/"
    echo ""
}

download_rafdb() {
    local dir="${DATA_DIR}/raf_db"
    mkdir -p "$dir"
    echo "=== RAF-DB (~2GB) — REQUEST ACCESS ==="
    echo "29,672 face images, 6 basic + compound expressions"
    echo ""
    echo "1. Visit: http://www.whdeng.cn/raf/model1.html"
    echo "2. Request download link"
    echo "3. Extract to: ${dir}/"
    echo ""
}

download_ckplus() {
    local dir="${DATA_DIR}/ck_plus"
    mkdir -p "$dir"
    echo "=== CK+ (~1GB) — REQUEST ACCESS ==="
    echo "5,876 images, 123 subjects, 8 emotions (lab-controlled)"
    echo ""
    echo "1. Request from University of Pittsburgh"
    echo "2. Extract to: ${dir}/"
    echo "   Expected structure:"
    echo "     ck_plus/cohn-kanade-images/S005/001/*.png"
    echo "     ck_plus/Emotion/S005/001/*.txt"
    echo ""
}

download_affwild2() {
    local dir="${DATA_DIR}/aff_wild2"
    mkdir -p "$dir"
    echo "=== Aff-Wild2 (~15GB subset) — REQUEST ACCESS ==="
    echo "564 videos, 2.8M frames, per-frame emotion annotations"
    echo ""
    echo "1. Email d.kollias@qmul.ac.uk with signed EULA"
    echo "2. Website: https://ibug.doc.ic.ac.uk/resources/aff-wild2/"
    echo "3. TIP: Request only fear/surprise/disgust subset to save storage"
    echo "4. Extract to: ${dir}/"
    echo ""
}

# ============================================================
# Main Menu
# ============================================================

show_menu() {
    echo ""
    echo "============================================"
    echo "  FYP Video — Face Expression Datasets"
    echo "============================================"
    echo ""
    echo "PHASE 1 — Free (start here):"
    echo "  1) FER2013          (~300MB)  [Kaggle]"
    echo "  2) RAVDESS Video    (~8GB)    [Zenodo — FREE]"
    echo ""
    echo "PHASE 2 — Request access:"
    echo "  3) DFEW             (~10GB)   [Email authors]"
    echo "  4) FERV39k          (~8GB)    [Email Fudan]"
    echo "  5) RAF-DB           (~2GB)    [Request]"
    echo "  6) CK+             (~1GB)    [Request]"
    echo "  7) Aff-Wild2 subset (~15GB)   [Email + EULA]"
    echo ""
    echo "  a) Show ALL instructions"
    echo "  q) Quit"
    echo ""
    read -rp "Select option: " choice

    case $choice in
        1)  download_fer2013 ;;
        2)  download_ravdess_video ;;
        3)  download_dfew ;;
        4)  download_ferv39k ;;
        5)  download_rafdb ;;
        6)  download_ckplus ;;
        7)  download_affwild2 ;;
        a|A)
            check_storage
            echo ""
            echo "========== PHASE 1: FREE =========="
            download_fer2013
            download_ravdess_video
            echo "========== PHASE 2: REQUEST ACCESS =========="
            download_dfew
            download_ferv39k
            download_rafdb
            download_ckplus
            download_affwild2
            ;;
        q|Q) exit 0 ;;
        *)  error "Invalid option" ;;
    esac
}

if [ "$1" = "--all" ]; then
    check_storage
    download_fer2013
    download_ravdess_video
    download_dfew
    download_ferv39k
    download_rafdb
    download_ckplus
    download_affwild2
else
    while true; do
        show_menu
    done
fi
