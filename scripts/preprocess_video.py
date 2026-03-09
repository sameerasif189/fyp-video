#!/usr/bin/env python3
"""
Preprocess raw video face expression datasets into a unified format for training.

This script:
  1. Scans all raw video/image dataset directories
  2. Maps each sample to one of 4 classes: calm, stressed, fearful, panic
  3. Detects and crops faces from video frames
  4. Samples a fixed number of frames per clip (temporal uniformity)
  5. Resizes and normalizes frames
  6. Saves processed frame sequences + labels as a single dataset
  7. Creates train/val/test splits

Supported datasets:
  - DFEW       (video clips, 7 emotions)
  - FERV39k    (video clips, 7 emotions)
  - FER2013    (images, 7 emotions)
  - RAF-DB     (images, 7 emotions)
  - CK+        (image sequences, 8 emotions)
  - Aff-Wild2  (video, per-frame annotations)
  - RAVDESS    (video, 8 emotions)

Usage:
  python scripts/preprocess_video.py
  python scripts/preprocess_video.py --config configs/video_config.yaml
  python scripts/preprocess_video.py --datasets dfew ferv39k fer2013
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def load_config(config_path="configs/video_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Face Detection
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """Unified face detector supporting multiple backends."""

    def __init__(self, backend="mediapipe", min_face_size=48, margin=0.3):
        self.backend = backend
        self.min_face_size = min_face_size
        self.margin = margin
        self._detector = None
        self._init_detector()

    def _init_detector(self):
        if self.backend == "mediapipe":
            try:
                import mediapipe as mp
                self._detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5
                )
            except ImportError:
                print("  mediapipe not installed, falling back to opencv")
                self.backend = "opencv"
                self._init_detector()

        elif self.backend == "opencv":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._detector = cv2.CascadeClassifier(cascade_path)

        elif self.backend == "dlib":
            try:
                import dlib
                self._detector = dlib.get_frontal_face_detector()
            except ImportError:
                print("  dlib not installed, falling back to opencv")
                self.backend = "opencv"
                self._init_detector()

    def detect(self, frame):
        """
        Detect the largest face in a frame.

        Returns:
            (x, y, w, h) bounding box with margin, or None if no face found.
        """
        h, w = frame.shape[:2]

        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._detector.process(rgb)
            if not results.detections:
                return None
            # Take the detection with highest confidence
            best = max(results.detections, key=lambda d: d.score[0])
            bbox = best.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

        elif self.backend == "opencv":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            if len(faces) == 0:
                return None
            # Take the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x1, y1, bw, bh = faces[0]

        elif self.backend == "dlib":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dets = self._detector(gray, 1)
            if len(dets) == 0:
                return None
            # Take the largest
            best = max(dets, key=lambda d: (d.right() - d.left()) * (d.bottom() - d.top()))
            x1 = best.left()
            y1 = best.top()
            bw = best.right() - best.left()
            bh = best.bottom() - best.top()
        else:
            return None

        # Skip tiny faces
        if bw < self.min_face_size or bh < self.min_face_size:
            return None

        # Apply margin
        mx = int(bw * self.margin)
        my = int(bh * self.margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(w, x1 + bw + 2 * mx)
        y2 = min(h, y1 + bh + 2 * my)

        return (x1, y1, x2 - x1, y2 - y1)

    def close(self):
        if self.backend == "mediapipe" and self._detector:
            self._detector.close()


# ─────────────────────────────────────────────────────────────────────────────
# Video / Frame Processing Utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames_from_video(video_path, num_frames=16, sampling="uniform"):
    """
    Extract a fixed number of frames from a video file.

    Args:
        video_path: path to video file (.mp4, .avi, etc.)
        num_frames: how many frames to sample
        sampling: 'uniform' (evenly spaced), 'random', or 'center'

    Returns:
        list of BGR frames (numpy arrays), or None on failure
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    # Determine which frame indices to sample
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    elif sampling == "uniform":
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()
    elif sampling == "random":
        indices = sorted(np.random.choice(total_frames, num_frames, replace=False).tolist())
    elif sampling == "center":
        center = total_frames // 2
        half = num_frames // 2
        start = max(0, center - half)
        indices = list(range(start, min(start + num_frames, total_frames)))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()

    # Pad with last frame if we got fewer than num_frames
    if len(frames) == 0:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


def extract_frames_from_image_sequence(image_dir, num_frames=16, sampling="uniform"):
    """
    Extract frames from a directory of ordered images (e.g., CK+, Aff-Wild2).

    Args:
        image_dir: directory containing sequential images
        num_frames: how many frames to sample

    Returns:
        list of BGR frames
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted([
        f for f in Path(image_dir).iterdir()
        if f.suffix.lower() in extensions
    ])

    if not images:
        return None

    total = len(images)
    if total <= num_frames:
        indices = list(range(total))
    elif sampling == "uniform":
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    else:
        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        frame = cv2.imread(str(images[idx]))
        if frame is not None:
            frames.append(frame)

    if len(frames) == 0:
        return None
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return frames[:num_frames]


def process_frames(frames, face_detector, frame_size=224, detect_face=True):
    """
    Process a list of frames: detect face, crop, resize.

    Args:
        frames: list of BGR frames
        face_detector: FaceDetector instance (or None)
        frame_size: output size (square)
        detect_face: whether to run face detection

    Returns:
        numpy array of shape (num_frames, frame_size, frame_size, 3), dtype uint8
    """
    processed = []
    last_bbox = None

    for frame in frames:
        if detect_face and face_detector:
            bbox = face_detector.detect(frame)
            if bbox is not None:
                last_bbox = bbox
            # Use last known bbox if detection fails on this frame
            if last_bbox is not None:
                x, y, w, h = last_bbox
                face = frame[y:y+h, x:x+w]
            else:
                face = frame
        else:
            face = frame

        # Resize to target size
        if face.size == 0:
            face = frame
        face = cv2.resize(face, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        processed.append(face)

    return np.array(processed, dtype=np.uint8)


def single_image_to_sequence(image_path, num_frames=16):
    """
    Convert a single image (e.g., FER2013) into a repeated frame sequence.
    This allows image datasets to use the same pipeline as video datasets.
    """
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    # If grayscale (FER2013), convert to BGR
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return [frame.copy() for _ in range(num_frames)]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_dfew(raw_dir, class_mapping):
    """
    Parse DFEW dataset.
    Expected structure:
      dfew/
        clip_0001/         # or single_video_0001.avi
          0001.jpg
          0002.jpg
          ...
      OR
      dfew/
        Clip/
          Happy/, Sad/, Neutral/, Angry/, Surprise/, Disgust/, Fear/
            clip_0001.avi
    Also supports extracted frames layout:
      dfew/
        frames/
          00001/0001.jpg, 0002.jpg, ...
        label.csv   (clip_id, emotion_label)
    """
    dfew_dir = Path(raw_dir) / "dfew"
    if not dfew_dir.exists():
        print("  DFEW directory not found, skipping.")
        return []

    entries = []

    # Strategy 1: label CSV + frames directory
    label_csv = None
    for candidate in ["label.csv", "labels.csv", "annotation.csv",
                       "DFEW_label.csv", "set_0.csv"]:
        p = dfew_dir / candidate
        if p.exists():
            label_csv = p
            break

    if label_csv:
        df = pd.read_csv(label_csv, header=None)
        # Expect columns: clip_id (or video name), emotion_label
        for _, row in df.iterrows():
            clip_id = str(row.iloc[0]).strip()
            emotion = str(row.iloc[1]).strip()
            game_class = class_mapping.get(emotion)
            if not game_class:
                continue

            # Try to find clip as video or frame directory
            clip_path = None
            for ext in [".avi", ".mp4", ".mov"]:
                p = dfew_dir / f"{clip_id}{ext}"
                if p.exists():
                    clip_path = p
                    break
            if clip_path is None:
                # Try frames directory
                for frames_parent in [dfew_dir / "frames", dfew_dir]:
                    p = frames_parent / clip_id
                    if p.is_dir():
                        clip_path = p
                        break

            if clip_path:
                entries.append({
                    "path": str(clip_path),
                    "label": game_class,
                    "source": "dfew",
                    "type": "sequence" if clip_path.is_dir() else "video",
                })

    # Strategy 2: Emotion-named subdirectories with video files
    emotion_dirs = {
        "Happy": "0", "Sad": "1", "Neutral": "2", "Angry": "3",
        "Surprise": "4", "Disgust": "5", "Fear": "6",
        "happy": "0", "sad": "1", "neutral": "2", "angry": "3",
        "surprise": "4", "disgust": "5", "fear": "6",
    }
    for emo_name, emo_code in emotion_dirs.items():
        emo_dir = dfew_dir / emo_name
        if not emo_dir.exists():
            # Check inside Clip/ subfolder
            emo_dir = dfew_dir / "Clip" / emo_name
        if not emo_dir.exists():
            continue
        game_class = class_mapping.get(emo_code)
        if not game_class:
            continue
        for vid in emo_dir.rglob("*"):
            if vid.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
                entries.append({
                    "path": str(vid),
                    "label": game_class,
                    "source": "dfew",
                    "type": "video",
                })

    # Strategy 3: Numbered subdirectories (extracted frames) without CSV
    if not entries:
        frames_dir = dfew_dir / "frames" if (dfew_dir / "frames").exists() else dfew_dir
        for subdir in sorted(frames_dir.iterdir()):
            if subdir.is_dir() and any(f.suffix.lower() in {".jpg", ".png"} for f in subdir.iterdir()):
                # Without labels, we can't map — skip
                pass

    print(f"  DFEW: {len(entries)} clips parsed")
    return entries


def parse_ferv39k(raw_dir, class_mapping):
    """
    Parse FERV39k dataset.
    Expected structure:
      ferv39k/
        4 folders by scenario, each with subfolders by scene containing .mp4 clips
      OR
      ferv39k/
        clips/
          clip_00001.mp4
        label.csv  (clip_name, emotion_label)
    """
    ferv_dir = Path(raw_dir) / "ferv39k"
    if not ferv_dir.exists():
        print("  FERV39k directory not found, skipping.")
        return []

    entries = []

    # Strategy 1: Label CSV
    label_csv = None
    for candidate in ["label.csv", "labels.csv", "annotation.csv",
                       "FERV39k_label.csv", "all_label.csv"]:
        p = ferv_dir / candidate
        if p.exists():
            label_csv = p
            break

    if label_csv:
        df = pd.read_csv(label_csv, header=None)
        for _, row in df.iterrows():
            clip_name = str(row.iloc[0]).strip()
            emotion = str(row.iloc[1]).strip()
            game_class = class_mapping.get(emotion)
            if not game_class:
                continue
            # Search for the video file
            clip_path = None
            for vid in ferv_dir.rglob(f"*{clip_name}*"):
                if vid.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
                    clip_path = vid
                    break
            if clip_path:
                entries.append({
                    "path": str(clip_path),
                    "label": game_class,
                    "source": "ferv39k",
                    "type": "video",
                })

    # Strategy 2: Emotion-named directories
    emotion_names = {
        "Angry": "0", "Disgust": "1", "Fear": "2", "Happy": "3",
        "Neutral": "4", "Sad": "5", "Surprise": "6",
        "angry": "0", "disgust": "1", "fear": "2", "happy": "3",
        "neutral": "4", "sad": "5", "surprise": "6",
    }
    for emo_name, emo_code in emotion_names.items():
        for emo_dir in ferv_dir.rglob(emo_name):
            if not emo_dir.is_dir():
                continue
            game_class = class_mapping.get(emo_code)
            if not game_class:
                continue
            for vid in emo_dir.iterdir():
                if vid.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
                    entries.append({
                        "path": str(vid),
                        "label": game_class,
                        "source": "ferv39k",
                        "type": "video",
                    })

    print(f"  FERV39k: {len(entries)} clips parsed")
    return entries


def parse_fer2013(raw_dir, class_mapping):
    """
    Parse FER2013 dataset.
    Expected structure:
      fer2013/
        train/angry/  train/disgust/  train/fear/ ...
        test/angry/   test/disgust/   ...
      OR
      fer2013/
        fer2013.csv   (emotion, pixels, Usage)
    """
    fer_dir = Path(raw_dir) / "fer2013"
    if not fer_dir.exists():
        print("  FER2013 directory not found, skipping.")
        return []

    entries = []
    emotion_names = {
        "angry": "0", "disgust": "1", "fear": "2", "happy": "3",
        "sad": "4", "surprise": "5", "neutral": "6",
    }

    # Strategy 1: Image folder structure (Kaggle format)
    for split in ["train", "test", "validation", "val"]:
        split_dir = fer_dir / split
        if not split_dir.exists():
            continue
        for emo_dir in split_dir.iterdir():
            if not emo_dir.is_dir():
                continue
            emo_name = emo_dir.name.lower()
            emo_code = emotion_names.get(emo_name)
            if emo_code is None:
                continue
            game_class = class_mapping.get(emo_code)
            if not game_class:
                continue
            for img in emo_dir.iterdir():
                if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    entries.append({
                        "path": str(img),
                        "label": game_class,
                        "source": "fer2013",
                        "type": "image",
                    })

    # Strategy 2: CSV format (original FER2013)
    csv_path = fer_dir / "fer2013.csv"
    if csv_path.exists() and not entries:
        df = pd.read_csv(csv_path)
        temp_dir = fer_dir / "_extracted_frames"
        temp_dir.mkdir(exist_ok=True)
        for idx, row in df.iterrows():
            emotion = str(row["emotion"])
            game_class = class_mapping.get(emotion)
            if not game_class:
                continue
            # Convert pixel string to image and save
            pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            img_path = temp_dir / f"{idx:06d}_e{emotion}.png"
            if not img_path.exists():
                cv2.imwrite(str(img_path), pixels)
            entries.append({
                "path": str(img_path),
                "label": game_class,
                "source": "fer2013",
                "type": "image",
            })

    print(f"  FER2013: {len(entries)} images parsed")
    return entries


def parse_rafdb(raw_dir, class_mapping):
    """
    Parse RAF-DB dataset.
    Expected structure:
      raf_db/
        basic/Image/aligned/   (aligned face images: train_00001_aligned.jpg)
        basic/EmoLabel/list_patition_label.txt  (image_name label)
      OR
      raf_db/
        1/ 2/ 3/ 4/ 5/ 6/ 7/   (emotion-numbered subdirectories)
    """
    raf_dir = Path(raw_dir) / "raf_db"
    if not raf_dir.exists():
        print("  RAF-DB directory not found, skipping.")
        return []

    entries = []

    # Strategy 1: Official structure with label file
    label_file = raf_dir / "basic" / "EmoLabel" / "list_patition_label.txt"
    image_dir = raf_dir / "basic" / "Image" / "aligned"
    if not image_dir.exists():
        image_dir = raf_dir / "basic" / "Image" / "original"

    if label_file.exists() and image_dir.exists():
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                img_name = parts[0]
                emotion = parts[1]
                game_class = class_mapping.get(emotion)
                if not game_class:
                    continue
                # RAF-DB aligned images have _aligned suffix
                stem = Path(img_name).stem
                img_path = None
                for suffix in ["_aligned.jpg", ".jpg", "_aligned.png", ".png"]:
                    candidate = image_dir / f"{stem}{suffix}"
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path:
                    entries.append({
                        "path": str(img_path),
                        "label": game_class,
                        "source": "raf_db",
                        "type": "image",
                    })

    # Strategy 2: Numbered subdirectories
    if not entries:
        for emo_code in class_mapping:
            emo_dir = raf_dir / str(emo_code)
            if not emo_dir.is_dir():
                continue
            game_class = class_mapping[emo_code]
            for img in emo_dir.rglob("*"):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    entries.append({
                        "path": str(img),
                        "label": game_class,
                        "source": "raf_db",
                        "type": "image",
                    })

    print(f"  RAF-DB: {len(entries)} images parsed")
    return entries


def parse_ckplus(raw_dir, class_mapping):
    """
    Parse CK+ (Extended Cohn-Kanade) dataset.
    Expected structure:
      ck_plus/
        cohn-kanade-images/S005/001/  (image sequences)
        Emotion/S005/001/S005_001_00000011_emotion.txt  (label file, last frame only)
    """
    ck_dir = Path(raw_dir) / "ck_plus"
    if not ck_dir.exists():
        print("  CK+ directory not found, skipping.")
        return []

    entries = []

    images_dir = ck_dir / "cohn-kanade-images"
    emotion_dir = ck_dir / "Emotion"

    if not images_dir.exists():
        # Flat structure: try to find image sequences directly
        images_dir = ck_dir
    if not emotion_dir.exists():
        emotion_dir = ck_dir / "emotion"
        if not emotion_dir.exists():
            emotion_dir = ck_dir / "labels"

    if emotion_dir.exists():
        # Parse emotion label files
        for label_file in emotion_dir.rglob("*.txt"):
            try:
                with open(label_file, "r") as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    emotion_code = str(int(float(content)))
            except (ValueError, IOError):
                continue

            game_class = class_mapping.get(emotion_code)
            if not game_class:
                continue

            # Find corresponding image sequence
            # Label path: Emotion/S005/001/... -> Images: cohn-kanade-images/S005/001/
            rel_parts = label_file.relative_to(emotion_dir).parts
            if len(rel_parts) >= 2:
                subject = rel_parts[0]
                sequence = rel_parts[1]
                seq_dir = images_dir / subject / sequence
                if seq_dir.is_dir():
                    entries.append({
                        "path": str(seq_dir),
                        "label": game_class,
                        "source": "ck_plus",
                        "type": "sequence",
                    })
    else:
        # Fallback: emotion-named directories
        for emo_name, emo_code in [
            ("neutral", "0"), ("anger", "1"), ("contempt", "2"),
            ("disgust", "3"), ("fear", "4"), ("happy", "5"),
            ("sadness", "6"), ("surprise", "7"),
        ]:
            emo_dir = ck_dir / emo_name
            if not emo_dir.exists():
                continue
            game_class = class_mapping.get(emo_code)
            if not game_class:
                continue
            for img in emo_dir.rglob("*"):
                if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    entries.append({
                        "path": str(img),
                        "label": game_class,
                        "source": "ck_plus",
                        "type": "image",
                    })

    print(f"  CK+: {len(entries)} sequences/images parsed")
    return entries


def parse_affwild2(raw_dir, class_mapping):
    """
    Parse Aff-Wild2 dataset.
    Expected structure:
      aff_wild2/
        videos/                    (video files .mp4/.avi)
        annotations/EXPR_Set/
          Train_Set/video_name.txt  (per-frame expression labels, one per line)
          Validation_Set/video_name.txt
    """
    aw_dir = Path(raw_dir) / "aff_wild2"
    if not aw_dir.exists():
        print("  Aff-Wild2 directory not found, skipping.")
        return []

    entries = []
    video_dir = aw_dir / "videos"
    if not video_dir.exists():
        video_dir = aw_dir

    # Find annotation files
    annot_dir = aw_dir / "annotations" / "EXPR_Set"
    if not annot_dir.exists():
        annot_dir = aw_dir / "annotations"

    if not annot_dir.exists():
        # No annotations — try emotion-named folders
        for emo_code, game_class in class_mapping.items():
            for vid in aw_dir.rglob("*"):
                if vid.suffix.lower() in {".avi", ".mp4"} and emo_code in vid.parent.name:
                    entries.append({
                        "path": str(vid),
                        "label": game_class,
                        "source": "aff_wild2",
                        "type": "video",
                    })
        print(f"  Aff-Wild2: {len(entries)} videos parsed")
        return entries

    # Process annotation files: each file corresponds to a video
    for split in ["Train_Set", "Validation_Set", "train", "val"]:
        split_dir = annot_dir / split
        if not split_dir.exists():
            continue
        for annot_file in split_dir.glob("*.txt"):
            video_name = annot_file.stem
            # Find matching video
            vid_path = None
            for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                candidate = video_dir / f"{video_name}{ext}"
                if candidate.exists():
                    vid_path = candidate
                    break
            if vid_path is None:
                continue

            # Read per-frame annotations and find dominant emotion
            try:
                with open(annot_file, "r") as f:
                    labels = []
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("Neutral"):
                            continue  # skip header
                        try:
                            label = int(line)
                            if label >= 0:  # -1 means unlabeled
                                labels.append(label)
                        except ValueError:
                            continue
            except IOError:
                continue

            if not labels:
                continue

            # Use majority emotion for the whole video
            label_counts = Counter(labels)
            dominant = str(label_counts.most_common(1)[0][0])
            game_class = class_mapping.get(dominant)
            if game_class:
                entries.append({
                    "path": str(vid_path),
                    "label": game_class,
                    "source": "aff_wild2",
                    "type": "video",
                })

    print(f"  Aff-Wild2: {len(entries)} videos parsed")
    return entries


def parse_ravdess_video(raw_dir, class_mapping):
    """
    Parse RAVDESS video files.
    Filename format: XX-XX-EE-XX-XX-XX-XX.mp4
    where EE (position 3, index 2) is the emotion code (01-08).

    Expected structure:
      ravdess/
        Actor_01/
          01-01-01-01-01-01-01.mp4
        Actor_02/
          ...
    """
    # Check both possible locations
    ravdess_dir = None
    for candidate in [
        Path(raw_dir) / "ravdess",
        Path(raw_dir).parent / "audio_visual_emotion" / "ravdess",
    ]:
        if candidate.exists():
            ravdess_dir = candidate
            break

    if ravdess_dir is None:
        print("  RAVDESS Video directory not found, skipping.")
        return []

    entries = []
    for vid_file in ravdess_dir.rglob("*.mp4"):
        name = vid_file.stem
        parts = name.split("-")
        if len(parts) < 3:
            continue
        emotion_code = parts[2]  # 01-08
        game_class = class_mapping.get(emotion_code)
        if game_class:
            entries.append({
                "path": str(vid_file),
                "label": game_class,
                "source": "ravdess_video",
                "type": "video",
            })

    print(f"  RAVDESS Video: {len(entries)} clips parsed")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Main Processing Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def collect_all_entries(raw_dir, config, selected_datasets=None):
    """Collect file paths and labels from all video/image datasets."""
    classes = config["classes"]
    all_entries = []

    parsers = [
        ("dfew", parse_dfew, "dfew_mapping"),
        ("ferv39k", parse_ferv39k, "ferv39k_mapping"),
        ("fer2013", parse_fer2013, "fer2013_mapping"),
        ("raf_db", parse_rafdb, "rafdb_mapping"),
        ("ck_plus", parse_ckplus, "ckplus_mapping"),
        ("aff_wild2", parse_affwild2, "affwild2_mapping"),
        ("ravdess", parse_ravdess_video, "ravdess_video_mapping"),
    ]

    print("\nParsing video/image datasets...")
    for name, parser_fn, mapping_key in parsers:
        if selected_datasets and name not in selected_datasets:
            continue
        if mapping_key in classes:
            all_entries.extend(parser_fn(raw_dir, classes[mapping_key]))
        else:
            print(f"  {name}: skipped (no mapping in config)")

    return all_entries


def process_and_save(entries, config, processed_dir):
    """Process all video/image entries and save frame sequences + labels."""
    video_cfg = config["video"]
    num_frames = video_cfg["num_frames"]
    frame_size = video_cfg["frame_size"]
    sampling = video_cfg["sampling_strategy"]
    detect_face = video_cfg["face_detection"]

    class_names = config["classes"]["names"]
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Initialize face detector
    face_detector = None
    if detect_face:
        face_detector = FaceDetector(
            backend=video_cfg["face_detector"],
            min_face_size=video_cfg["min_face_size"],
            margin=video_cfg["face_margin"],
        )

    features = []
    labels = []
    metadata = []
    skipped = 0

    print(f"\nProcessing {len(entries)} video/image entries...")
    print(f"  Frames per clip: {num_frames}")
    print(f"  Frame size: {frame_size}x{frame_size}")
    print(f"  Face detection: {video_cfg['face_detector'] if detect_face else 'disabled'}")
    print(f"  Sampling: {sampling}")
    print()

    for entry in tqdm(entries, desc="Processing videos"):
        path = entry["path"]
        entry_type = entry.get("type", "video")

        # Extract frames based on entry type
        if entry_type == "video":
            frames = extract_frames_from_video(path, num_frames, sampling)
        elif entry_type == "sequence":
            frames = extract_frames_from_image_sequence(path, num_frames, sampling)
        elif entry_type == "image":
            frames = single_image_to_sequence(path, num_frames)
        else:
            frames = None

        if frames is None:
            skipped += 1
            continue

        # Process: face detect, crop, resize
        processed = process_frames(frames, face_detector, frame_size, detect_face)

        label_idx = class_to_idx[entry["label"]]
        features.append(processed)
        labels.append(label_idx)
        metadata.append({
            "path": entry["path"],
            "label": entry["label"],
            "label_idx": label_idx,
            "source": entry["source"],
            "type": entry_type,
        })

    if face_detector:
        face_detector.close()

    if not features:
        print("\nERROR: No samples were successfully processed.")
        return None, None, None

    # Stack into arrays
    # features shape: (N, num_frames, H, W, 3) as uint8
    features = np.array(features, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)

    print(f"\nProcessed: {len(features)} samples ({skipped} skipped)")
    print(f"Feature shape: {features.shape}")
    print(f"Memory: {features.nbytes / 1e9:.2f} GB")

    # Print class distribution
    print("\nClass distribution:")
    label_counts = Counter(labels)
    for idx, name in enumerate(class_names):
        count = label_counts.get(idx, 0)
        pct = count / len(labels) * 100 if len(labels) > 0 else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    return features, labels, metadata


def create_splits(features, labels, metadata, config, processed_dir):
    """Create train/val/test splits and save."""
    train_cfg = config["training"]
    train_ratio = train_cfg["train_ratio"]
    val_ratio = train_cfg["val_ratio"]

    processed_dir = Path(processed_dir)

    # Stratified split
    X_train, X_temp, y_train, y_temp, meta_train, meta_temp = train_test_split(
        features, labels, metadata,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=train_cfg["seed"],
    )

    relative_val = val_ratio / (val_ratio + train_cfg["test_ratio"])
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=(1 - relative_val),
        stratify=y_temp,
        random_state=train_cfg["seed"],
    )

    # Save splits
    print(f"\nSaving splits to {processed_dir}...")

    # For large video datasets, save as memory-mapped files
    total_bytes = features.nbytes
    use_memmap = total_bytes > 2e9  # > 2GB -> use memmap

    for split_name, X, y, meta in [
        ("train", X_train, y_train, meta_train),
        ("val", X_val, y_val, meta_val),
        ("test", X_test, y_test, meta_test),
    ]:
        if use_memmap:
            # Save as memory-mapped .npy for large datasets
            fp = np.memmap(
                processed_dir / f"{split_name}_frames.npy",
                dtype=np.uint8, mode="w+", shape=X.shape,
            )
            fp[:] = X[:]
            fp.flush()
            del fp
            # Save shape info
            np.save(processed_dir / f"{split_name}_frames_shape.npy", np.array(X.shape))
        else:
            np.save(processed_dir / f"{split_name}_frames.npy", X)

        np.save(processed_dir / f"{split_name}_labels.npy", y)
        pd.DataFrame(meta).to_csv(
            processed_dir / f"{split_name}_metadata.csv", index=False
        )
        print(f"  {split_name}: {len(X)} samples")

    # Save dataset info
    dataset_info = {
        "class_names": config["classes"]["names"],
        "num_classes": config["classes"]["num_classes"],
        "num_frames": config["video"]["num_frames"],
        "frame_size": config["video"]["frame_size"],
        "face_detection": config["video"]["face_detection"],
        "face_detector": config["video"]["face_detector"],
        "sampling_strategy": config["video"]["sampling_strategy"],
        "feature_shape": list(features.shape[1:]),
        "total_samples": len(features),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "memmap": use_memmap,
    }
    with open(processed_dir / "dataset_info.yaml", "w") as f:
        yaml.dump(dataset_info, f, default_flow_style=False)

    print(f"\nDataset info saved to {processed_dir / 'dataset_info.yaml'}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess video face expression datasets"
    )
    parser.add_argument(
        "--config", default="configs/video_config.yaml",
        help="Config file path"
    )
    parser.add_argument(
        "--raw-dir", default=None,
        help="Override raw data directory"
    )
    parser.add_argument(
        "--processed-dir", default=None,
        help="Override processed data directory"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        choices=["dfew", "ferv39k", "fer2013", "raf_db", "ck_plus", "aff_wild2", "ravdess"],
        help="Only process specific datasets"
    )
    parser.add_argument(
        "--no-face-detection", action="store_true",
        help="Disable face detection (use full frames)"
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Use manifest CSV (path, label, source, type)"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    if args.no_face_detection:
        config["video"]["face_detection"] = False

    raw_dir = args.raw_dir or config["datasets"]["raw_dir"]
    processed_dir = args.processed_dir or config["datasets"]["processed_dir"]

    print("=" * 60)
    print("Video Face Expression Dataset Preprocessor")
    print("=" * 60)
    print(f"  Raw dir:       {raw_dir}")
    print(f"  Processed dir: {processed_dir}")
    print(f"  Datasets:      {args.datasets or 'all'}")

    # Step 1: Collect entries
    manifest_path = args.manifest or os.path.join(
        config["datasets"]["base_dir"], "video_manifest.csv"
    )
    if os.path.exists(manifest_path):
        print(f"\nLoading manifest from {manifest_path}")
        df = pd.read_csv(manifest_path)
        entries = df.to_dict("records")
        print(f"  {len(entries)} entries from manifest")
    else:
        entries = collect_all_entries(raw_dir, config, args.datasets)

    if not entries:
        print(f"\nERROR: No video/image files found in {raw_dir}")
        print("Download datasets first — see DATASETS.md or run ./download_datasets.sh")
        sys.exit(1)

    # Step 2: Process videos and extract frame sequences
    features, labels, metadata = process_and_save(entries, config, processed_dir)
    if features is None:
        sys.exit(1)

    # Step 3: Create train/val/test splits
    create_splits(features, labels, metadata, config, processed_dir)

    print("\n" + "=" * 60)
    print("Video preprocessing complete!")
    print(f"Output: {processed_dir}/")
    print("Next step: python scripts/train_video.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
