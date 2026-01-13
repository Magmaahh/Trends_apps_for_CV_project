import os
import re
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- CONFIGURATION PATHS AND CONSTANTS ---

# Dataset paths
DATASET_INIT = "dataset/init"
DATASET_OUTPUT = "dataset/output"
GOLD_STORE_DIR = "dataset/output/gold_store"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
IMG_SIZE = 112
EMBEDDING_DIM = 128
MIN_PHONEME_SAMPLES = 5

# GRID Corpus Grammar (for filename decoding)
GRID_GRAMMAR = {
    0: {"b": "bin", "l": "lay", "p": "place", "s": "set"},
    1: {"b": "blue", "g": "green", "r": "red", "w": "white"},
    2: {"a": "at", "b": "by", "i": "in", "w": "with"},
    3: {l: l.upper() for l in "abcdefghijklmnopqrstuvwxyz"}, # letters
    4: {"z": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"},
    5: {"n": "now", "p": "please", "s": "soon"}
}

# --- UTILITY FUNCTIONS ---

# Loads speaker list
def load_speakers():
    """
    Returns a sorted list of speaker IDs found in the dataset.
    """
    audio_root = os.path.join(DATASET_INIT, "audio_25k")
    if not os.path.exists(audio_root):
        return []

    speakers = [
        d for d in os.listdir(audio_root)
        if os.path.isdir(os.path.join(audio_root, d)) and d.startswith("s")
    ]

    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    return speakers

# Loads speakers with video embeddings
def load_embedding_speakers():
    """
    Returns speaker IDs for which video embeddings exist.
    """
    if not os.path.exists(DATASET_OUTPUT):
        return []
        
    folders = [
        d for d in os.listdir(DATASET_OUTPUT)
        if d.startswith("video_embeddings_s")
    ]

    speakers = [f.split("_")[-1] for f in folders]
    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    return speakers

# Parses TextGrid file
def parse_textgrid(textgrid_path: str, tier_name: str = "phones") -> List[Dict[str, float]]:
    """
    Parse a TextGrid file to extract phoneme intervals with timestamps.
    """
    path = Path(textgrid_path)
    if not path.exists():
        raise FileNotFoundError(f"TextGrid not found: {path}")

    intervals: List[Dict[str, float]] = []
    try:
        from textgrid import TextGrid  # type: ignore

        tg = TextGrid.fromFile(str(path))
        tier = next((t for t in tg.tiers if t.name == tier_name), None)
        if tier is None:
            return intervals
        for interval in tier.intervals:
            label = interval.mark.strip()
            if label and label not in {"sil", "sp", "<eps>"}:
                intervals.append({"phoneme": label, "start": float(interval.minTime), "end": float(interval.maxTime)})
        return intervals
    except Exception:
        # Fallback manual parsing
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        in_tier = False
        current: Dict[str, Optional[float]] = {}
        for line in lines:
            line = line.strip()
            if f'name = "{tier_name}"' in line:
                in_tier = True
                continue
            if in_tier and line.startswith('name = "'):
                break
            if in_tier and line.startswith("intervals ["):
                current = {"start": None, "end": None}
            if in_tier:
                if line.startswith("xmin ="):
                    current["start"] = float(line.split("=")[1].strip())
                elif line.startswith("xmax ="):
                    current["end"] = float(line.split("=")[1].strip())
                elif line.startswith("text ="):
                    text = line.split("=")[1].strip().replace('"', "")
                    if current.get("start") is not None and current.get("end") is not None:
                        if text and text not in {"sil", "sp", "<eps>"}:
                            intervals.append(
                                {"phoneme": text, "start": float(current["start"]), "end": float(current["end"])}
                            )
        return intervals

# Parses GRID .align files
def parse_align_file(align_path: str) -> str:
    """
    Convert GRID dataset .align file to clean uppercase text.
    """
    path = Path(align_path)
    if not path.exists():
        raise FileNotFoundError(f"Align file not found: {path}")
    words: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = re.split(r"\s+", line.strip())
            if len(parts) == 3:
                word = parts[2]
                if word not in {"sil", "sp"}:
                    words.append(word)
    return " ".join(words).upper()

# Loads video frames
def load_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """
    Load all frames from a video file along with its frame rate.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("No frames read from video.")
    return frames, fps

# Aggregates embeddings to create gold standard profile
def aggregate_embeddings(embedding_files: List[Path]) -> Dict[str, List[float]]:
    """
    Aggregate embeddings from multiple videos to create a gold standard profile.
    """
    # Step 1: Collect all embeddings for each phoneme
    aggregated: Dict[str, List[np.ndarray]] = {}
    for npz_file in embedding_files:
        data = np.load(npz_file)
        for phoneme, arr in data.items():
            arr = np.atleast_2d(arr)
            aggregated.setdefault(phoneme, []).extend([row for row in arr])

    # Step 2: Compute normalized centroid for each phoneme
    gold: Dict[str, List[float]] = {}
    for phoneme, vectors in aggregated.items():
        stack = np.stack(vectors)
        centroid = stack.mean(axis=0)
        
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        gold[phoneme] = (centroid / norm).tolist()
    
    return gold

# Decodes GRID filenames
def decode_grid_filename(stem: str):
    """
    Decode a 6-character GRID filename into a sentence.
    """
    if len(stem) != 6:
        return None

    words = []
    for idx, char in enumerate(stem.lower()):
        mapping = GRID_GRAMMAR.get(idx)
        if mapping is None or char not in mapping:
            return None
        words.append(mapping[char])

    return " ".join(words).upper()
