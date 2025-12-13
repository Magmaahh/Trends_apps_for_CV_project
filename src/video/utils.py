from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_textgrid(textgrid_path: str, tier_name: str = "phones") -> List[Dict[str, float]]:
    """
    Parse a TextGrid file to extract phoneme intervals with timestamps.
    
    TextGrid is a file format used by Praat and MFA to store time-aligned annotations.
    It contains "tiers" (layers) of annotations, where each tier has intervals with labels.
    
    This function extracts the "phones" tier which contains phoneme-level alignments.
    
    Args:
        textgrid_path: Path to .TextGrid file
        tier_name: Name of tier to extract (default: "phones")
        
    Returns:
        List of dictionaries, each containing:
        - phoneme: Phoneme label (e.g., "IH1", "AE1")
        - start: Start time in seconds
        - end: End time in seconds
        
    Example output:
        [
            {"phoneme": "HH", "start": 0.0, "end": 0.1},
            {"phoneme": "EH1", "start": 0.1, "end": 0.25},
            {"phoneme": "L", "start": 0.25, "end": 0.35},
            ...
        ]
    
    Note: Filters out silence markers ("sil", "sp", "<eps>")
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


def parse_align_file(align_path: str) -> str:
    """
    Convert GRID dataset .align file to clean uppercase text.
    
    The GRID audiovisual speech corpus uses .align files to store word-level
    alignments. This function extracts just the words (ignoring timestamps)
    to create a transcript.
    
    Args:
        align_path: Path to .align file
        
    Returns:
        Space-separated uppercase transcript
        
    Example:
        Input file content:
            0.0 0.5 bin
            0.5 1.0 blue
            1.0 1.5 at
        Output:
            "BIN BLUE AT"
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


def load_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    """
    Load all frames from a video file along with its frame rate.
    
    This is a utility function used by the pipeline to load the entire video
    into memory for processing. Each frame is a BGR numpy array (OpenCV format).
    
    Args:
        video_path: Path to video file
        
    Returns:
        Tuple of (frames, fps) where:
        - frames: List of BGR numpy arrays, one per frame
        - fps: Frames per second (used to convert time to frame indices)
        
    Note: For long videos, this may consume significant memory. Consider
    processing videos in chunks for production use.
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


def aggregate_embeddings(embedding_files: List[Path]) -> Dict[str, List[float]]:
    """
    Aggregate embeddings from multiple videos to create a gold standard profile.
    
    This function is used to build the reference "gold standard" profile from
    multiple authentic videos of a person. It combines all embeddings for each
    phoneme and computes a representative centroid.
    
    Process:
    1. Load all .npz files (each contains embeddings from one video)
    2. Collect all embeddings for each phoneme across all videos
    3. Compute mean (centroid) for each phoneme
    4. Normalize to unit length (for cosine similarity comparison)
    
    Args:
        embedding_files: List of paths to .npz files, each containing
                        phoneme embeddings from a single video
                        
    Returns:
        Dictionary mapping phoneme labels to normalized centroid vectors.
        Format: {"IH1": [128-D normalized vector], "AE1": [...], ...}
        
    Example:
        If you have 5 authentic videos of a person:
        - video1.npz: {"IH1": [emb1, emb2], "AE1": [emb3]}
        - video2.npz: {"IH1": [emb4], "AE1": [emb5, emb6]}
        - ...
        
        Result: {"IH1": mean([emb1, emb2, emb4, ...]), "AE1": mean([emb3, emb5, emb6, ...])}
        
    Note: The resulting profile represents the "average" way this person
    pronounces each phoneme, which serves as the reference for comparison.
    """
    import json

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
