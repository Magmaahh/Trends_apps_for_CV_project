from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_textgrid(textgrid_path: str, tier_name: str = "phones") -> List[Dict[str, float]]:
    """Parse TextGrid to list of {phoneme, start, end}."""
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
    """Convert GRID .align file to clean uppercase text."""
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
    """Load all frames and fps from video."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Impossibile aprire video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError("Nessun frame letto dal video.")
    return frames, fps


def aggregate_embeddings(embedding_files: List[Path]) -> Dict[str, List[float]]:
    """Aggregate per-video npz embeddings into gold centroids."""
    import json

    aggregated: Dict[str, List[np.ndarray]] = {}
    for npz_file in embedding_files:
        data = np.load(npz_file)
        for phoneme, arr in data.items():
            arr = np.atleast_2d(arr)
            aggregated.setdefault(phoneme, []).extend([row for row in arr])

    gold: Dict[str, List[float]] = {}
    for phoneme, vectors in aggregated.items():
        stack = np.stack(vectors)
        centroid = stack.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            continue
        gold[phoneme] = (centroid / norm).tolist()
    return gold
