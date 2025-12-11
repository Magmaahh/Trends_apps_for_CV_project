from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional


def parse_textgrid(textgrid_path: str, tier_name: str = "phones") -> List[Dict[str, float]]:
    """
    Parse a TextGrid file and return a list of phoneme intervals.

    Each item has keys: phoneme, start, end (seconds).
    Falls back to a light parser if the textgrid package is unavailable.
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
                intervals.append(
                    {
                        "phoneme": label,
                        "start": float(interval.minTime),
                        "end": float(interval.maxTime),
                    }
                )
        return intervals
    except Exception:
        # Lightweight fallback parser for basic TextGrid structure
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        in_target_tier = False
        current_interval: Dict[str, Optional[float]] = {}

        for line in lines:
            line = line.strip()
            if f'name = "{tier_name}"' in line:
                in_target_tier = True
                continue
            if in_target_tier and line.startswith('name = "'):
                break
            if in_target_tier and line.startswith("intervals ["):
                current_interval = {"start": None, "end": None}
            if in_target_tier:
                if line.startswith("xmin ="):
                    current_interval["start"] = float(line.split("=")[1].strip())
                elif line.startswith("xmax ="):
                    current_interval["end"] = float(line.split("=")[1].strip())
                elif line.startswith("text ="):
                    text = line.split("=")[1].strip().replace('"', "")
                    if current_interval.get("start") is not None and current_interval.get("end") is not None:
                        if text and text not in {"sil", "sp", "<eps>"}:
                            intervals.append(
                                {
                                    "phoneme": text,
                                    "start": float(current_interval["start"]),  # type: ignore[arg-type]
                                    "end": float(current_interval["end"]),      # type: ignore[arg-type]
                                }
                            )
        return intervals


def parse_align_file(align_path: str) -> str:
    """
    Convert a GRID .align file into clean uppercase text for MFA.
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
