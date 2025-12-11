from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine


class IdentityMatcher:
    """Compare input embeddings against a gold-standard identity JSON."""

    def __init__(self, gold_json: Path, threshold: float = 0.75) -> None:
        self.gold_path = Path(gold_json)
        if not self.gold_path.exists():
            raise FileNotFoundError(f"Gold JSON non trovato: {self.gold_path}")
        with self.gold_path.open("r", encoding="utf-8") as handle:
            self.gold = json.load(handle)
        self.threshold = threshold

    def compare(self, input_embeddings: Dict[str, List[np.ndarray]]) -> Dict[str, object]:
        per_phoneme: Dict[str, float] = {}
        scores: List[float] = []

        for phoneme, vectors in input_embeddings.items():
            if phoneme not in self.gold:
                continue
            gold_vec = np.array(self.gold[phoneme]["vector"], dtype=np.float32)
            for vec in vectors:
                vec = np.array(vec, dtype=np.float32)
                sim = 1 - cosine(gold_vec, vec)
                if np.isnan(sim):
                    continue
                scores.append(sim)
                per_phoneme.setdefault(phoneme, 0.0)
                per_phoneme[phoneme] = max(per_phoneme[phoneme], float(sim))

        global_score = float(np.mean(scores)) if scores else 0.0
        verdict = "REAL" if global_score >= self.threshold else "FAKE"
        return {
            "global_score": global_score,
            "threshold": self.threshold,
            "verdict": verdict,
            "per_phoneme_best": per_phoneme,
        }
