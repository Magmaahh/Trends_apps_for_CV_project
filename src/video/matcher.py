from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine


class IdentityMatcher:
    """
    Compare video embeddings against a gold-standard identity profile for deepfake detection.
    
    This class implements the verification step: given embeddings extracted from a test video,
    it compares them against a known authentic "gold standard" profile of the claimed speaker.
    
    How it works:
    1. Load gold standard: A JSON file containing reference embeddings for each phoneme
       from authentic videos of the person
    2. Compare test embeddings: For each phoneme in the test video, compute similarity
       (cosine similarity) with the gold standard
    3. Aggregate scores: Average all similarities to get a global authenticity score
    4. Make verdict: If score >= threshold, video is likely REAL; otherwise, likely FAKE
    
    Key Insight:
    Deepfake videos may look visually convincing, but subtle inconsistencies in mouth
    movements during specific phonemes will result in lower similarity scores.
    """

    def __init__(self, gold_json: Path, threshold: float = 0.75) -> None:
        """
        Initialize the matcher with a gold standard profile.
        
        Args:
            gold_json: Path to JSON file containing reference embeddings
                      Format: {"phoneme": {"vector": [128-D list]}, ...}
            threshold: Similarity threshold for REAL/FAKE decision (default: 0.75)
                      Higher = stricter (fewer false positives, more false negatives)
                      Lower = more lenient (more false positives, fewer false negatives)
        """
        self.gold_path = Path(gold_json)
        if not self.gold_path.exists():
            raise FileNotFoundError(f"Gold JSON not found: {self.gold_path}")
        with self.gold_path.open("r", encoding="utf-8") as handle:
            self.gold = json.load(handle)
        self.threshold = threshold

    def compare(self, input_embeddings: Dict[str, List[np.ndarray]]) -> Dict[str, object]:
        """
        Compare test video embeddings against gold standard and return verdict.
        
        Process:
        1. For each phoneme in the test video:
           - Compare each occurrence against the gold standard embedding
           - Use cosine similarity: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
        2. Keep track of best (highest) similarity for each phoneme
        3. Compute global score as mean of all similarities
        4. Make verdict based on threshold
        
        Args:
            input_embeddings: Dict mapping phoneme labels to lists of embedding vectors
                             from the test video. Format: {"IH1": [emb1, emb2, ...], ...}
        
        Returns:
            Dictionary containing:
            - global_score: Overall similarity score (0.0 to 1.0)
            - threshold: The threshold used for decision
            - verdict: "REAL" or "FAKE"
            - per_phoneme_best: Best similarity score for each phoneme
        
        Example:
            {
                "global_score": 0.82,
                "threshold": 0.75,
                "verdict": "REAL",
                "per_phoneme_best": {"IH1": 0.85, "AE1": 0.79, ...}
            }
        """
        per_phoneme: Dict[str, float] = {}
        scores: List[float] = []

        # Compare each phoneme's embeddings
        for phoneme, vectors in input_embeddings.items():
            # Skip phonemes not in gold standard (may be rare phonemes)
            if phoneme not in self.gold:
                continue
            
            gold_vec = np.array(self.gold[phoneme]["vector"], dtype=np.float32)
            
            # Compare each occurrence of this phoneme
            for vec in vectors:
                vec = np.array(vec, dtype=np.float32)
                # Cosine similarity: 1 - cosine_distance
                sim = 1 - cosine(gold_vec, vec)
                if np.isnan(sim):
                    continue
                scores.append(sim)
                # Keep best (highest) score for this phoneme
                per_phoneme.setdefault(phoneme, 0.0)
                per_phoneme[phoneme] = max(per_phoneme[phoneme], float(sim))

        # Compute global score and make verdict
        global_score = float(np.mean(scores)) if scores else 0.0
        verdict = "REAL" if global_score >= self.threshold else "FAKE"
        
        return {
            "global_score": global_score,
            "threshold": self.threshold,
            "verdict": verdict,
            "per_phoneme_best": per_phoneme,
        }
