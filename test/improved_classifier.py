"""
Improved Multi-Metric Classifier for REAL vs FAKE Detection

Uses the best discriminative metrics found in multi_metric_analysis.py:
- Audio: test_kurtosis, test_cv, test_std_dist, l2_distance, cosine_sim
- Video: l2_distance, test_kurtosis, test_std_dist, test_cv, cosine_sim

Usage:
    python improved_classifier.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import skew, kurtosis
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONFIGURATION
# =============================================================================

PERSON = "trump"
PROCESSED_BASE = Path("../dataset/trump_biden/processed") / PERSON

# Video categories
REFERENCE_IDS = [f"t-{i:02d}" for i in range(9, 15)]  # t-09 to t-14
REAL_IDS = ["t-08", "t-15"]  # Real but not in training
FAKE_IDS = [f"t-{i:02d}" for i in range(0, 8)]  # t-00 to t-07

# Metric weights (from effect size analysis)
AUDIO_WEIGHTS = {
    'test_kurtosis': 2.013,
    'test_cv': 1.293,
    'test_std_dist': 1.156,
    'l2_distance': 0.876,
    'cosine_sim': 0.815,
}

VIDEO_WEIGHTS = {
    'l2_distance': 2.588,
    'test_kurtosis': 2.374,
    'test_std_dist': 1.557,
    'test_cv': 1.423,
    'cosine_sim': 1.312,
}

# =============================================================================


def load_embeddings_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {}
    
    for phoneme in data.files:
        emb = data[phoneme]
        if emb.ndim == 2:
            emb = np.mean(emb, axis=0)
        embeddings[phoneme] = emb
    
    return embeddings


def load_category_embeddings(
    video_ids: List[str],
    modality: str = "audio"
) -> Dict[str, List[np.ndarray]]:
    """Load embeddings for a category, keeping separate per video."""
    phoneme_embeddings = defaultdict(list)
    
    for video_id in video_ids:
        npz_path = PROCESSED_BASE / video_id / f"{video_id}_{modality}.npz"
        
        if not npz_path.exists():
            continue
        
        embeddings = load_embeddings_npz(str(npz_path))
        
        for phoneme, vec in embeddings.items():
            phoneme_embeddings[phoneme].append(vec)
    
    return dict(phoneme_embeddings)


def compute_metrics_per_phoneme(
    reference_embs: List[np.ndarray],
    test_embs: List[np.ndarray]
) -> Dict[str, float]:
    """Compute metrics between reference and test embeddings."""
    ref_centroid = np.mean(reference_embs, axis=0)
    test_centroid = np.mean(test_embs, axis=0)
    
    metrics = {}
    
    # Cosine Similarity
    metrics['cosine_sim'] = 1 - cosine(ref_centroid, test_centroid)
    
    # L2 Distance
    metrics['l2_distance'] = euclidean(ref_centroid, test_centroid)
    
    # Test statistics (distance to reference centroid)
    test_dists = [euclidean(emb, ref_centroid) for emb in test_embs]
    metrics['test_std_dist'] = np.std(test_dists)
    test_mean = np.mean(test_dists)
    metrics['test_cv'] = metrics['test_std_dist'] / (test_mean + 1e-8)
    
    # Distribution metrics
    if len(test_dists) > 2:
        metrics['test_kurtosis'] = kurtosis(test_dists)
    else:
        metrics['test_kurtosis'] = 0
    
    return metrics


def compute_weighted_score(
    metrics: Dict[str, float],
    weights: Dict[str, float],
    reference_stats: Dict[str, Tuple[float, float]]
) -> float:
    """
    Compute weighted score normalized by reference statistics.
    
    Args:
        metrics: Computed metrics
        weights: Metric weights (effect sizes)
        reference_stats: Reference (real_mean, real_std) for each metric
        
    Returns:
        Weighted normalized score (higher = more similar to REAL)
    """
    total_score = 0
    total_weight = 0
    
    for metric_name, weight in weights.items():
        if metric_name not in metrics or metric_name not in reference_stats:
            continue
        
        value = metrics[metric_name]
        ref_mean, ref_std = reference_stats[metric_name]
        
        # Normalize: distance from reference mean in std units
        # Lower distance = more similar to REAL
        normalized_dist = abs(value - ref_mean) / (ref_std + 1e-8)
        
        # Invert: higher score = closer to REAL
        similarity_score = 1 / (1 + normalized_dist)
        
        total_score += weight * similarity_score
        total_weight += weight
    
    return total_score / (total_weight + 1e-8)


class ImprovedClassifier:
    """Multi-metric classifier for REAL vs FAKE detection."""
    
    def __init__(self):
        self.reference_audio = {}
        self.reference_video = {}
        self.audio_stats = {}
        self.video_stats = {}
    
    def train(self, ref_ids: List[str], real_ids: List[str]):
        """
        Train by computing reference statistics from REAL videos.
        
        Args:
            ref_ids: Reference video IDs (for building centroids)
            real_ids: Real test video IDs (for computing "REAL" statistics)
        """
        print("="*80)
        print("🎓 TRAINING IMPROVED CLASSIFIER")
        print("="*80)
        
        # Load reference embeddings (for centroids)
        self.reference_audio = load_category_embeddings(ref_ids, "audio")
        self.reference_video = load_category_embeddings(ref_ids, "video")
        
        print(f"✓ Loaded reference: {len(self.reference_audio)} audio phonemes, {len(self.reference_video)} video phonemes")
        
        # Compute statistics from REAL videos
        real_audio = load_category_embeddings(real_ids, "audio")
        real_video = load_category_embeddings(real_ids, "video")
        
        print(f"✓ Loaded real test: {len(real_audio)} audio phonemes, {len(real_video)} video phonemes")
        
        # Compute metric statistics for REAL samples
        print("\nComputing REAL statistics...")
        self.audio_stats = self._compute_reference_stats(
            self.reference_audio, real_audio, "audio"
        )
        self.video_stats = self._compute_reference_stats(
            self.reference_video, real_video, "video"
        )
        
        print("✓ Training complete")
    
    def _compute_reference_stats(
        self,
        ref_embs: Dict,
        real_embs: Dict,
        modality: str
    ) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std of each metric for REAL samples."""
        
        common = set(ref_embs.keys()) & set(real_embs.keys())
        
        all_metrics = defaultdict(list)
        
        for phoneme in common:
            if len(ref_embs[phoneme]) < 2 or len(real_embs[phoneme]) < 1:
                continue
            
            metrics = compute_metrics_per_phoneme(
                ref_embs[phoneme],
                real_embs[phoneme]
            )
            
            for key, val in metrics.items():
                all_metrics[key].append(val)
        
        # Compute mean and std for each metric
        stats = {}
        for key, values in all_metrics.items():
            stats[key] = (np.mean(values), np.std(values))
        
        print(f"  {modality}: {len(stats)} metrics computed from {len(common)} phonemes")
        
        return stats
    
    def predict(self, test_id: str) -> Dict:
        """
        Predict if test video is REAL or FAKE.
        
        Returns:
            Dict with scores and verdict
        """
        # Load test embeddings
        test_audio_path = PROCESSED_BASE / test_id / f"{test_id}_audio.npz"
        test_video_path = PROCESSED_BASE / test_id / f"{test_id}_video.npz"
        
        if not test_audio_path.exists() or not test_video_path.exists():
            return {"error": "Embeddings not found"}
        
        test_audio_raw = load_embeddings_npz(str(test_audio_path))
        test_video_raw = load_embeddings_npz(str(test_video_path))
        
        # Convert to list format
        test_audio = {p: [v] for p, v in test_audio_raw.items()}
        test_video = {p: [v] for p, v in test_video_raw.items()}
        
        # Compute scores
        audio_scores = []
        video_scores = []
        
        # Audio
        for phoneme in set(self.reference_audio.keys()) & set(test_audio.keys()):
            if len(self.reference_audio[phoneme]) < 2:
                continue
            
            metrics = compute_metrics_per_phoneme(
                self.reference_audio[phoneme],
                test_audio[phoneme]
            )
            
            score = compute_weighted_score(metrics, AUDIO_WEIGHTS, self.audio_stats)
            audio_scores.append(score)
        
        # Video
        for phoneme in set(self.reference_video.keys()) & set(test_video.keys()):
            if len(self.reference_video[phoneme]) < 2:
                continue
            
            metrics = compute_metrics_per_phoneme(
                self.reference_video[phoneme],
                test_video[phoneme]
            )
            
            score = compute_weighted_score(metrics, VIDEO_WEIGHTS, self.video_stats)
            video_scores.append(score)
        
        # Aggregate
        audio_score = np.mean(audio_scores) if audio_scores else 0
        video_score = np.mean(video_scores) if video_scores else 0
        
        # Combined score (weighted by modality discrimination power)
        # Video has stronger discrimination, so give it more weight
        combined_score = (audio_score * 0.4 + video_score * 0.6)
        
        # Determine verdict
        if combined_score > 0.65:
            verdict = "REAL"
            confidence = min(85 + (combined_score - 0.65) * 100, 99)
        elif combined_score > 0.5:
            verdict = "LIKELY REAL"
            confidence = 50 + (combined_score - 0.5) * 70
        elif combined_score > 0.35:
            verdict = "UNCERTAIN"
            confidence = 25 + (combined_score - 0.35) * 50
        else:
            verdict = "FAKE"
            confidence = combined_score * 100
        
        return {
            "video_id": test_id,
            "verdict": verdict,
            "confidence": confidence,
            "audio_score": audio_score,
            "video_score": video_score,
            "combined_score": combined_score
        }


def main():
    print("="*80)
    print("🔬 IMPROVED MULTI-METRIC CLASSIFIER")
    print("="*80)
    print(f"Person: {PERSON.upper()}")
    print("="*80)
    
    # Train classifier
    classifier = ImprovedClassifier()
    classifier.train(REFERENCE_IDS, REAL_IDS)
    
    # Test on REAL videos
    print("\n" + "="*80)
    print("🔵 TESTING ON REAL VIDEOS")
    print("="*80)
    print(f"{'Video':<10} {'Audio':>8} {'Video':>8} {'Combined':>8} {'Verdict':<15} {'Confidence'}")
    print("-"*80)
    
    real_results = []
    for test_id in REAL_IDS:
        result = classifier.predict(test_id)
        if "error" in result:
            continue
        real_results.append(result)
        
        status = "🟢" if result["combined_score"] > 0.6 else "🟡" if result["combined_score"] > 0.4 else "🔴"
        print(f"{status} {result['video_id']:<8} {result['audio_score']:>8.3f} {result['video_score']:>8.3f} "
              f"{result['combined_score']:>8.3f} {result['verdict']:<15} {result['confidence']:.1f}%")
    
    # Test on FAKE videos
    print("\n" + "="*80)
    print("🔴 TESTING ON FAKE VIDEOS")
    print("="*80)
    print(f"{'Video':<10} {'Audio':>8} {'Video':>8} {'Combined':>8} {'Verdict':<15} {'Confidence'}")
    print("-"*80)
    
    fake_results = []
    for test_id in FAKE_IDS:
        result = classifier.predict(test_id)
        if "error" in result:
            continue
        fake_results.append(result)
        
        status = "🟢" if result["combined_score"] < 0.4 else "🟡" if result["combined_score"] < 0.6 else "🔴"
        print(f"{status} {result['video_id']:<8} {result['audio_score']:>8.3f} {result['video_score']:>8.3f} "
              f"{result['combined_score']:>8.3f} {result['verdict']:<15} {result['confidence']:.1f}%")
    
    # Summary
    print("\n" + "="*80)
    print("📊 CLASSIFICATION SUMMARY")
    print("="*80)
    
    if real_results and fake_results:
        real_avg = np.mean([r["combined_score"] for r in real_results])
        fake_avg = np.mean([r["combined_score"] for r in fake_results])
        separation = real_avg - fake_avg
        
        real_correct = sum(1 for r in real_results if r["combined_score"] > 0.5)
        fake_correct = sum(1 for r in fake_results if r["combined_score"] <= 0.5)
        
        accuracy = (real_correct + fake_correct) / (len(real_results) + len(fake_results))
        
        print(f"REAL average score:  {real_avg:.3f}")
        print(f"FAKE average score:  {fake_avg:.3f}")
        print(f"Separation:          {separation:.3f}")
        print()
        print(f"REAL correctly classified: {real_correct}/{len(real_results)}")
        print(f"FAKE correctly classified: {fake_correct}/{len(fake_results)}")
        print(f"Overall accuracy:          {accuracy*100:.1f}%")
        print("="*80)
        
        if separation > 0.2:
            print("✅ GOOD SEPARATION - Classifier works well!")
        elif separation > 0.1:
            print("⚠️  MODERATE SEPARATION - Some discrimination")
        else:
            print("❌ POOR SEPARATION - Classifier needs improvement")


if __name__ == "__main__":
    main()
