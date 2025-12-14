"""
Grid Search for Optimal Multimodal Verification Parameters

This script finds the best hyperparameters by testing multiple combinations
and evaluating on:
- Real held-out videos (should be high compatibility)
- Fake videos (should be low compatibility)

The goal is to maximize: real_score - fake_score

Usage:
    python grid_search.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from itertools import product

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from multimodal_space (we'll modify the class inline)
from multimodal_space import (
    MultimodalCompatibilitySpace,
    load_audio_embeddings,
    load_video_embeddings_npz
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Test BOTH persons
PERSONS = ["trump", "biden"]

# Paths
PROCESSED_DIR = Path("../dataset/trump_biden/processed")

# Grid search parameters - EXTENSIVE SEARCH (1540 configurations)
THRESHOLD_SIGMA_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0]
THRESHOLD_MULTIPLIER_VALUES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0]
LAMBDA_REG_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# =============================================================================


def get_processed_paths(video_id: str, person: str) -> Dict[str, Path]:
    """Get paths for processed files."""
    processed_dir = PROCESSED_DIR / person / video_id
    return {
        "audio_emb": processed_dir / f"{video_id}_audio.npz",
        "video_emb": processed_dir / f"{video_id}_video.npz"
    }


def load_video_embeddings_single(video_id: str, person: str) -> Tuple[Dict, Dict]:
    """Load embeddings for a single video."""
    paths = get_processed_paths(video_id, person)
    
    if not paths["audio_emb"].exists() or not paths["video_emb"].exists():
        return {}, {}
    
    audio_emb = load_audio_embeddings(str(paths["audio_emb"]))
    video_emb = load_video_embeddings_npz(str(paths["video_emb"]))
    
    return audio_emb, video_emb


def load_videos_aggregated(video_ids: List[str], person: str) -> Tuple[Dict, Dict]:
    """Load and aggregate embeddings from multiple videos."""
    audio_agg: Dict[str, List[np.ndarray]] = defaultdict(list)
    video_agg: Dict[str, List[np.ndarray]] = defaultdict(list)
    
    for vid in video_ids:
        audio_emb, video_emb = load_video_embeddings_single(vid, person)
        
        if not audio_emb or not video_emb:
            continue
        
        for phoneme in set(audio_emb.keys()) & set(video_emb.keys()):
            a = audio_emb[phoneme]
            v = video_emb[phoneme]
            
            if a.ndim == 2:
                a = np.mean(a, axis=0)
            if v.ndim == 2:
                v = np.mean(v, axis=0)
            
            audio_agg[phoneme].append(a)
            video_agg[phoneme].append(v)
    
    # Convert to arrays
    audio_arrays = {k: np.array(v) for k, v in audio_agg.items()}
    video_arrays = {k: np.array(v) for k, v in video_agg.items()}
    
    return audio_arrays, video_arrays


class ConfigurableMultimodalSpace(MultimodalCompatibilitySpace):
    """Multimodal space with configurable threshold parameters."""
    
    def __init__(self, lambda_reg: float = 1.0, 
                 threshold_sigma: float = 2.0,
                 threshold_multiplier: float = 3.0):
        super().__init__(lambda_reg)
        self.threshold_sigma = threshold_sigma
        self.threshold_multiplier = threshold_multiplier
    
    def train(self, audio_embeddings, video_embeddings, min_samples=2):
        """Train with custom threshold parameters."""
        common_phonemes = set(audio_embeddings.keys()) & set(video_embeddings.keys())
        
        if not common_phonemes:
            return {}
        
        all_distances = []
        
        for phoneme in common_phonemes:
            audio_emb = audio_embeddings[phoneme]
            video_emb = video_embeddings[phoneme]
            
            if audio_emb.ndim == 1:
                audio_emb = audio_emb.reshape(1, -1)
            if video_emb.ndim == 1:
                video_emb = video_emb.reshape(1, -1)
            
            if audio_emb.shape[0] != video_emb.shape[0]:
                audio_emb = np.mean(audio_emb, axis=0, keepdims=True)
                video_emb = np.mean(video_emb, axis=0, keepdims=True)
            
            self.centroids_audio[phoneme] = np.mean(audio_emb, axis=0)
            self.centroids_video[phoneme] = np.mean(video_emb, axis=0)
            
            try:
                W = self._ridge_regression(audio_emb, video_emb)
                self.W_maps[phoneme] = W
                
                predicted_video = audio_emb @ W.T
                errors = np.linalg.norm(predicted_video - video_emb, axis=1)
                
                mean_error = np.mean(errors)
                std_error = np.std(errors) if len(errors) > 1 else mean_error * 0.5
                
                # Custom threshold calculation
                stat_threshold = mean_error + self.threshold_sigma * std_error
                mult_threshold = mean_error * self.threshold_multiplier
                
                self.thresholds[phoneme] = max(stat_threshold, mult_threshold)
                all_distances.extend(errors.tolist())
                
            except Exception:
                pass
        
        if all_distances:
            mean_dist = np.mean(all_distances)
            std_dist = np.std(all_distances)
            stat_global = mean_dist + self.threshold_sigma * std_dist
            mult_global = mean_dist * self.threshold_multiplier
            self.global_threshold = max(stat_global, mult_global)
        
        return {}
    
    def verify_silent(self, test_audio, test_video) -> Dict:
        """Verify without printing."""
        common_phonemes = set(test_audio.keys()) & set(test_video.keys()) & set(self.W_maps.keys())
        
        if not common_phonemes:
            return {"compatibility_ratio": 0.0, "confidence": 0.0, "average_error": float('inf')}
        
        compatible_count = 0
        total_error = 0.0
        
        for phoneme in common_phonemes:
            audio_emb = test_audio[phoneme]
            video_emb = test_video[phoneme]
            
            if audio_emb.ndim == 2:
                audio_emb = np.mean(audio_emb, axis=0)
            if video_emb.ndim == 2:
                video_emb = np.mean(video_emb, axis=0)
            
            error, threshold, is_compatible = self.compute_compatibility_score(
                audio_emb, video_emb, phoneme
            )
            
            total_error += error
            if is_compatible:
                compatible_count += 1
        
        total_phonemes = len(common_phonemes)
        compatibility_ratio = compatible_count / total_phonemes
        avg_error = total_error / total_phonemes
        
        if compatibility_ratio >= 0.7:
            confidence = min(85 + (compatibility_ratio - 0.7) * 50, 99)
        elif compatibility_ratio >= 0.5:
            confidence = 50 + (compatibility_ratio - 0.5) * 70
        elif compatibility_ratio >= 0.3:
            confidence = 25 + (compatibility_ratio - 0.3) * 50
        else:
            confidence = compatibility_ratio * 100
        
        return {
            "compatibility_ratio": compatibility_ratio,
            "confidence": confidence,
            "average_error": avg_error,
            "compatible_phonemes": compatible_count,
            "total_phonemes": total_phonemes
        }


def evaluate_config(
    sigma: float,
    multiplier: float,
    lambda_reg: float,
    train_audio: Dict,
    train_video: Dict,
    test_real_videos: List[str],
    test_fake_videos: List[str],
    person: str
) -> Dict:
    """Evaluate a single configuration."""
    
    # Train model
    space = ConfigurableMultimodalSpace(
        lambda_reg=lambda_reg,
        threshold_sigma=sigma,
        threshold_multiplier=multiplier
    )
    space.train(train_audio, train_video, min_samples=1)
    
    # Test on Real held-out (per-video, then average)
    real_results = []
    for vid in test_real_videos:
        audio, video = load_video_embeddings_single(vid, person)
        if audio and video:
            result = space.verify_silent(audio, video)
            real_results.append(result)
    
    # Test on Fake (per-video, then average)
    fake_results = []
    for vid in test_fake_videos:
        audio, video = load_video_embeddings_single(vid, person)
        if audio and video:
            result = space.verify_silent(audio, video)
            fake_results.append(result)
    
    # Aggregate
    if real_results:
        avg_real_compat = np.mean([r["compatibility_ratio"] for r in real_results])
        avg_real_conf = np.mean([r["confidence"] for r in real_results])
        avg_real_error = np.mean([r["average_error"] for r in real_results])
    else:
        avg_real_compat = 0
        avg_real_conf = 0
        avg_real_error = float('inf')
    
    if fake_results:
        avg_fake_compat = np.mean([r["compatibility_ratio"] for r in fake_results])
        avg_fake_conf = np.mean([r["confidence"] for r in fake_results])
        avg_fake_error = np.mean([r["average_error"] for r in fake_results])
    else:
        avg_fake_compat = 0
        avg_fake_conf = 0
        avg_fake_error = 0
    
    # Compute score: maximize separation
    # We want: high real_compat, low fake_compat
    separation = avg_real_compat - avg_fake_compat
    conf_diff = (avg_real_conf - avg_fake_conf) / 100
    
    # Combined score (separation is most important)
    score = separation * 100 + conf_diff * 10
    
    return {
        "sigma": sigma,
        "multiplier": multiplier,
        "lambda_reg": lambda_reg,
        "real_compat": float(avg_real_compat),
        "real_conf": float(avg_real_conf),
        "real_error": float(avg_real_error),
        "fake_compat": float(avg_fake_compat),
        "fake_conf": float(avg_fake_conf),
        "fake_error": float(avg_fake_error),
        "separation": float(separation),
        "score": float(score),
        "n_real_tested": len(real_results),
        "n_fake_tested": len(fake_results)
    }


def run_grid_search_for_person(person: str) -> Tuple[Dict, List[Dict]]:
    """Run grid search for a single person."""
    
    # Generate IDs based on person
    prefix = person[0]  # 't' for trump, 'b' for biden
    train_ids = [f"{prefix}-{i:02d}" for i in range(9, 15)]  # X-09 to X-14
    test_real_ids = [f"{prefix}-15"]  # X-15
    test_fake_ids = [f"{prefix}-{i:02d}" for i in range(0, 8)]  # X-00 to X-07
    
    print(f"\n{'='*80}")
    print(f"🔍 GRID SEARCH FOR: {person.upper()}")
    print(f"{'='*80}")
    print(f"Training: {train_ids}")
    print(f"Test Real: {test_real_ids}")
    print(f"Test Fake: {test_fake_ids}")
    print()
    
    # Load training data
    print("Loading training data...")
    train_audio, train_video = load_videos_aggregated(train_ids, person)
    print(f"✓ Loaded {len(train_audio)} phonemes from training")
    print()
    
    # Grid search
    total_configs = len(THRESHOLD_SIGMA_VALUES) * len(THRESHOLD_MULTIPLIER_VALUES) * len(LAMBDA_REG_VALUES)
    print(f"Testing {total_configs} configurations...")
    print()
    
    results = []
    best_score = float('-inf')
    best_config = None
    
    for i, (sigma, mult, lam) in enumerate(product(
        THRESHOLD_SIGMA_VALUES,
        THRESHOLD_MULTIPLIER_VALUES,
        LAMBDA_REG_VALUES
    )):
        result = evaluate_config(
            sigma, mult, lam,
            train_audio, train_video,
            test_real_ids, test_fake_ids,
            person
        )
        result["person"] = person
        results.append(result)
        
        if result["score"] > best_score:
            best_score = result["score"]
            best_config = result
        
        # Progress (every 100 configs or if best)
        if i % 100 == 0 or result == best_config:
            status = "🏆" if result == best_config else "  "
            print(f"{status} [{i+1:4d}/{total_configs}] σ={sigma:5.1f}, mult={mult:5.1f}, λ={lam:6.2f} → "
                  f"Real:{result['real_compat']*100:5.1f}% Fake:{result['fake_compat']*100:5.1f}% "
                  f"Sep:{result['separation']*100:+6.1f}% Score:{result['score']:+7.1f}")
    
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print()
    print(f"📊 TOP 5 FOR {person.upper()}")
    print("-" * 65)
    
    for i, r in enumerate(results[:5]):
        print(f"{i+1:<3} σ={r['sigma']:>5.1f}, mult={r['multiplier']:>5.1f}, λ={r['lambda_reg']:>6.2f} → "
              f"Real:{r['real_compat']*100:5.1f}% Fake:{r['fake_compat']*100:5.1f}% "
              f"Sep:{r['separation']*100:+6.1f}%")
    
    return best_config, results


def main():
    print("=" * 80)
    print("🔍 EXTENSIVE GRID SEARCH - TRUMP & BIDEN")
    print("=" * 80)
    print()
    print(f"Persons: {PERSONS}")
    print(f"Sigma values: {len(THRESHOLD_SIGMA_VALUES)}")
    print(f"Multiplier values: {len(THRESHOLD_MULTIPLIER_VALUES)}")
    print(f"Lambda values: {len(LAMBDA_REG_VALUES)}")
    total = len(THRESHOLD_SIGMA_VALUES) * len(THRESHOLD_MULTIPLIER_VALUES) * len(LAMBDA_REG_VALUES)
    print(f"Total configs per person: {total}")
    print(f"Total configs overall: {total * len(PERSONS)}")
    
    all_results = {}
    all_configs = []
    
    for person in PERSONS:
        best_config, results = run_grid_search_for_person(person)
        all_results[person] = {
            "best_config": best_config,
            "all_results": results
        }
        all_configs.extend(results)
    
    # Find globally best config (averaged across both persons)
    print()
    print("=" * 80)
    print("📊 COMBINED RESULTS")
    print("=" * 80)
    
    # Group by config
    config_scores = defaultdict(list)
    for r in all_configs:
        key = (r["sigma"], r["multiplier"], r["lambda_reg"])
        config_scores[key].append(r["score"])
    
    # Average scores
    avg_scores = []
    for key, scores in config_scores.items():
        avg_scores.append({
            "sigma": key[0],
            "multiplier": key[1],
            "lambda_reg": key[2],
            "avg_score": np.mean(scores),
            "scores": scores
        })
    
    avg_scores.sort(key=lambda x: x["avg_score"], reverse=True)
    
    print()
    print(f"{'Rank':<5} {'σ':>6} {'mult':>6} {'λ':>7} {'Trump':>8} {'Biden':>8} {'Avg':>8}")
    print("-" * 55)
    
    for i, r in enumerate(avg_scores[:10]):
        trump_score = r["scores"][0] if len(r["scores"]) > 0 else 0
        biden_score = r["scores"][1] if len(r["scores"]) > 1 else 0
        print(f"{i+1:<5} {r['sigma']:>6.1f} {r['multiplier']:>6.1f} {r['lambda_reg']:>7.2f} "
              f"{trump_score:>+8.1f} {biden_score:>+8.1f} {r['avg_score']:>+8.1f}")
    
    # Best overall config
    best_overall = avg_scores[0] if avg_scores else None
    
    print()
    print("=" * 80)
    print("🏆 BEST OVERALL CONFIGURATION")
    print("=" * 80)
    
    if best_overall:
        print(f"""
THRESHOLD_SIGMA = {best_overall['sigma']}
THRESHOLD_MULTIPLIER = {best_overall['multiplier']}
LAMBDA_REG = {best_overall['lambda_reg']}

Average Score: {best_overall['avg_score']:+.1f}
""")
    
    # Per-person best
    for person in PERSONS:
        bc = all_results[person]["best_config"]
        if bc:
            print(f"Best for {person.upper()}:")
            print(f"  σ={bc['sigma']}, mult={bc['multiplier']}, λ={bc['lambda_reg']}")
            print(f"  Real: {bc['real_compat']*100:.1f}%, Fake: {bc['fake_compat']*100:.1f}%, Sep: {bc['separation']*100:+.1f}%")
            print()
    
    # Save results
    output_file = Path(__file__).parent / "grid_search_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "best_overall": best_overall,
            "top_10_averaged": avg_scores[:10],
            "per_person": all_results,
            "config": {
                "persons": PERSONS,
                "sigma_values": THRESHOLD_SIGMA_VALUES,
                "multiplier_values": THRESHOLD_MULTIPLIER_VALUES,
                "lambda_values": LAMBDA_REG_VALUES
            }
        }, f, indent=2, default=float)
    
    print(f"📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
