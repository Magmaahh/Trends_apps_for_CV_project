"""
Script to compute cosine similarity between two .npz files.

This script loads embeddings from two .npz files and computes the cosine
similarity between matching phonemes. It's useful for comparing embeddings
from different videos or comparing test embeddings with gold standards.

Usage:
    python compare_npz_similarity.py <file1.npz> <file2.npz>
"""

# Fix OpenMP duplicate library issue on macOS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple


def load_embeddings(npz_path: Path) -> Dict[str, np.ndarray]:
    """
    Load embeddings from an .npz file.
    
    Args:
        npz_path: Path to the .npz file
        
    Returns:
        Dictionary mapping phoneme names to their embeddings (averaged if multiple)
    """
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {}
    
    # Load phoneme embeddings directly (keys are phoneme names)
    for phoneme in data.files:
        emb_array = data[phoneme]
        
        if isinstance(emb_array, np.ndarray):
            # If multiple embeddings for same phoneme, average them
            if emb_array.ndim == 2:
                embeddings[phoneme] = np.mean(emb_array, axis=0)
            else:
                embeddings[phoneme] = emb_array
    
    return embeddings


def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)
    """
    # Cosine similarity = 1 - cosine distance
    similarity = 1 - cosine(vec1, vec2)
    return float(similarity) if not np.isnan(similarity) else 0.0


def compare_embeddings(
    file1: Path,
    file2: Path
) -> Tuple[Dict[str, float], float, List[str], List[str]]:
    """
    Compare embeddings from two .npz files.
    
    Args:
        file1: Path to first .npz file
        file2: Path to second .npz file
        
    Returns:
        Tuple of (per_phoneme_similarities, global_similarity, phonemes_only_in_file1, phonemes_only_in_file2)
    """
    print(f"Loading embeddings from {file1.name}...")
    embeddings1 = load_embeddings(file1)
    print(f"  Loaded {len(embeddings1)} phoneme types")
    
    print(f"Loading embeddings from {file2.name}...")
    embeddings2 = load_embeddings(file2)
    print(f"  Loaded {len(embeddings2)} phoneme types")
    print()
    
    # Find common phonemes and compute similarities
    common_phonemes = set(embeddings1.keys()) & set(embeddings2.keys())
    phonemes_only_in_1 = set(embeddings1.keys()) - set(embeddings2.keys())
    phonemes_only_in_2 = set(embeddings2.keys()) - set(embeddings1.keys())
    
    similarities = {}
    for phoneme in common_phonemes:
        vec1 = embeddings1[phoneme]
        vec2 = embeddings2[phoneme]
        sim = compute_cosine_similarity(vec1, vec2)
        similarities[phoneme] = sim
    
    # Compute global similarity as mean of all phoneme similarities
    global_sim = float(np.mean(list(similarities.values()))) if similarities else 0.0
    
    return similarities, global_sim, list(phonemes_only_in_1), list(phonemes_only_in_2)


def compute_confidence_and_verdict(similarities: Dict[str, float]) -> Dict[str, any]:
    """
    Compute confidence level and verdict based on similarity scores.
    Primary criterion: percentage of phonemes with similarity ≥ 0.9
    
    Args:
        similarities: Dictionary of per-phoneme similarities
        
    Returns:
        Dictionary with confidence metrics and verdict
    """
    if not similarities:
        return {
            "verdict": "INSUFFICIENT DATA",
            "confidence": "N/A",
            "confidence_score": 0.0,
            "match_probability": 0.0,
            "excellent_count": 0,
            "excellent_percentage": 0.0,
            "weighted_score": 0.0
        }
    
    total_phonemes = len(similarities)
    
    # Count phonemes by category (primary focus on ≥0.9)
    excellent = sum(1 for s in similarities.values() if s >= 0.9)
    acceptable = sum(1 for s in similarities.values() if 0.8 <= s < 0.9)
    questionable = sum(1 for s in similarities.values() if 0.5 <= s < 0.8)
    poor = sum(1 for s in similarities.values() if s < 0.5)
    
    # Calculate percentages
    excellent_pct = (excellent / total_phonemes) * 100
    
    # Weighted score (≥0.9 has much higher weight)
    weighted_score = 0.0
    for sim in similarities.values():
        if sim >= 0.9:
            weighted_score += 1.0  # Full weight
        elif sim >= 0.8:
            weighted_score += 0.3  # Minor contribution
        elif sim >= 0.5:
            weighted_score += 0.1  # Very minor contribution
        # <0.5 contributes 0
    
    weighted_score = (weighted_score / total_phonemes) * 100
    
    # Determine confidence level based on excellent phonemes percentage
    if excellent_pct >= 70:
        confidence = "HIGH"
        confidence_score = 3
        match_probability = min(85 + (excellent_pct - 70) * 0.5, 99)
        verdict = "SAME PERSON"
    elif excellent_pct >= 40:
        confidence = "MEDIUM"
        confidence_score = 2
        match_probability = 50 + (excellent_pct - 40) * 1.0
        verdict = "LIKELY SAME PERSON"
    elif excellent_pct >= 20:
        confidence = "LOW"
        confidence_score = 1
        match_probability = 25 + (excellent_pct - 20) * 1.0
        verdict = "UNCERTAIN"
    else:
        confidence = "VERY LOW"
        confidence_score = 0
        match_probability = excellent_pct
        verdict = "DIFFERENT PERSON"
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "confidence_score": confidence_score,
        "match_probability": match_probability,
        "excellent_count": excellent,
        "excellent_percentage": excellent_pct,
        "acceptable_count": acceptable,
        "questionable_count": questionable,
        "poor_count": poor,
        "weighted_score": weighted_score,
        "total_phonemes": total_phonemes
    }


def print_results(
    similarities: Dict[str, float],
    global_similarity: float,
    only_in_1: List[str],
    only_in_2: List[str],
    file1_name: str,
    file2_name: str
) -> None:
    """
    Pretty print similarity results with focus on ≥0.9 similarity threshold.
    
    Args:
        similarities: Dictionary of per-phoneme similarities
        global_similarity: Overall similarity score
        only_in_1: Phonemes only in first file
        only_in_2: Phonemes only in second file
        file1_name: Name of first file (reference signature)
        file2_name: Name of second file (sample to verify)
    """
    print(f"IDENTITY VERIFICATION ANALYSIS")
    print(f"Reference: {file1_name}")
    print(f"Sample:    {file2_name}")
    
    # Compute confidence and verdict
    metrics = compute_confidence_and_verdict(similarities)
    
    # Print prominent verdict section
    print(f"VERDICT: {metrics['verdict']}")
    
    print(f"Confidence Level:  {metrics['confidence']} ({metrics['excellent_percentage']:.1f}%)")
    print(f"Match Probability: {metrics['match_probability']:.1f}%")
    print(f"Reliable Matches:  {metrics['excellent_count']}/{metrics['total_phonemes']} phonemes (≥0.9 similarity)")
    print(f"Weighted Score:    {metrics['weighted_score']:.1f}/100")
    
    if similarities:
        # Sort phonemes by similarity (highest first)
        sorted_phonemes = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("DETAILED PHONEME ANALYSIS:")
        for phoneme, sim in sorted_phonemes:
            # Visual indicator and category
            if sim >= 0.9:
                indicator = ""
                category = "[EXCELLENT - HIGH CONFIDENCE]"
            elif sim >= 0.8:
                indicator = ""
                category = "[ACCEPTABLE - MINOR WEIGHT]"
            elif sim >= 0.5:
                indicator = ""
                category = "[QUESTIONABLE - MINIMAL WEIGHT]"
            else:
                indicator = ""
                category = "[POOR - NO WEIGHT]"
            
            print(f"  {phoneme:8s}: {sim:7.4f}  {category}")
        
        print()
        
        # Enhanced statistics focused on 0.9 threshold
        print("STATISTICS BREAKDOWN:")
        print(f"  Excellent (≥ 0.9):      {metrics['excellent_count']:3d} phonemes ({metrics['excellent_percentage']:5.1f}%) [PRIMARY CRITERION]")
        print(f"  Acceptable (0.8-0.9):   {metrics['acceptable_count']:3d} phonemes ({(metrics['acceptable_count']/metrics['total_phonemes'])*100:5.1f}%) [Minor weight]")
        print(f"  Questionable (0.5-0.8): {metrics['questionable_count']:3d} phonemes ({(metrics['questionable_count']/metrics['total_phonemes'])*100:5.1f}%) [Minimal weight]")
        print(f"  Poor (< 0.5):           {metrics['poor_count']:3d} phonemes ({(metrics['poor_count']/metrics['total_phonemes'])*100:5.1f}%) [No weight]")
        
        print(f"  Global Similarity (reference): {global_similarity:.4f}")
    
    # Report phonemes only in one file
    if only_in_1:
        print(f"PHONEMES ONLY IN REFERENCE ({file1_name}):")
        print(f"  {', '.join(sorted(only_in_1))}")
    
    if only_in_2:
        print(f"PHONEMES ONLY IN SAMPLE ({file2_name}):")
        print(f"  {', '.join(sorted(only_in_2))}")


def main():
    """Main function."""
    # ========================================================================
    # CONFIGURE FILE PATHS HERE
    # ========================================================================
    file1 = Path("test/signatures/s1/voice_profile_s1.npz")
    file2 = Path("test/samples/s1/voice_sig.npz")
    # ========================================================================
    
    # Check if files exist
    if not file1.exists():
        print(f"Error: File not found: {file1}")
        sys.exit(1)
    
    if not file2.exists():
        print(f"Error: File not found: {file2}")
        sys.exit(1)
    
    print("Cosine Similarity Comparison")
    print()
    
    try:
        # Compare embeddings
        similarities, global_sim, only_in_1, only_in_2 = compare_embeddings(
            file1, file2
        )
        
        # Print results
        print_results(
            similarities,
            global_sim,
            only_in_1,
            only_in_2,
            file1.name,
            file2.name
        )
        
        # Save results to file
        output_file = Path("cosine_similarity_results.json")
        import json
        results = {
            "file1": str(file1),
            "file2": str(file2),
            "global_similarity": global_sim,
            "per_phoneme_similarities": similarities,
            "phonemes_only_in_file1": only_in_1,
            "phonemes_only_in_file2": only_in_2
        }
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
