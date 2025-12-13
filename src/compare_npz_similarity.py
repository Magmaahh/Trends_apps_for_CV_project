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
    print(f"  ✓ Loaded {len(embeddings1)} phoneme types")
    
    print(f"Loading embeddings from {file2.name}...")
    embeddings2 = load_embeddings(file2)
    print(f"  ✓ Loaded {len(embeddings2)} phoneme types")
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


def print_results(
    similarities: Dict[str, float],
    global_similarity: float,
    only_in_1: List[str],
    only_in_2: List[str],
    file1_name: str,
    file2_name: str
) -> None:
    """
    Pretty print similarity results.
    
    Args:
        similarities: Dictionary of per-phoneme similarities
        global_similarity: Overall similarity score
        only_in_1: Phonemes only in first file
        only_in_2: Phonemes only in second file
        file1_name: Name of first file
        file2_name: Name of second file
    """
    print("=" * 80)
    print(f"COSINE SIMILARITY COMPARISON")
    print("=" * 80)
    print(f"File 1: {file1_name}")
    print(f"File 2: {file2_name}")
    print()
    
    print(f"GLOBAL SIMILARITY: {global_similarity:.4f}")
    print(f"Common phonemes analyzed: {len(similarities)}")
    print()
    
    if similarities:
        # Sort phonemes by similarity (highest first)
        sorted_phonemes = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("PER-PHONEME SIMILARITIES (sorted by score):")
        print("-" * 80)
        for phoneme, sim in sorted_phonemes:
            # Visual indicator
            if sim >= 0.9:
                indicator = "✓✓✓"  # Excellent
            elif sim >= 0.75:
                indicator = "✓✓ "  # Good
            elif sim >= 0.5:
                indicator = "✓  "  # Moderate
            elif sim >= 0.25:
                indicator = "~  "  # Low
            else:
                indicator = "✗  "  # Poor
            
            print(f"  {indicator} {phoneme:8s}: {sim:7.4f}")
        
        print()
        
        # Statistics
        high_sim = sum(1 for s in similarities.values() if s >= 0.75)
        medium_sim = sum(1 for s in similarities.values() if 0.5 <= s < 0.75)
        low_sim = sum(1 for s in similarities.values() if s < 0.5)
        
        print("STATISTICS:")
        print(f"  High similarity (≥ 0.75):   {high_sim} phonemes")
        print(f"  Medium similarity (0.5-0.75): {medium_sim} phonemes")
        print(f"  Low similarity (< 0.5):      {low_sim} phonemes")
        print()
    
    # Report phonemes only in one file
    if only_in_1:
        print(f"PHONEMES ONLY IN {file1_name}:")
        print(f"  {', '.join(sorted(only_in_1))}")
        print()
    
    if only_in_2:
        print(f"PHONEMES ONLY IN {file2_name}:")
        print(f"  {', '.join(sorted(only_in_2))}")
        print()
    
    print("=" * 80)


def main():
    """Main function."""
    # ========================================================================
    # CONFIGURE FILE PATHS HERE
    # ========================================================================
    file1 = Path("../signatures/s1/voice_profile_s1.npz")
    file2 = Path("../test/s1/voice_sig.npz")
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
