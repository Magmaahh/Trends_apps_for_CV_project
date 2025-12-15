#!/usr/bin/env python3
"""
Analyze Real vs Fake Embedding Overlap

This script analyzes whether current embeddings (Wav2Vec2 audio + visual) 
can distinguish between real and fake videos of the same person.

Hypothesis: Current embeddings are IDENTITY-ORIENTED, not AUTHENTICITY-ORIENTED
→ Real and Fake videos of same person will have similar embeddings

Dataset: Trump/Biden
- Fake: t-00 to t-07, b-00 to b-07
- Real: t-08 to t-15, b-08 to b-15
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from collections import defaultdict
import json


def load_embeddings(npz_path):
    """Load embeddings from .npz file"""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {}
    
    for phoneme in data.files:
        emb_array = data[phoneme]
        
        if isinstance(emb_array, np.ndarray):
            if emb_array.ndim == 2:
                embeddings[phoneme] = np.mean(emb_array, axis=0)
            else:
                embeddings[phoneme] = emb_array
    
    return embeddings


def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity"""
    return 1 - cosine(vec1, vec2)


def analyze_dataset(dataset_path, person='trump', modality='audio'):
    """
    Analyze real vs fake embeddings for a person
    
    Args:
        dataset_path: Path to processed dataset
        person: 'trump' or 'biden'
        modality: 'audio' or 'video'
    
    Returns:
        dict with analysis results
    """
    dataset_path = Path(dataset_path)
    
    # Collect embeddings
    real_embeddings = []
    fake_embeddings = []
    real_ids = []
    fake_ids = []
    
    # Determine prefix and range
    prefix = 't' if person == 'trump' else 'b'
    fake_range = range(0, 8)  # 00-07
    real_range = range(8, 16)  # 08-15
    
    # File suffix based on modality
    suffix = '_audio.npz' if modality == 'audio' else '_video.npz'
    
    print(f"\n{'='*80}")
    print(f"Analyzing {person.upper()} - {modality.upper()} embeddings")
    print(f"{'='*80}")
    
    # Load fake embeddings
    print(f"\nLoading FAKE embeddings ({prefix}-00 to {prefix}-07)...")
    for i in fake_range:
        video_id = f"{prefix}-{i:02d}"
        emb_path = dataset_path / person / video_id / f"{video_id}{suffix}"
        
        if emb_path.exists():
            embeddings = load_embeddings(emb_path)
            if embeddings:
                fake_embeddings.append(embeddings)
                fake_ids.append(video_id)
                print(f"  ✓ {video_id}: {len(embeddings)} phonemes")
        else:
            print(f"  ✗ {video_id}: Not found")
    
    # Load real embeddings
    print(f"\nLoading REAL embeddings ({prefix}-08 to {prefix}-15)...")
    for i in real_range:
        video_id = f"{prefix}-{i:02d}"
        emb_path = dataset_path / person / video_id / f"{video_id}{suffix}"
        
        if emb_path.exists():
            embeddings = load_embeddings(emb_path)
            if embeddings:
                real_embeddings.append(embeddings)
                real_ids.append(video_id)
                print(f"  ✓ {video_id}: {len(embeddings)} phonemes")
        else:
            print(f"  ✗ {video_id}: Not found")
    
    print(f"\nLoaded: {len(fake_embeddings)} fake, {len(real_embeddings)} real")
    
    if len(fake_embeddings) == 0 or len(real_embeddings) == 0:
        print("ERROR: Not enough data to analyze")
        return None
    
    # Find common phonemes across ALL videos
    all_phonemes = set()
    for emb_dict in fake_embeddings + real_embeddings:
        all_phonemes.update(emb_dict.keys())
    
    print(f"Total unique phonemes: {len(all_phonemes)}")
    
    # Compute statistics
    results = {
        'person': person,
        'modality': modality,
        'n_fake': len(fake_embeddings),
        'n_real': len(real_embeddings),
        'phonemes': list(all_phonemes),
        'fake_ids': fake_ids,
        'real_ids': real_ids
    }
    
    return results, fake_embeddings, real_embeddings, all_phonemes


def compute_similarity_matrix(embeddings1, embeddings2, common_phonemes):
    """
    Compute pairwise similarity matrix between two sets of embeddings
    
    Returns:
        similarity_matrix: (n1, n2) array of average similarities
    """
    n1 = len(embeddings1)
    n2 = len(embeddings2)
    
    similarity_matrix = np.zeros((n1, n2))
    
    for i, emb1_dict in enumerate(embeddings1):
        for j, emb2_dict in enumerate(embeddings2):
            # Compute similarity for common phonemes
            sims = []
            for phoneme in common_phonemes:
                if phoneme in emb1_dict and phoneme in emb2_dict:
                    vec1 = emb1_dict[phoneme]
                    vec2 = emb2_dict[phoneme]
                    sim = compute_cosine_similarity(vec1, vec2)
                    sims.append(sim)
            
            if sims:
                similarity_matrix[i, j] = np.mean(sims)
    
    return similarity_matrix


def plot_similarity_distributions(fake_embeddings, real_embeddings, common_phonemes, person, modality, output_dir):
    """Plot similarity distributions"""
    
    print("\nComputing similarity distributions...")
    
    # 1. Fake-Fake similarities
    print("  Computing Fake-Fake similarities...")
    fake_fake_sims = []
    for i in range(len(fake_embeddings)):
        for j in range(i+1, len(fake_embeddings)):
            sims = []
            for phoneme in common_phonemes:
                if phoneme in fake_embeddings[i] and phoneme in fake_embeddings[j]:
                    sim = compute_cosine_similarity(
                        fake_embeddings[i][phoneme],
                        fake_embeddings[j][phoneme]
                    )
                    sims.append(sim)
            if sims:
                fake_fake_sims.append(np.mean(sims))
    
    # 2. Real-Real similarities
    print("  Computing Real-Real similarities...")
    real_real_sims = []
    for i in range(len(real_embeddings)):
        for j in range(i+1, len(real_embeddings)):
            sims = []
            for phoneme in common_phonemes:
                if phoneme in real_embeddings[i] and phoneme in real_embeddings[j]:
                    sim = compute_cosine_similarity(
                        real_embeddings[i][phoneme],
                        real_embeddings[j][phoneme]
                    )
                    sims.append(sim)
            if sims:
                real_real_sims.append(np.mean(sims))
    
    # 3. Real-Fake similarities (CRITICAL)
    print("  Computing Real-Fake similarities...")
    real_fake_sims = []
    for real_emb in real_embeddings:
        for fake_emb in fake_embeddings:
            sims = []
            for phoneme in common_phonemes:
                if phoneme in real_emb and phoneme in fake_emb:
                    sim = compute_cosine_similarity(
                        real_emb[phoneme],
                        fake_emb[phoneme]
                    )
                    sims.append(sim)
            if sims:
                real_fake_sims.append(np.mean(sims))
    
    # Plot distributions
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    bins = np.linspace(0, 1, 50)
    
    ax.hist(fake_fake_sims, bins=bins, alpha=0.5, label=f'Fake-Fake (n={len(fake_fake_sims)})', color='red', edgecolor='black')
    ax.hist(real_real_sims, bins=bins, alpha=0.5, label=f'Real-Real (n={len(real_real_sims)})', color='green', edgecolor='black')
    ax.hist(real_fake_sims, bins=bins, alpha=0.7, label=f'Real-Fake (n={len(real_fake_sims)})', color='orange', edgecolor='black')
    
    ax.axvline(np.mean(fake_fake_sims), color='red', linestyle='--', linewidth=2, label=f'Fake-Fake mean: {np.mean(fake_fake_sims):.3f}')
    ax.axvline(np.mean(real_real_sims), color='green', linestyle='--', linewidth=2, label=f'Real-Real mean: {np.mean(real_real_sims):.3f}')
    ax.axvline(np.mean(real_fake_sims), color='orange', linestyle='--', linewidth=2, label=f'Real-Fake mean: {np.mean(real_fake_sims):.3f}')
    
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{person.upper()} - {modality.upper()} Similarity Distributions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add text box with statistics
    textstr = f'Fake-Fake: {np.mean(fake_fake_sims):.3f} ± {np.std(fake_fake_sims):.3f}\n'
    textstr += f'Real-Real: {np.mean(real_real_sims):.3f} ± {np.std(real_real_sims):.3f}\n'
    textstr += f'Real-Fake: {np.mean(real_fake_sims):.3f} ± {np.std(real_fake_sims):.3f}\n\n'
    textstr += f'⚠️ OVERLAP: {np.mean(real_fake_sims):.1%} similarity\n'
    textstr += f'→ {"HIGH" if np.mean(real_fake_sims) > 0.7 else "LOW"} overlap detected'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{person}_{modality}_similarity_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return {
        'fake_fake_mean': float(np.mean(fake_fake_sims)),
        'fake_fake_std': float(np.std(fake_fake_sims)),
        'real_real_mean': float(np.mean(real_real_sims)),
        'real_real_std': float(np.std(real_real_sims)),
        'real_fake_mean': float(np.mean(real_fake_sims)),
        'real_fake_std': float(np.std(real_fake_sims))
    }


def plot_embedding_space(fake_embeddings, real_embeddings, common_phonemes, person, modality, output_dir):
    """Plot 2D projection of embedding space using PCA/TSNE"""
    
    print("\nCreating 2D embedding space visualization...")
    
    # Collect all embedding vectors
    all_vectors = []
    all_labels = []
    all_ids = []
    
    for i, emb_dict in enumerate(fake_embeddings):
        # Average across phonemes
        vectors = [emb_dict[p] for p in common_phonemes if p in emb_dict]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            all_vectors.append(avg_vector)
            all_labels.append('Fake')
            all_ids.append(f'F{i}')
    
    for i, emb_dict in enumerate(real_embeddings):
        vectors = [emb_dict[p] for p in common_phonemes if p in emb_dict]
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            all_vectors.append(avg_vector)
            all_labels.append('Real')
            all_ids.append(f'R{i}')
    
    X = np.array(all_vectors)
    
    # Apply PCA first for dimensionality reduction
    print(f"  Original dimension: {X.shape[1]}")
    pca = PCA(n_components=min(50, X.shape[0]-1))
    X_pca = pca.fit_transform(X)
    print(f"  PCA reduced to: {X_pca.shape[1]} (variance explained: {pca.explained_variance_ratio_.sum():.2%})")
    
    # Apply TSNE
    print("  Applying t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0]-1))
    X_2d = tsne.fit_transform(X_pca)
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    colors = {'Fake': 'red', 'Real': 'green'}
    for label in ['Fake', 'Real']:
        mask = np.array(all_labels) == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                  c=colors[label], label=label, alpha=0.6, s=100, edgecolors='black')
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(f'{person.upper()} - {modality.upper()} Embedding Space (t-SNE)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f'{person}_{modality}_embedding_space_tsne.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main analysis function"""
    
    # Configuration
    dataset_path = Path("dataset/trump_biden/processed")
    output_dir = Path("test/deepfake_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("REAL VS FAKE EMBEDDING OVERLAP ANALYSIS")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    
    # Run analysis for both persons and both modalities
    all_results = {}
    
    for person in ['trump', 'biden']:
        for modality in ['audio', 'video']:
            
            result = analyze_dataset(dataset_path, person, modality)
            
            if result is None:
                print(f"\n⚠️  Skipping {person} {modality} - insufficient data")
                continue
            
            results, fake_embeddings, real_embeddings, common_phonemes = result
            
            # Compute and plot similarity distributions
            sim_stats = plot_similarity_distributions(
                fake_embeddings, real_embeddings, common_phonemes,
                person, modality, output_dir
            )
            
            results['similarity_stats'] = sim_stats
            
            # Plot embedding space
            plot_embedding_space(
                fake_embeddings, real_embeddings, common_phonemes,
                person, modality, output_dir
            )
            
            all_results[f'{person}_{modality}'] = results
    
    # Save results
    results_file = output_dir / 'analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {results_file}")
    print(f"Plots saved to: {output_dir}")
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - Real-Fake Similarity (CRITICAL METRIC)")
    print(f"{'='*80}")
    
    for key, results in all_results.items():
        if 'similarity_stats' in results:
            stats = results['similarity_stats']
            real_fake_mean = stats['real_fake_mean']
            person, modality = key.split('_')
            
            status = "🔴 HIGH OVERLAP" if real_fake_mean > 0.7 else "🟢 LOW OVERLAP"
            print(f"{person.upper():6s} {modality.upper():5s}: {real_fake_mean:.3f} ± {stats['real_fake_std']:.3f}  {status}")
    
    print(f"\n{'='*80}")
    print("INTERPRETATION")
    print(f"{'='*80}")
    print("If Real-Fake similarity > 0.7:")
    print("  → Current embeddings are IDENTITY-ORIENTED")
    print("  → Real and Fake videos of same person have similar embeddings")
    print("  → System CANNOT distinguish Real from Fake")
    print("  → Need to add ARTIFACT-SPECIFIC features")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
