"""
Multi-Metric Analysis for REAL vs FAKE Discrimination

This script computes multiple metrics beyond cosine similarity to find
which features best discriminate between REAL and FAKE videos.

Metrics computed:
1. Cosine Similarity
2. L2 Distance (Euclidean)
3. Statistical Features (mean, std, coefficient of variation)
4. Distribution Metrics (skewness, kurtosis)
5. Cross-Modal Consistency (audio-video alignment)

Usage:
    python multi_metric_analysis.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
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

OUTPUT_DIR = Path("visualizations/multi_metrics")

# =============================================================================


def load_embeddings_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {}
    
    for phoneme in data.files:
        emb = data[phoneme]
        if emb.ndim == 2:
            emb = np.mean(emb, axis=0)  # Average if multiple
        embeddings[phoneme] = emb
    
    return embeddings


def load_category_embeddings(
    video_ids: List[str],
    modality: str = "audio"
) -> Dict[str, List[np.ndarray]]:
    """
    Load embeddings for a category of videos, keeping separate per video.
    
    Returns:
        Dict mapping phoneme -> list of embedding vectors (one per video)
    """
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
    """
    Compute multiple metrics between reference and test embeddings.
    
    Args:
        reference_embs: List of reference embedding vectors
        test_embs: List of test embedding vectors
        
    Returns:
        Dictionary of metric values
    """
    # Compute reference centroid
    ref_centroid = np.mean(reference_embs, axis=0)
    
    # Compute test centroid
    test_centroid = np.mean(test_embs, axis=0)
    
    metrics = {}
    
    # 1. Cosine Similarity (centroid-to-centroid)
    metrics['cosine_sim'] = 1 - cosine(ref_centroid, test_centroid)
    
    # 2. L2 Distance (centroid-to-centroid)
    metrics['l2_distance'] = euclidean(ref_centroid, test_centroid)
    
    # 3. Reference statistics
    ref_dists = [euclidean(emb, ref_centroid) for emb in reference_embs]
    metrics['ref_mean_dist'] = np.mean(ref_dists)
    metrics['ref_std_dist'] = np.std(ref_dists)
    metrics['ref_cv'] = metrics['ref_std_dist'] / (metrics['ref_mean_dist'] + 1e-8)
    
    # 4. Test statistics (distance to reference centroid)
    test_dists = [euclidean(emb, ref_centroid) for emb in test_embs]
    metrics['test_mean_dist'] = np.mean(test_dists)
    metrics['test_std_dist'] = np.std(test_dists)
    metrics['test_cv'] = metrics['test_std_dist'] / (metrics['test_mean_dist'] + 1e-8)
    
    # 5. Distribution metrics
    if len(test_dists) > 2:
        metrics['test_skewness'] = skew(test_dists)
        metrics['test_kurtosis'] = kurtosis(test_dists)
    else:
        metrics['test_skewness'] = 0
        metrics['test_kurtosis'] = 0
    
    # 6. Variance ratio (test/reference)
    metrics['variance_ratio'] = metrics['test_std_dist'] / (metrics['ref_std_dist'] + 1e-8)
    
    # 7. Mean distance ratio (test/reference)
    metrics['mean_dist_ratio'] = metrics['test_mean_dist'] / (metrics['ref_mean_dist'] + 1e-8)
    
    return metrics


def analyze_modality(
    modality: str,
    ref_ids: List[str],
    real_ids: List[str],
    fake_ids: List[str]
) -> Tuple[Dict, Dict]:
    """
    Analyze a single modality (audio or video).
    
    Returns:
        Tuple of (real_metrics, fake_metrics) - each is dict[phoneme -> metrics]
    """
    print(f"\n{'='*80}")
    print(f"📊 Analyzing {modality.upper()} Modality")
    print(f"{'='*80}")
    
    # Load embeddings
    print("Loading embeddings...")
    ref_embs = load_category_embeddings(ref_ids, modality)
    real_embs = load_category_embeddings(real_ids, modality)
    fake_embs = load_category_embeddings(fake_ids, modality)
    
    print(f"Reference phonemes: {len(ref_embs)}")
    print(f"Real phonemes: {len(real_embs)}")
    print(f"Fake phonemes: {len(fake_embs)}")
    
    # Find common phonemes
    common_phonemes = set(ref_embs.keys()) & set(real_embs.keys()) & set(fake_embs.keys())
    print(f"Common phonemes: {len(common_phonemes)}")
    
    # Compute metrics for each phoneme
    real_metrics = {}
    fake_metrics = {}
    
    for phoneme in sorted(common_phonemes):
        ref = ref_embs[phoneme]
        real = real_embs[phoneme]
        fake = fake_embs[phoneme]
        
        # Need at least 2 samples
        if len(ref) < 2 or len(real) < 1 or len(fake) < 1:
            continue
        
        real_metrics[phoneme] = compute_metrics_per_phoneme(ref, real)
        fake_metrics[phoneme] = compute_metrics_per_phoneme(ref, fake)
    
    print(f"✓ Computed metrics for {len(real_metrics)} phonemes")
    
    return real_metrics, fake_metrics


def compute_discrimination_power(
    real_metrics: Dict[str, Dict[str, float]],
    fake_metrics: Dict[str, Dict[str, float]]
) -> Dict[str, Dict]:
    """
    Compute how well each metric discriminates between REAL and FAKE.
    
    Returns:
        Dict mapping metric_name -> {separation, effect_size, real_mean, fake_mean, ...}
    """
    # Collect all metric values
    metric_names = list(next(iter(real_metrics.values())).keys())
    
    discrimination = {}
    
    for metric in metric_names:
        # Collect values across all phonemes
        real_values = [m[metric] for m in real_metrics.values() if metric in m]
        fake_values = [m[metric] for m in fake_metrics.values() if metric in m]
        
        if not real_values or not fake_values:
            continue
        
        real_mean = np.mean(real_values)
        fake_mean = np.mean(fake_values)
        real_std = np.std(real_values)
        fake_std = np.std(fake_values)
        
        # Separation (absolute difference in means)
        separation = abs(real_mean - fake_mean)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((real_std**2 + fake_std**2) / 2)
        effect_size = separation / (pooled_std + 1e-8)
        
        # Normalized separation (separation / average std)
        norm_separation = separation / ((real_std + fake_std) / 2 + 1e-8)
        
        discrimination[metric] = {
            'real_mean': real_mean,
            'real_std': real_std,
            'fake_mean': fake_mean,
            'fake_std': fake_std,
            'separation': separation,
            'effect_size': effect_size,
            'norm_separation': norm_separation,
            'real_values': real_values,
            'fake_values': fake_values
        }
    
    return discrimination


def plot_discrimination_comparison(
    audio_disc: Dict,
    video_disc: Dict
):
    """Create comparison plots for discrimination power."""
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort by effect size
    audio_sorted = sorted(audio_disc.items(), key=lambda x: x[1]['effect_size'], reverse=True)
    video_sorted = sorted(video_disc.items(), key=lambda x: x[1]['effect_size'], reverse=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Metric Discrimination Power: REAL vs FAKE', fontsize=16, fontweight='bold')
    
    # Audio - Effect Size
    ax = axes[0, 0]
    metrics = [m for m, _ in audio_sorted[:8]]
    values = [audio_disc[m]['effect_size'] for m in metrics]
    colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in values]
    ax.barh(metrics, values, color=colors, alpha=0.7)
    ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=11)
    ax.set_title('AUDIO - Top Discriminative Metrics', fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.3, label='Large effect')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Video - Effect Size
    ax = axes[0, 1]
    metrics = [m for m, _ in video_sorted[:8]]
    values = [video_disc[m]['effect_size'] for m in metrics]
    colors = ['green' if v > 0.5 else 'orange' if v > 0.2 else 'red' for v in values]
    ax.barh(metrics, values, color=colors, alpha=0.7)
    ax.set_xlabel('Effect Size (Cohen\'s d)', fontsize=11)
    ax.set_title('VIDEO - Top Discriminative Metrics', fontweight='bold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='x', alpha=0.3)
    
    # Audio - Boxplot comparison for top 3 metrics
    ax = axes[1, 0]
    top_3_audio = [m for m, _ in audio_sorted[:3]]
    data_to_plot = []
    labels = []
    for metric in top_3_audio:
        data_to_plot.extend([audio_disc[metric]['real_values'], audio_disc[metric]['fake_values']])
        labels.extend([f'{metric}\nREAL', f'{metric}\nFAKE'])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('blue' if i % 2 == 0 else 'red')
        patch.set_alpha(0.6)
    ax.set_title('AUDIO - Distribution Comparison (Top 3)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    # Video - Boxplot comparison for top 3 metrics
    ax = axes[1, 1]
    top_3_video = [m for m, _ in video_sorted[:3]]
    data_to_plot = []
    labels = []
    for metric in top_3_video:
        data_to_plot.extend([video_disc[metric]['real_values'], video_disc[metric]['fake_values']])
        labels.extend([f'{metric}\nREAL', f'{metric}\nFAKE'])
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor('blue' if i % 2 == 0 else 'red')
        patch.set_alpha(0.6)
    ax.set_title('VIDEO - Distribution Comparison (Top 3)', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = OUTPUT_DIR / "discrimination_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def print_discrimination_report(
    audio_disc: Dict,
    video_disc: Dict
):
    """Print detailed discrimination report."""
    
    print("\n" + "="*80)
    print("📊 DISCRIMINATION POWER ANALYSIS")
    print("="*80)
    
    # Sort by effect size
    audio_sorted = sorted(audio_disc.items(), key=lambda x: x[1]['effect_size'], reverse=True)
    video_sorted = sorted(video_disc.items(), key=lambda x: x[1]['effect_size'], reverse=True)
    
    print("\n🎵 AUDIO MODALITY - Top Discriminative Metrics")
    print("-" * 80)
    print(f"{'Metric':<25} {'Effect Size':>12} {'Separation':>12} {'REAL μ':>10} {'FAKE μ':>10}")
    print("-" * 80)
    
    for metric, stats in audio_sorted[:10]:
        effect = stats['effect_size']
        sep = stats['separation']
        real_m = stats['real_mean']
        fake_m = stats['fake_mean']
        
        marker = "🟢" if effect > 0.5 else "🟡" if effect > 0.2 else "🔴"
        print(f"{marker} {metric:<23} {effect:>12.3f} {sep:>12.3f} {real_m:>10.3f} {fake_m:>10.3f}")
    
    print("\n📹 VIDEO MODALITY - Top Discriminative Metrics")
    print("-" * 80)
    print(f"{'Metric':<25} {'Effect Size':>12} {'Separation':>12} {'REAL μ':>10} {'FAKE μ':>10}")
    print("-" * 80)
    
    for metric, stats in video_sorted[:10]:
        effect = stats['effect_size']
        sep = stats['separation']
        real_m = stats['real_mean']
        fake_m = stats['fake_mean']
        
        marker = "🟢" if effect > 0.5 else "🟡" if effect > 0.2 else "🔴"
        print(f"{marker} {metric:<23} {effect:>12.3f} {sep:>12.3f} {real_m:>10.3f} {fake_m:>10.3f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("  Effect Size: 🟢 >0.5 (Good) | 🟡 0.2-0.5 (Medium) | 🔴 <0.2 (Poor)")
    print("  Higher effect size = better discrimination between REAL and FAKE")
    print("="*80)


def main():
    print("="*80)
    print("🔬 MULTI-METRIC DISCRIMINATION ANALYSIS")
    print("="*80)
    print(f"Person: {PERSON.upper()}")
    print(f"Reference videos: {len(REFERENCE_IDS)}")
    print(f"Real test videos: {len(REAL_IDS)}")
    print(f"Fake videos: {len(FAKE_IDS)}")
    print("="*80)
    
    # Analyze both modalities
    audio_real, audio_fake = analyze_modality("audio", REFERENCE_IDS, REAL_IDS, FAKE_IDS)
    video_real, video_fake = analyze_modality("video", REFERENCE_IDS, REAL_IDS, FAKE_IDS)
    
    # Compute discrimination power
    print("\n" + "="*80)
    print("🎯 Computing Discrimination Power")
    print("="*80)
    
    audio_disc = compute_discrimination_power(audio_real, audio_fake)
    video_disc = compute_discrimination_power(video_real, video_fake)
    
    # Print report
    print_discrimination_report(audio_disc, video_disc)
    
    # Create visualizations
    print("\n" + "="*80)
    print("📈 Creating Visualizations")
    print("="*80)
    
    plot_discrimination_comparison(audio_disc, video_disc)
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
