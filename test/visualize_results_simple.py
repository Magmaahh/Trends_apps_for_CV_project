#!/usr/bin/env python3
"""
Simple visualization of classifier results using saved data
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def load_results():
    """Load saved results"""
    results_path = Path("test/artifact_classifier_results/artifact_classifier_results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def plot_test_results(results, output_dir):
    """
    Plot test set results
    """
    print("\nCreating test results visualization...")
    
    test_data = results['test_videos']
    ids = test_data['ids']
    y_true = test_data['true_labels']
    y_pred = test_data['predictions']
    probs = test_data['probabilities']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix (top-left)
    cm = np.array(results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0], 
                cbar=False, square=True, linewidths=2, linecolor='black')
    axes[0,0].set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('True Label', fontsize=14, fontweight='bold')
    axes[0,0].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    axes[0,0].set_xticklabels(['Fake', 'Real'], fontsize=12)
    axes[0,0].set_yticklabels(['Fake', 'Real'], fontsize=12)
    
    # Add text annotations
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i,j] / cm[i].sum() * 100
            axes[0,0].text(j+0.5, i+0.7, f'({percentage:.0f}%)', 
                          ha='center', va='center', fontsize=10, color='gray')
    
    # 2. Prediction Probabilities (top-right)
    x_pos = np.arange(len(ids))
    colors = ['green' if y_pred[i] == y_true[i] else 'red' for i in range(len(ids))]
    
    bars = axes[0,1].bar(x_pos, probs, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0,1].axhline(y=0.5, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Decision Threshold')
    axes[0,1].set_xlabel('Test Video', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Probability (Real)', fontsize=14, fontweight='bold')
    axes[0,1].set_title('Prediction Probabilities', fontsize=16, fontweight='bold')
    axes[0,1].set_xticks(x_pos)
    axes[0,1].set_xticklabels(ids, rotation=45, ha='right', fontsize=11)
    axes[0,1].set_ylim([0, 1])
    axes[0,1].grid(axis='y', alpha=0.3)
    axes[0,1].legend(fontsize=11)
    
    # Add labels on bars
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        label = 'Real' if y_true[i] == 1 else 'Fake'
        pred_label = 'Real' if y_pred[i] == 1 else 'Fake'
        correct = '✓' if y_pred[i] == y_true[i] else '✗'
        axes[0,1].text(bar.get_x() + bar.get_width()/2, prob + 0.03, 
                      f'{label}\n{correct}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 3. Metrics Summary (bottom-left)
    axes[1,0].axis('off')
    
    metrics_text = f"""
    📊 PERFORMANCE METRICS
    {'='*40}
    
    Accuracy:     {results['accuracy']:.1%}  {'✅' if results['accuracy'] > 0.7 else '❌'}
    Precision:    {results['precision']:.1%}  (Fake detection)
    Recall:       {results['recall']:.1%}  (Real detection)
    F1 Score:     {results['f1_score']:.3f}
    ROC-AUC:      {results['roc_auc']:.3f}  {'✅ Perfect!' if results['roc_auc'] == 1.0 else ''}
    
    {'='*40}
    Cross-Validation (5-fold):
    Mean: {np.mean(results['cv_scores']):.1%} ± {np.std(results['cv_scores']):.1%}
    
    Scores: {', '.join([f'{s:.1%}' for s in results['cv_scores']])}
    """
    
    axes[1,0].text(0.1, 0.5, metrics_text, fontsize=12, family='monospace',
                  verticalalignment='center', bbox=dict(boxstyle='round', 
                  facecolor='wheat', alpha=0.5))
    
    # 4. Test Set Breakdown (bottom-right)
    # Count correct vs wrong predictions
    correct_fake = sum(1 for i in range(len(ids)) if y_true[i] == 0 and y_pred[i] == 0)
    wrong_fake = sum(1 for i in range(len(ids)) if y_true[i] == 0 and y_pred[i] == 1)
    correct_real = sum(1 for i in range(len(ids)) if y_true[i] == 1 and y_pred[i] == 1)
    wrong_real = sum(1 for i in range(len(ids)) if y_true[i] == 1 and y_pred[i] == 0)
    
    categories = ['Fake\nCorrect', 'Fake\nWrong', 'Real\nCorrect', 'Real\nWrong']
    values = [correct_fake, wrong_fake, correct_real, wrong_real]
    colors_bar = ['lightgreen', 'salmon', 'green', 'red']
    
    bars = axes[1,1].bar(categories, values, color=colors_bar, edgecolor='black', linewidth=2, alpha=0.8)
    axes[1,1].set_ylabel('Count', fontsize=14, fontweight='bold')
    axes[1,1].set_title('Prediction Breakdown', fontsize=16, fontweight='bold')
    axes[1,1].set_ylim([0, max(values) + 1])
    axes[1,1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                      f'{val}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    plt.suptitle('🎯 Artifact-Based Deepfake Classifier - Results Summary', 
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'results_summary.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    
    return output_path


def plot_comparison_baseline(results, output_dir):
    """
    Plot comparison with baseline (Wav2Vec2)
    """
    print("\nCreating baseline comparison...")
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    methods = ['Wav2Vec2\n(Baseline)', 'Artifact Features\n(Our Method)']
    accuracies = [0.50, results['accuracy']]  # Baseline vs ours
    colors = ['red', 'green']
    
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=3)
    
    ax.set_ylabel('Accuracy', fontsize=16, fontweight='bold')
    ax.set_title('🎯 Deepfake Detection: Baseline vs Artifact-Based', fontsize=18, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Random Guess')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        label = f'{acc:.0%}'
        if acc > 0.7:
            label += '\n✅ SUCCESS'
        elif acc > 0.5:
            label += '\n🟡 Moderate'
        else:
            label += '\n❌ Failure'
        
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
               label, ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    # Add improvement annotation
    improvement = (accuracies[1] - accuracies[0]) / accuracies[0] * 100
    ax.annotate(f'+{improvement:.0f}% improvement', 
               xy=(0.5, 0.7), xytext=(0.5, 0.85),
               fontsize=16, fontweight='bold', color='green',
               ha='center',
               arrowprops=dict(arrowstyle='->', lw=3, color='green'))
    
    # Add explanation text
    explanation = """
    Wav2Vec2 Problem:
    • Identity-oriented (recognizes WHO speaks)
    • Real-Fake overlap: 0.82-0.84
    • Cannot detect synthesis artifacts
    
    Artifact Features Solution:
    • Authenticity-oriented (detects HOW it was created)
    • LFCC, Phase, Harmonic, Formant, Spectral, Energy
    • Captures synthesis artifacts
    """
    
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
           family='monospace')
    
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    output_path = output_dir / 'baseline_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def main():
    """Main function"""
    output_dir = Path("test/artifact_classifier_results")
    
    print("=" * 80)
    print("VISUALIZING CLASSIFIER RESULTS")
    print("=" * 80)
    
    # Load results
    results = load_results()
    
    print(f"\nLoaded results:")
    print(f"  Accuracy: {results['accuracy']:.1%}")
    print(f"  Test videos: {len(results['test_videos']['ids'])}")
    
    # Create visualizations
    summary_path = plot_test_results(results, output_dir)
    plot_comparison_baseline(results, output_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated plots:")
    print(f"  1. results_summary.png      - Complete results overview")
    print(f"  2. baseline_comparison.png  - Comparison with Wav2Vec2 baseline")
    print(f"\nAll plots saved to: {output_dir}")
    print("=" * 80)
    
    # Open the summary
    import subprocess
    subprocess.run(['open', str(summary_path)])


if __name__ == "__main__":
    main()
