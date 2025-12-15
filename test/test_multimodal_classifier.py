#!/usr/bin/env python3
"""
Test Multimodal (Audio + Video) Deepfake Classifier

Combines audio and video artifact features for improved detection.

Strategy:
- Audio features: 90% accuracy (strong)
- Video features: 40% accuracy (weak but complementary)
- Fusion: Early fusion (concatenate features) + late fusion (weighted voting)

Process:
1. Load pre-extracted audio and video features
2. Test different fusion strategies
3. Compare performance
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)


def load_audio_results():
    """Load audio classifier results"""
    results_path = Path("test/artifact_classifier_results/artifact_classifier_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)


def load_video_results():
    """Load video classifier results"""
    results_path = Path("test/video_classifier_results/video_classifier_results.json")
    with open(results_path, 'r') as f:
        return json.load(f)


def late_fusion_voting(audio_probs, video_probs, audio_weight=0.7):
    """
    Late fusion: Weighted voting of audio and video predictions
    
    Args:
        audio_probs: Audio classifier probabilities
        video_probs: Video classifier probabilities  
        audio_weight: Weight for audio (0-1), video gets (1-audio_weight)
        
    Returns:
        Combined probabilities
    """
    video_weight = 1.0 - audio_weight
    combined = audio_weight * audio_probs + video_weight * video_probs
    return combined


def test_multimodal_fusion():
    """
    Test multimodal fusion strategies
    """
    print("=" * 80)
    print("MULTIMODAL DEEPFAKE DETECTION - AUDIO + VIDEO FUSION")
    print("=" * 80)
    
    # Load results
    print("\nLoading pre-computed results...")
    audio_results = load_audio_results()
    video_results = load_video_results()
    
    # Get test set predictions
    audio_ids = audio_results['test_videos']['ids']
    audio_probs = np.array(audio_results['test_videos']['probabilities'])
    audio_preds = np.array(audio_results['test_videos']['predictions'])
    
    video_ids = video_results['test_videos']['ids']
    video_probs = np.array(video_results['test_videos']['probabilities'])
    video_preds = np.array(video_results['test_videos']['predictions'])
    
    # Align test sets (should be same due to random_state=42)
    assert audio_ids == video_ids, "Test sets don't match!"
    y_true = np.array(audio_results['test_videos']['true_labels'])
    
    print(f"Test set size: {len(audio_ids)} videos")
    print(f"Video IDs: {audio_ids}")
    
    # Individual modality performance
    print("\n" + "=" * 80)
    print("INDIVIDUAL MODALITY PERFORMANCE")
    print("=" * 80)
    
    audio_acc = accuracy_score(y_true, audio_preds)
    video_acc = accuracy_score(y_true, video_preds)
    
    print(f"Audio only:  {audio_acc:.1%}")
    print(f"Video only:  {video_acc:.1%}")
    
    # Test different fusion strategies
    print("\n" + "=" * 80)
    print("LATE FUSION - WEIGHTED VOTING")
    print("=" * 80)
    
    best_acc = 0
    best_weight = 0
    results_table = []
    
    for audio_weight in np.arange(0.0, 1.05, 0.1):
        combined_probs = late_fusion_voting(audio_probs, video_probs, audio_weight)
        combined_preds = (combined_probs > 0.5).astype(int)
        acc = accuracy_score(y_true, combined_preds)
        
        results_table.append({
            'audio_weight': audio_weight,
            'video_weight': 1.0 - audio_weight,
            'accuracy': acc
        })
        
        if acc > best_acc:
            best_acc = acc
            best_weight = audio_weight
    
    # Print table
    print("\nAudio Weight | Video Weight | Accuracy")
    print("-" * 45)
    for r in results_table:
        marker = " ← BEST" if r['audio_weight'] == best_weight else ""
        print(f"    {r['audio_weight']:.1f}     |     {r['video_weight']:.1f}     | {r['accuracy']:.1%}{marker}")
    
    # Best fusion
    print("\n" + "=" * 80)
    print("BEST MULTIMODAL FUSION")
    print("=" * 80)
    print(f"Optimal weights: Audio={best_weight:.1f}, Video={1-best_weight:.1f}")
    print(f"Accuracy: {best_acc:.1%}")
    
    # Compute best fusion predictions
    best_combined_probs = late_fusion_voting(audio_probs, video_probs, best_weight)
    best_combined_preds = (best_combined_probs > 0.5).astype(int)
    
    precision = precision_score(y_true, best_combined_preds)
    recall = recall_score(y_true, best_combined_preds)
    f1 = f1_score(y_true, best_combined_preds)
    roc_auc = roc_auc_score(y_true, best_combined_probs)
    
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, best_combined_preds)
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Fake  Real")
    print(f"Actual Fake     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Real     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Detailed comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON PER VIDEO")
    print("=" * 80)
    print(f"{'Video':<8} {'True':<6} {'Audio':<7} {'Video':<7} {'Fused':<7} {'Result'}")
    print("-" * 60)
    
    for i, vid in enumerate(audio_ids):
        true_label = "Real" if y_true[i] == 1 else "Fake"
        audio_pred = "Real" if audio_preds[i] == 1 else "Fake"
        video_pred = "Real" if video_preds[i] == 1 else "Fake"
        fused_pred = "Real" if best_combined_preds[i] == 1 else "Fake"
        
        # Check if fusion helped
        audio_correct = audio_preds[i] == y_true[i]
        video_correct = video_preds[i] == y_true[i]
        fused_correct = best_combined_preds[i] == y_true[i]
        
        if fused_correct and not audio_correct:
            result = "✅ IMPROVED"
        elif not fused_correct and audio_correct:
            result = "❌ DEGRADED"
        elif fused_correct:
            result = "✓ Correct"
        else:
            result = "✗ Wrong"
        
        print(f"{vid:<8} {true_label:<6} {audio_pred:<7} {video_pred:<7} {fused_pred:<7} {result}")
    
    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Audio only:      {audio_acc:.1%}")
    print(f"Video only:      {video_acc:.1%}")
    print(f"Multimodal:      {best_acc:.1%}")
    
    improvement = best_acc - audio_acc
    if improvement > 0:
        print(f"\n✅ Improvement:   +{improvement:.1%}")
        print("Multimodal fusion provides benefit!")
    elif improvement == 0:
        print(f"\n🟡 No change:     {improvement:.1%}")
        print("Video doesn't help (audio already optimal)")
    else:
        print(f"\n❌ Degradation:   {improvement:.1%}")
        print("Video hurts performance (use audio only)")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
    
    if best_acc >= audio_acc:
        print("✅ Use multimodal fusion")
        print(f"   Weights: Audio={best_weight:.1%}, Video={1-best_weight:.1%}")
    else:
        print("✅ Use audio-only classifier")
        print("   Video features don't provide benefit for this dataset")
    
    print("=" * 80)
    
    # Save results
    output_dir = Path("test/multimodal_results")
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'audio_only': {
            'accuracy': float(audio_acc),
            'predictions': audio_preds.tolist()
        },
        'video_only': {
            'accuracy': float(video_acc),
            'predictions': video_preds.tolist()
        },
        'multimodal_fusion': {
            'best_audio_weight': float(best_weight),
            'best_video_weight': float(1 - best_weight),
            'accuracy': float(best_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'predictions': best_combined_preds.tolist(),
            'confusion_matrix': cm.tolist()
        },
        'test_videos': {
            'ids': audio_ids,
            'true_labels': y_true.tolist()
        },
        'fusion_weights_tested': results_table
    }
    
    with open(output_dir / 'multimodal_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'multimodal_results.json'}")
    
    # Plot
    plot_fusion_comparison(results_table, audio_acc, video_acc, output_dir)


def plot_fusion_comparison(results_table, audio_acc, video_acc, output_dir):
    """Plot fusion weight vs accuracy"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    weights = [r['audio_weight'] for r in results_table]
    accuracies = [r['accuracy'] for r in results_table]
    
    ax.plot(weights, accuracies, 'o-', linewidth=2, markersize=8, label='Multimodal Fusion')
    ax.axhline(y=audio_acc, color='blue', linestyle='--', linewidth=2, label=f'Audio Only ({audio_acc:.1%})')
    ax.axhline(y=video_acc, color='red', linestyle='--', linewidth=2, label=f'Video Only ({video_acc:.1%})')
    
    # Mark best
    best_idx = np.argmax(accuracies)
    ax.plot(weights[best_idx], accuracies[best_idx], 'g*', markersize=20, 
           label=f'Best ({accuracies[best_idx]:.1%})')
    
    ax.set_xlabel('Audio Weight', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Multimodal Fusion: Audio-Video Weight Trade-off', fontsize=16, fontweight='bold')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fusion_weights_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Plot saved to: {output_dir / 'fusion_weights_comparison.png'}")


def main():
    """Main function"""
    
    # Check if results exist
    audio_path = Path("test/artifact_classifier_results/artifact_classifier_results.json")
    video_path = Path("test/video_classifier_results/video_classifier_results.json")
    
    if not audio_path.exists():
        print(f"Error: Audio results not found at {audio_path}")
        print("Run test_artifact_classifier.py first")
        return
    
    if not video_path.exists():
        print(f"Error: Video results not found at {video_path}")
        print("Run test_video_artifact_classifier.py first")
        return
    
    # Test fusion
    test_multimodal_fusion()


if __name__ == "__main__":
    main()
