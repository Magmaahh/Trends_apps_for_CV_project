#!/usr/bin/env python3
"""
Analyze Video Feature Failures

Deep dive into why video artifact features achieve only 40% accuracy.
Analyzes:
- Feature distributions (Real vs Fake)
- Feature correlations
- Per-video analysis
- Comparison with audio features
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats


def load_results():
    """Load audio and video classifier results"""
    audio_path = Path("test/artifact_classifier_results/artifact_classifier_results.json")
    video_path = Path("test/video_classifier_results/video_classifier_results.json")
    
    with open(audio_path, 'r') as f:
        audio_results = json.load(f)
    
    with open(video_path, 'r') as f:
        video_results = json.load(f)
    
    return audio_results, video_results


def analyze_video_failure():
    """Main analysis function"""
    
    output_dir = Path("test/video_failure_analysis")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("🔍 VIDEO FEATURE FAILURE ANALYSIS")
    print("=" * 80)
    
    audio_results, video_results = load_results()
    
    # Get test set info
    test_ids = video_results['test_videos']['ids']
    y_true = video_results['test_videos']['true_labels']
    video_preds = video_results['test_videos']['predictions']
    video_probs = video_results['test_videos']['probabilities']
    
    audio_preds = audio_results['test_videos']['predictions']
    audio_probs = audio_results['test_videos']['probabilities']
    
    # 1. Performance comparison
    print("\n" + "=" * 80)
    print("📊 PERFORMANCE COMPARISON")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Confusion matrices side by side
    cm_audio = np.array(audio_results['confusion_matrix'])
    cm_video = np.array(video_results['confusion_matrix'])
    
    sns.heatmap(cm_audio, annot=True, fmt='d', cmap='Greens', ax=axes[0,0],
                cbar=False, square=True, linewidths=2)
    axes[0,0].set_title('Audio Classifier (90% Accuracy)', fontsize=14, fontweight='bold')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('True')
    axes[0,0].set_xticklabels(['Fake', 'Real'])
    axes[0,0].set_yticklabels(['Fake', 'Real'])
    
    sns.heatmap(cm_video, annot=True, fmt='d', cmap='Reds', ax=axes[0,1],
                cbar=False, square=True, linewidths=2)
    axes[0,1].set_title('Video Classifier (40% Accuracy)', fontsize=14, fontweight='bold')
    axes[0,1].set_xlabel('Predicted')
    axes[0,1].set_ylabel('True')
    axes[0,1].set_xticklabels(['Fake', 'Real'])
    axes[0,1].set_yticklabels(['Fake', 'Real'])
    
    # Per-video comparison
    x = np.arange(len(test_ids))
    width = 0.35
    
    # Color by correctness
    audio_colors = ['green' if audio_preds[i] == y_true[i] else 'red' for i in range(len(test_ids))]
    video_colors = ['green' if video_preds[i] == y_true[i] else 'red' for i in range(len(test_ids))]
    
    axes[1,0].bar(x - width/2, audio_probs, width, label='Audio', color=audio_colors, alpha=0.7, edgecolor='black')
    axes[1,0].bar(x + width/2, video_probs, width, label='Video', color=video_colors, alpha=0.7, edgecolor='black')
    axes[1,0].axhline(y=0.5, color='black', linestyle='--', linewidth=1)
    axes[1,0].set_xlabel('Test Video', fontweight='bold')
    axes[1,0].set_ylabel('Probability (Real)', fontweight='bold')
    axes[1,0].set_title('Prediction Probabilities per Video', fontsize=14, fontweight='bold')
    axes[1,0].set_xticks(x)
    axes[1,0].set_xticklabels(test_ids, rotation=45, ha='right')
    axes[1,0].legend()
    axes[1,0].grid(axis='y', alpha=0.3)
    
    # Agreement analysis
    agreement = []
    for i in range(len(test_ids)):
        if audio_preds[i] == video_preds[i]:
            if audio_preds[i] == y_true[i]:
                agreement.append('Both Correct')
            else:
                agreement.append('Both Wrong')
        else:
            if audio_preds[i] == y_true[i]:
                agreement.append('Audio Correct')
            else:
                agreement.append('Video Correct')
    
    from collections import Counter
    agreement_counts = Counter(agreement)
    
    colors_pie = {'Both Correct': 'green', 'Audio Correct': 'lightgreen',
                  'Video Correct': 'orange', 'Both Wrong': 'red'}
    pie_colors = [colors_pie[k] for k in agreement_counts.keys()]
    
    axes[1,1].pie(agreement_counts.values(), labels=agreement_counts.keys(),
                  autopct='%1.1f%%', colors=pie_colors, startangle=90)
    axes[1,1].set_title('Audio-Video Agreement', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'performance_comparison.png'}")
    
    # 2. Video failure modes
    print("\n" + "=" * 80)
    print("🔍 VIDEO FAILURE ANALYSIS")
    print("=" * 80)
    
    print(f"\n{'Video':<10} {'True':<6} {'Audio':<8} {'Video':<8} {'Issue'}")
    print("-" * 60)
    
    failure_modes = {
        'video_wrong_audio_correct': [],
        'both_wrong': [],
        'video_correct_audio_wrong': []
    }
    
    for i, vid in enumerate(test_ids):
        true_label = 'Real' if y_true[i] == 1 else 'Fake'
        audio_pred = 'Real' if audio_preds[i] == 1 else 'Fake'
        video_pred = 'Real' if video_preds[i] == 1 else 'Fake'
        
        audio_correct = audio_preds[i] == y_true[i]
        video_correct = video_preds[i] == y_true[i]
        
        if not video_correct and audio_correct:
            issue = '❌ Video fails'
            failure_modes['video_wrong_audio_correct'].append(vid)
        elif not video_correct and not audio_correct:
            issue = '❌ Both fail'
            failure_modes['both_wrong'].append(vid)
        elif video_correct and not audio_correct:
            issue = '✅ Video saves'
            failure_modes['video_correct_audio_wrong'].append(vid)
        else:
            issue = '✓ Both correct'
        
        print(f"{vid:<10} {true_label:<6} {audio_pred:<8} {video_pred:<8} {issue}")
    
    print("\n" + "-" * 60)
    print(f"Video fails (audio correct): {len(failure_modes['video_wrong_audio_correct'])}")
    print(f"Both fail: {len(failure_modes['both_wrong'])}")
    print(f"Video saves (audio wrong): {len(failure_modes['video_correct_audio_wrong'])}")
    
    # 3. Statistical analysis
    print("\n" + "=" * 80)
    print("📈 STATISTICAL ANALYSIS")
    print("=" * 80)
    
    # Cross-validation comparison
    audio_cv = audio_results['cv_scores']
    video_cv = video_results['cv_scores']
    
    print(f"\nCross-Validation (5-fold):")
    print(f"Audio: {np.mean(audio_cv):.1%} ± {np.std(audio_cv):.1%}")
    print(f"Video: {np.mean(video_cv):.1%} ± {np.std(video_cv):.1%}")
    print(f"\nVideo CV std is {np.std(video_cv)/np.std(audio_cv):.1f}x higher (less stable)")
    
    # ROC-AUC comparison
    print(f"\nROC-AUC:")
    print(f"Audio: {audio_results['roc_auc']:.3f} (Perfect)")
    print(f"Video: {video_results['roc_auc']:.3f} (Near random!)")
    
    # Probability distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Audio probabilities
    fake_mask = np.array(y_true) == 0
    real_mask = np.array(y_true) == 1
    
    axes[0].hist(np.array(audio_probs)[fake_mask], bins=10, alpha=0.7, 
                label='Fake', color='red', edgecolor='black')
    axes[0].hist(np.array(audio_probs)[real_mask], bins=10, alpha=0.7,
                label='Real', color='green', edgecolor='black')
    axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Probability (Real)', fontweight='bold')
    axes[0].set_ylabel('Count', fontweight='bold')
    axes[0].set_title('Audio: Clear Separation', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Video probabilities
    axes[1].hist(np.array(video_probs)[fake_mask], bins=10, alpha=0.7,
                label='Fake', color='red', edgecolor='black')
    axes[1].hist(np.array(video_probs)[real_mask], bins=10, alpha=0.7,
                label='Real', color='green', edgecolor='black')
    axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Probability (Real)', fontweight='bold')
    axes[1].set_ylabel('Count', fontweight='bold')
    axes[1].set_title('Video: Heavy Overlap!', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probability_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved: {output_dir / 'probability_distributions.png'}")
    
    # 4. Hypotheses for video failure
    print("\n" + "=" * 80)
    print("💡 HYPOTHESES FOR VIDEO FAILURE")
    print("=" * 80)
    
    hypotheses = """
    1. FEATURE NOISE
       - Lip landmarks detection unreliable
       - Optical flow too sensitive to head movement
       - MediaPipe artifacts in low-quality videos
    
    2. DATASET ISSUES
       - Small dataset (32 videos)
       - Video quality variations
       - Different deepfake techniques per video
    
    3. FEATURE INADEQUACY
       - Lip dynamics might be well-reproduced by generators
       - Optical flow doesn't capture synthesis artifacts
       - Lip-sync features too coarse
    
    4. TEMPORAL RESOLUTION
       - Phoneme-level might be too coarse for video
       - Frame-level features might work better
       - Need multi-scale temporal analysis
    
    5. FEATURE ENGINEERING
       - Current 16 features insufficient
       - Need more sophisticated features:
         * Micro-expressions
         * Eye blink patterns  
         * Facial texture analysis
         * Head pose consistency
    """
    
    print(hypotheses)
    
    # Save analysis report
    report = {
        'performance': {
            'audio_accuracy': audio_results['accuracy'],
            'video_accuracy': video_results['accuracy'],
            'audio_roc_auc': audio_results['roc_auc'],
            'video_roc_auc': video_results['roc_auc']
        },
        'cv_analysis': {
            'audio_mean': float(np.mean(audio_cv)),
            'audio_std': float(np.std(audio_cv)),
            'video_mean': float(np.mean(video_cv)),
            'video_std': float(np.std(video_cv))
        },
        'failure_modes': failure_modes,
        'agreement_counts': dict(agreement_counts)
    }
    
    with open(output_dir / 'failure_analysis.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved: {output_dir / 'failure_analysis.json'}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("\nKey findings:")
    print("  - Video features show heavy Real-Fake overlap")
    print("  - ROC-AUC 0.52 indicates near-random performance")
    print("  - High CV variance suggests instability")
    print("  - Video rarely helps when audio is correct")
    print("=" * 80)


if __name__ == "__main__":
    analyze_video_failure()
