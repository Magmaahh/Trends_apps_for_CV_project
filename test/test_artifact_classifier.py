#!/usr/bin/env python3
"""
Test Artifact-Based Deepfake Classifier

This script tests whether artifact features can discriminate between real and fake videos,
unlike the identity-oriented Wav2Vec2 embeddings which showed 82-84% overlap.

Process:
1. Extract artifact features from Trump/Biden dataset
2. Aggregate features per video
3. Train Random Forest classifier
4. Evaluate performance (accuracy, precision, recall, F1)
5. Compare with baseline (Wav2Vec2 similarity)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from audio.phonemes2artifact_features import PhonemeArtifactExtractor


def extract_features_from_video(audio_path, textgrid_path, extractor):
    """
    Extract artifact features from a video
    
    Returns:
        dict: {phoneme_type: aggregated_features}
    """
    # Extract per-phoneme features
    features_list = extractor.process_file(audio_path, textgrid_path)
    
    if not features_list:
        return None
    
    # Define expected feature keys (exclude MFCC due to failures)
    expected_keys = [
        'lfcc_mean', 'lfcc_var',  # LFCC features (2)
        'phase_var', 'phase_diff_var', 'inst_freq_var', 'group_delay_var',  # Phase (4)
        'hnr', 'f0_mean', 'f0_std',  # Harmonic (3)
        'f1_mean', 'f2_mean', 'f3_mean', 'f1_std', 'f2_std', 'f3_std',  # Formant (6)
        'spectral_centroid_mean', 'spectral_centroid_var',  # Spectral (10)
        'spectral_flatness_mean', 'spectral_flatness_var',
        'spectral_rolloff_mean', 'spectral_rolloff_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'zcr_mean', 'zcr_var',
        'rms_mean', 'rms_var', 'envelope_var', 'energy_instability'  # Energy (4)
    ]
    
    # Aggregate by phoneme type (average across occurrences)
    phoneme_features = defaultdict(list)
    
    for item in features_list:
        phoneme = item['phoneme']
        
        # Collect features in consistent order
        feature_vector = []
        for key in expected_keys:
            if key in item:
                value = item[key]
                if isinstance(value, np.ndarray):
                    feature_vector.extend(value.tolist())
                else:
                    feature_vector.append(float(value))
            else:
                # Missing feature - pad with zero
                if key in ['lfcc_mean', 'lfcc_var']:
                    feature_vector.extend([0.0] * 13)  # LFCC is 13-D
                else:
                    feature_vector.append(0.0)  # Scalar feature
        
        phoneme_features[phoneme].append(feature_vector)
    
    # Average features for each phoneme type
    aggregated = {}
    for phoneme, feat_list in phoneme_features.items():
        # Now all vectors have same dimension
        aggregated[phoneme] = np.mean(feat_list, axis=0)
    
    return aggregated


def create_feature_matrix(all_video_features):
    """
    Create a feature matrix from per-video phoneme features
    
    Strategy: Use common phonemes across all videos
    """
    # Find common phonemes
    all_phonemes = set()
    for video_feats in all_video_features.values():
        if video_feats:
            all_phonemes.update(video_feats.keys())
    
    common_phonemes = sorted(all_phonemes)
    
    print(f"Total unique phonemes: {len(common_phonemes)}")
    print(f"Phonemes: {common_phonemes[:10]}...")
    
    # Create feature matrix
    X = []
    video_ids = []
    
    for video_id, phoneme_feats in all_video_features.items():
        if phoneme_feats is None:
            continue
        
        # Concatenate features from all common phonemes
        video_feature_vector = []
        
        for phoneme in common_phonemes:
            if phoneme in phoneme_feats:
                video_feature_vector.extend(phoneme_feats[phoneme])
            else:
                # If phoneme missing, use zeros
                feat_dim = len(list(phoneme_feats.values())[0])
                video_feature_vector.extend([0.0] * feat_dim)
        
        X.append(video_feature_vector)
        video_ids.append(video_id)
    
    return np.array(X), video_ids, common_phonemes


def process_dataset(dataset_path, output_dir):
    """
    Process entire Trump/Biden dataset
    
    Returns:
        dict: {video_id: features}
        dict: {video_id: label (0=fake, 1=real)}
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = PhonemeArtifactExtractor()
    
    all_features = {}
    labels = {}
    
    print("=" * 80)
    print("PROCESSING DATASET - ARTIFACT FEATURES EXTRACTION")
    print("=" * 80)
    
    # Process Trump videos
    for person in ['trump', 'biden']:
        prefix = 't' if person == 'trump' else 'b'
        
        print(f"\n{person.upper()} videos:")
        
        # Fake videos (00-07)
        for i in range(0, 8):
            video_id = f"{prefix}-{i:02d}"
            audio_path = dataset_path / person / video_id / f"{video_id}.wav"
            textgrid_path = dataset_path / person / video_id / f"{video_id}.TextGrid"
            
            if audio_path.exists() and textgrid_path.exists():
                print(f"  Processing {video_id} (FAKE)...", end=' ')
                features = extract_features_from_video(
                    str(audio_path), str(textgrid_path), extractor
                )
                
                if features:
                    all_features[video_id] = features
                    labels[video_id] = 0  # Fake
                    print(f"✓ {len(features)} phonemes")
                else:
                    print("✗ Failed")
        
        # Real videos (08-15)
        for i in range(8, 16):
            video_id = f"{prefix}-{i:02d}"
            audio_path = dataset_path / person / video_id / f"{video_id}.wav"
            textgrid_path = dataset_path / person / video_id / f"{video_id}.TextGrid"
            
            if audio_path.exists() and textgrid_path.exists():
                print(f"  Processing {video_id} (REAL)...", end=' ')
                features = extract_features_from_video(
                    str(audio_path), str(textgrid_path), extractor
                )
                
                if features:
                    all_features[video_id] = features
                    labels[video_id] = 1  # Real
                    print(f"✓ {len(features)} phonemes")
                else:
                    print("✗ Failed")
    
    print(f"\n✓ Processed {len(all_features)} videos total")
    print(f"  - Fake: {sum(1 for l in labels.values() if l == 0)}")
    print(f"  - Real: {sum(1 for l in labels.values() if l == 1)}")
    
    return all_features, labels


def train_and_evaluate(X, y, video_ids, output_dir):
    """
    Train Random Forest classifier and evaluate
    """
    output_dir = Path(output_dir)
    
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 80)
    
    # Train/test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, video_ids, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Train set: {len(X_train)} videos")
    print(f"Test set: {len(X_test)} videos")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "=" * 80)
    print("RESULTS - ARTIFACT-BASED CLASSIFIER")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f} (fake detection rate)")
    print(f"Recall:    {recall:.3f} (real detection rate)")
    print(f"F1 Score:  {f1:.3f}")
    print(f"ROC-AUC:   {roc_auc:.3f}")
    
    # Cross-validation
    print("\nCross-validation (5-fold):")
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print(f"  Mean accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("                Fake  Real")
    print(f"Actual Fake     {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Real     {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Plot confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion matrix heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xticklabels(['Fake', 'Real'])
    axes[0].set_yticklabels(['Fake', 'Real'])
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'artifact_classifier_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plots saved to: {output_dir / 'artifact_classifier_results.png'}")
    
    # Feature importance
    feature_importance = clf.feature_importances_
    top_20_idx = np.argsort(feature_importance)[-20:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(20), feature_importance[top_20_idx])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Index')
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature importance saved to: {output_dir / 'feature_importance.png'}")
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'cv_scores': cv_scores.tolist(),
        'confusion_matrix': cm.tolist(),
        'test_videos': {
            'ids': ids_test,
            'true_labels': y_test.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
    }
    
    with open(output_dir / 'artifact_classifier_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to: {output_dir / 'artifact_classifier_results.json'}")
    
    return results


def main():
    """Main function"""
    dataset_path = Path("dataset/trump_biden/processed")
    output_dir = Path("test/artifact_classifier_results")
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Process dataset
    all_features, labels = process_dataset(dataset_path, output_dir)
    
    if not all_features:
        print("Error: No features extracted")
        return
    
    # Create feature matrix
    print("\n" + "=" * 80)
    print("CREATING FEATURE MATRIX")
    print("=" * 80)
    
    X, video_ids, common_phonemes = create_feature_matrix(all_features)
    y = np.array([labels[vid] for vid in video_ids])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {len(y)} (Fake: {sum(y==0)}, Real: {sum(y==1)})")
    
    # Train and evaluate
    results = train_and_evaluate(X, y, video_ids, output_dir)
    
    # Comparison with baseline
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINE (Wav2Vec2)")
    print("=" * 80)
    print("Baseline (Wav2Vec2 similarity):")
    print("  - Real-Fake overlap: 0.82-0.84")
    print("  - Expected accuracy: ~50% (random guess)")
    print()
    print(f"Artifact-Based Classifier:")
    print(f"  - Accuracy: {results['accuracy']:.1%}")
    print(f"  - ROC-AUC: {results['roc_auc']:.3f}")
    print()
    
    if results['accuracy'] > 0.7:
        print("✅ SUCCESS: Artifact features significantly outperform baseline!")
        print("   → Features capture synthesis artifacts, not just identity")
    elif results['accuracy'] > 0.6:
        print("🟡 MODERATE: Some improvement over baseline")
        print("   → Features partially capture artifacts")
    else:
        print("❌ FAILURE: No improvement over baseline")
        print("   → Features may not be discriminative enough")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
