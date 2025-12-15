#!/usr/bin/env python3
"""
Test Video Artifact-Based Deepfake Classifier

Similar to audio classifier, but using video artifact features:
- Lip aperture dynamics (velocity, acceleration, smoothness)
- Optical flow patterns (magnitude, direction consistency)
- Audio-video synchronization (lip-sync quality)

Process:
1. Extract video artifact features from Trump/Biden dataset
2. Aggregate features per video
3. Train Random Forest classifier
4. Evaluate performance vs baseline
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
from video.video_artifact_features import VideoArtifactExtractor


def parse_textgrid(textgrid_path: Path):
    """
    Parse TextGrid file to get phoneme intervals
    
    Returns:
        List of dicts with 'phoneme', 'start', 'end'
    """
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    intervals = []
    lines = content.split('\n')
    
    in_phones_tier = False
    in_interval = False
    current_interval = {}
    
    for line in lines:
        line = line.strip()
        
        if 'name = "phones"' in line:
            in_phones_tier = True
            continue
        
        if in_phones_tier:
            if line.startswith('intervals ['):
                in_interval = True
                current_interval = {}
            elif in_interval:
                if 'xmin' in line:
                    current_interval['start'] = float(line.split('=')[1].strip())
                elif 'xmax' in line:
                    current_interval['end'] = float(line.split('=')[1].strip())
                elif 'text' in line:
                    phoneme = line.split('=')[1].strip().strip('"')
                    if phoneme and phoneme != 'sp' and phoneme != 'sil':
                        current_interval['phoneme'] = phoneme
                        intervals.append(current_interval.copy())
                    in_interval = False
            elif 'item [' in line and 'item [1]' not in line:
                # End of phones tier
                break
    
    return intervals


def extract_features_from_video(video_path: Path, audio_path: Path, 
                                textgrid_path: Path, extractor: VideoArtifactExtractor):
    """
    Extract video artifact features from a video
    
    Returns:
        dict: {phoneme_type: aggregated_features}
    """
    # Parse TextGrid to get phoneme timings
    intervals = parse_textgrid(textgrid_path)
    
    if not intervals:
        print(f"    Warning: No phoneme intervals found")
        return None
    
    # Get video FPS
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    # Extract features per phoneme occurrence
    features_list = []
    
    for interval in intervals:
        phoneme = interval['phoneme']
        start_time = interval['start']
        end_time = interval['end']
        
        # Convert time to frames
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        if end_frame <= start_frame:
            end_frame = start_frame + 1
        
        try:
            features = extractor.process_video_interval(
                video_path=str(video_path),
                audio_path=str(audio_path),
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                phoneme=phoneme
            )
            
            if features:
                features_list.append(features)
        except Exception as e:
            print(f"    Warning: Failed to extract features for {phoneme}: {e}")
            continue
    
    if not features_list:
        return None
    
    # Define expected feature keys (16 features)
    expected_keys = [
        # Lip dynamics (9)
        'lip_aperture_mean', 'lip_aperture_std',
        'lip_velocity_mean', 'lip_velocity_std',
        'lip_acceleration_mean', 'lip_acceleration_std',
        'lip_smoothness', 'lip_range', 'lip_velocity_entropy',
        # Optical flow (4)
        'flow_mag_mean', 'flow_mag_std',
        'flow_dir_consistency', 'flow_spatial_variation',
        # Lip-sync (3)
        'lipsync_correlation', 'lipsync_lag_ms', 'lipsync_quality'
    ]
    
    # Aggregate by phoneme type
    phoneme_features = defaultdict(list)
    
    for item in features_list:
        phoneme = item['phoneme']
        
        # Collect features in consistent order
        feature_vector = []
        for key in expected_keys:
            if key in item:
                feature_vector.append(float(item[key]))
            else:
                feature_vector.append(0.0)
        
        phoneme_features[phoneme].append(feature_vector)
    
    # Average features for each phoneme type
    aggregated = {}
    for phoneme, feat_list in phoneme_features.items():
        aggregated[phoneme] = np.mean(feat_list, axis=0)
    
    return aggregated


def create_feature_matrix(all_video_features):
    """
    Create feature matrix from per-video phoneme features
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
                # If phoneme missing, use zeros (16 features)
                video_feature_vector.extend([0.0] * 16)
        
        X.append(video_feature_vector)
        video_ids.append(video_id)
    
    return np.array(X), video_ids, common_phonemes


def process_dataset(dataset_path: Path, output_dir: Path):
    """
    Process entire Trump/Biden dataset
    
    Returns:
        dict: {video_id: features}
        dict: {video_id: label (0=fake, 1=real)}
    """
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = VideoArtifactExtractor()
    
    all_features = {}
    labels = {}
    
    print("=" * 80)
    print("PROCESSING DATASET - VIDEO ARTIFACT FEATURES EXTRACTION")
    print("=" * 80)
    
    # Process Trump and Biden videos
    for person in ['trump', 'biden']:
        prefix = 't' if person == 'trump' else 'b'
        person_dir = dataset_path.parent / person
        
        print(f"\n{person.upper()} videos:")
        
        # Fake videos (00-07)
        for i in range(0, 8):
            video_id = f"{prefix}-{i:02d}"
            video_path = person_dir / f"{video_id}.mp4"
            audio_path = dataset_path / person / video_id / f"{video_id}.wav"
            textgrid_path = dataset_path / person / video_id / f"{video_id}.TextGrid"
            
            if video_path.exists() and audio_path.exists() and textgrid_path.exists():
                print(f"  Processing {video_id} (FAKE)...", end=' ', flush=True)
                try:
                    features = extract_features_from_video(
                        video_path, audio_path, textgrid_path, extractor
                    )
                    
                    if features:
                        all_features[video_id] = features
                        labels[video_id] = 0  # Fake
                        print(f"✓ {len(features)} phonemes")
                    else:
                        print("✗ Failed")
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        # Real videos (08-15)
        for i in range(8, 16):
            video_id = f"{prefix}-{i:02d}"
            video_path = person_dir / f"{video_id}.mp4"
            audio_path = dataset_path / person / video_id / f"{video_id}.wav"
            textgrid_path = dataset_path / person / video_id / f"{video_id}.TextGrid"
            
            if video_path.exists() and audio_path.exists() and textgrid_path.exists():
                print(f"  Processing {video_id} (REAL)...", end=' ', flush=True)
                try:
                    features = extract_features_from_video(
                        video_path, audio_path, textgrid_path, extractor
                    )
                    
                    if features:
                        all_features[video_id] = features
                        labels[video_id] = 1  # Real
                        print(f"✓ {len(features)} phonemes")
                    else:
                        print("✗ Failed")
                except Exception as e:
                    print(f"✗ Error: {e}")
    
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
    print("TRAINING RANDOM FOREST CLASSIFIER - VIDEO FEATURES")
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
    print("RESULTS - VIDEO ARTIFACT-BASED CLASSIFIER")
    print("=" * 80)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
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
    
    # Save results
    results = {
        'modality': 'video',
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
    
    with open(output_dir / 'video_classifier_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir / 'video_classifier_results.json'}")
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)
    print(f"Audio features:  90.0% accuracy")
    print(f"Video features:  {accuracy:.1%} accuracy")
    
    if accuracy > 0.7:
        print("\n✅ SUCCESS: Video features work for deepfake detection!")
    elif accuracy > 0.6:
        print("\n🟡 MODERATE: Some discriminative power")
    else:
        print("\n❌ NEEDS IMPROVEMENT: Low discriminative power")
    
    print("=" * 80)
    
    return results


def main():
    """Main function"""
    dataset_path = Path("dataset/trump_biden/processed")
    output_dir = Path("test/video_classifier_results")
    
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


if __name__ == "__main__":
    main()
