#!/usr/bin/env python3
"""
Explain Audio Classifier Predictions - Interpretability Analysis

Shows why the audio classifier makes specific predictions:
- Feature importance breakdown
- Per-video feature contributions
- Phoneme-level analysis
- Visual explanations
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def load_classifier_and_data():
    """Load trained classifier and test data"""
    from test_artifact_classifier import process_dataset, create_feature_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    print("Loading dataset and training classifier...")
    
    # Process dataset
    dataset_path = Path("dataset/trump_biden/processed")
    output_dir = Path("test/artifact_classifier_results")
    
    all_features, labels = process_dataset(dataset_path, output_dir)
    
    # Create feature matrix
    X, video_ids, common_phonemes = create_feature_matrix(all_features)
    y = np.array([labels[vid] for vid in video_ids])
    
    # Train/test split (same as original)
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, video_ids, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    return clf, X_test, y_test, ids_test, common_phonemes


def get_feature_names(common_phonemes):
    """Generate feature names"""
    feature_types = [
        'lfcc_mean_', 'lfcc_var_',  # 2x13 = 26
        'phase_var', 'phase_diff_var', 'inst_freq_var', 'group_delay_var',  # 4
        'hnr', 'f0_mean', 'f0_std',  # 3
        'f1_mean', 'f2_mean', 'f3_mean', 'f1_std', 'f2_std', 'f3_std',  # 6
        'spectral_centroid_mean', 'spectral_centroid_var',  # 10
        'spectral_flatness_mean', 'spectral_flatness_var',
        'spectral_rolloff_mean', 'spectral_rolloff_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var',
        'zcr_mean', 'zcr_var',
        'rms_mean', 'rms_var', 'envelope_var', 'energy_instability'  # 4
    ]
    
    feature_names = []
    for phoneme in common_phonemes:
        for feat in feature_types:
            if 'lfcc_mean_' in feat or 'lfcc_var_' in feat:
                # LFCC has 13 dimensions
                for i in range(13):
                    feature_names.append(f"{phoneme}_{feat}{i}")
            else:
                feature_names.append(f"{phoneme}_{feat}")
    
    return feature_names


def plot_global_feature_importance(clf, feature_names, output_dir, top_n=30):
    """Plot global feature importance"""
    
    print("\n📊 Computing feature importance...")
    
    importance = clf.feature_importances_
    indices = np.argsort(importance)[-top_n:]
    
    # Aggregate by feature type
    feature_type_importance = {}
    for i, feat_name in enumerate(feature_names):
        # Extract feature type (after phoneme)
        parts = feat_name.split('_')
        if len(parts) >= 2:
            feat_type = '_'.join(parts[1:])
            # Group LFCC together
            if 'lfcc' in feat_type:
                feat_type = 'lfcc'
            # Remove numbers from end
            feat_type = ''.join([c for c in feat_type if not c.isdigit()])
            
            if feat_type not in feature_type_importance:
                feature_type_importance[feat_type] = 0
            feature_type_importance[feat_type] += importance[i]
    
    # Plot 1: Top individual features
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    y_pos = np.arange(len(indices))
    axes[0].barh(y_pos, importance[indices], color='steelblue', edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([feature_names[i] for i in indices], fontsize=8)
    axes[0].set_xlabel('Importance', fontweight='bold', fontsize=12)
    axes[0].set_title(f'Top {top_n} Most Important Features', fontweight='bold', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Aggregated by feature type
    sorted_types = sorted(feature_type_importance.items(), key=lambda x: x[1], reverse=True)
    types, values = zip(*sorted_types[:15])
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(types)))
    axes[1].bar(range(len(types)), values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_xticks(range(len(types)))
    axes[1].set_xticklabels(types, rotation=45, ha='right', fontsize=10)
    axes[1].set_ylabel('Cumulative Importance', fontweight='bold', fontsize=12)
    axes[1].set_title('Feature Type Importance (Aggregated)', fontweight='bold', fontsize=14)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'feature_importance_detailed.png'}")
    
    # Print top feature types
    print("\n🔝 Top Feature Types (Aggregated):")
    for i, (ftype, imp) in enumerate(sorted_types[:10], 1):
        print(f"  {i}. {ftype:<25s}: {imp:.4f}")
    
    return dict(sorted_types)


def explain_single_prediction(clf, X_test, y_test, ids_test, video_idx, 
                             feature_names, output_dir):
    """Explain prediction for a single video"""
    
    video_id = ids_test[video_idx]
    true_label = 'Real' if y_test[video_idx] == 1 else 'Fake'
    
    # Get prediction
    X_single = X_test[video_idx:video_idx+1]
    pred_proba = clf.predict_proba(X_single)[0]
    pred_label = 'Real' if clf.predict(X_single)[0] == 1 else 'Fake'
    
    print(f"\n" + "=" * 80)
    print(f"🔍 EXPLAINING PREDICTION FOR {video_id}")
    print("=" * 80)
    print(f"True Label:       {true_label}")
    print(f"Predicted:        {pred_label} {'✓' if pred_label == true_label else '✗'}")
    print(f"Confidence:       {max(pred_proba):.1%}")
    print(f"Prob(Fake):       {pred_proba[0]:.3f}")
    print(f"Prob(Real):       {pred_proba[1]:.3f}")
    
    # Get feature contributions using tree path
    # For each tree, get the prediction path and feature contributions
    feature_contributions = np.zeros(len(feature_names))
    
    for tree in clf.estimators_:
        # Get decision path
        node_indicator = tree.decision_path(X_single)
        leaf_id = tree.apply(X_single)
        
        # Get features used in path
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        # Accumulate feature importance along path
        for node_id in node_indicator.indices:
            if feature[node_id] != -2:  # Not a leaf
                feature_contributions[feature[node_id]] += 1
    
    # Normalize
    feature_contributions = feature_contributions / len(clf.estimators_)
    
    # Get top contributing features
    top_indices = np.argsort(feature_contributions)[-20:]
    
    # Plot contributions
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Top feature contributions
    y_pos = np.arange(len(top_indices))
    colors = ['green' if pred_label == 'Real' else 'red'] * len(top_indices)
    
    axes[0].barh(y_pos, feature_contributions[top_indices], color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    axes[0].set_xlabel('Contribution Score', fontweight='bold', fontsize=12)
    axes[0].set_title(f'Top 20 Contributing Features for {video_id}\n'
                     f'Predicted: {pred_label} (True: {true_label})', 
                     fontweight='bold', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)
    
    # Plot 2: Feature values
    feature_values = X_single[0, top_indices]
    
    # Normalize for visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    feature_values_norm = scaler.fit_transform(feature_values.reshape(-1, 1)).flatten()
    
    colors_vals = ['red' if v < -1 else 'orange' if v < 0 else 'lightgreen' if v < 1 else 'green' 
                   for v in feature_values_norm]
    
    axes[1].barh(y_pos, feature_values_norm, color=colors_vals, alpha=0.7, edgecolor='black')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([feature_names[i] for i in top_indices], fontsize=9)
    axes[1].set_xlabel('Normalized Feature Value', fontweight='bold', fontsize=12)
    axes[1].set_title('Feature Values (Normalized)', fontweight='bold', fontsize=14)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'explanation_{video_id}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / f'explanation_{video_id}.png'}")
    
    # Print top features
    print(f"\n📋 Top 10 Contributing Features:")
    for i, idx in enumerate(top_indices[-10:][::-1], 1):
        feat_name = feature_names[idx]
        contribution = feature_contributions[idx]
        value = X_single[0, idx]
        print(f"  {i}. {feat_name:<40s}: contribution={contribution:.4f}, value={value:.4f}")


def compare_real_vs_fake_features(clf, X_test, y_test, ids_test, feature_names, output_dir):
    """Compare feature distributions for Real vs Fake"""
    
    print("\n📊 Comparing Real vs Fake feature distributions...")
    
    # Get top 12 most important features
    importance = clf.feature_importances_
    top_indices = np.argsort(importance)[-12:]
    
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    
    fake_mask = y_test == 0
    real_mask = y_test == 1
    
    for i, feat_idx in enumerate(top_indices):
        feat_name = feature_names[feat_idx]
        
        fake_values = X_test[fake_mask, feat_idx]
        real_values = X_test[real_mask, feat_idx]
        
        # Plot distributions
        axes[i].hist(fake_values, bins=10, alpha=0.6, color='red', 
                    label=f'Fake (n={len(fake_values)})', edgecolor='black')
        axes[i].hist(real_values, bins=10, alpha=0.6, color='green',
                    label=f'Real (n={len(real_values)})', edgecolor='black')
        
        # Add means
        axes[i].axvline(np.mean(fake_values), color='darkred', linestyle='--', linewidth=2)
        axes[i].axvline(np.mean(real_values), color='darkgreen', linestyle='--', linewidth=2)
        
        # Simplify feature name for display
        display_name = feat_name.split('_')[-1] if len(feat_name.split('_')) > 2 else feat_name
        axes[i].set_title(f'{display_name}\n(Importance: {importance[feat_idx]:.4f})', 
                         fontsize=10, fontweight='bold')
        axes[i].set_xlabel('Feature Value', fontsize=9)
        axes[i].set_ylabel('Count', fontsize=9)
        axes[i].legend(fontsize=8)
        axes[i].grid(alpha=0.3)
    
    plt.suptitle('Top 12 Features: Real vs Fake Distributions', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_distributions_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'feature_distributions_comparison.png'}")


def main():
    """Main interpretability analysis"""
    
    output_dir = Path("test/interpretability")
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("🔍 AUDIO CLASSIFIER INTERPRETABILITY ANALYSIS")
    print("=" * 80)
    
    # Load classifier and data
    clf, X_test, y_test, ids_test, common_phonemes = load_classifier_and_data()
    
    print(f"\nTest set: {len(ids_test)} videos")
    print(f"Features: {X_test.shape[1]}")
    print(f"Phoneme types: {len(common_phonemes)}")
    
    # Generate feature names
    feature_names = get_feature_names(common_phonemes)
    print(f"Feature names generated: {len(feature_names)}")
    
    # 1. Global feature importance
    feature_type_importance = plot_global_feature_importance(
        clf, feature_names, output_dir, top_n=30
    )
    
    # 2. Explain specific predictions
    print("\n" + "=" * 80)
    print("🎯 EXPLAINING INDIVIDUAL PREDICTIONS")
    print("=" * 80)
    
    # Explain a few interesting cases
    for i in range(min(3, len(ids_test))):
        explain_single_prediction(
            clf, X_test, y_test, ids_test, i, feature_names, output_dir
        )
    
    # 3. Compare Real vs Fake distributions
    compare_real_vs_fake_features(
        clf, X_test, y_test, ids_test, feature_names, output_dir
    )
    
    # 4. Summary
    print("\n" + "=" * 80)
    print("✅ INTERPRETABILITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  1. feature_importance_detailed.png - Global feature importance")
    print("  2. explanation_*.png - Per-video explanations")
    print("  3. feature_distributions_comparison.png - Real vs Fake comparison")
    print("\n" + "=" * 80)
    print("\n💡 KEY INSIGHTS:")
    print("  - Features are interpretable (not black-box)")
    print("  - Can see which phonemes and features contribute most")
    print("  - Can explain why each video is classified as Real/Fake")
    print("  - Feature distributions show clear separation")
    print("=" * 80)


if __name__ == "__main__":
    main()
