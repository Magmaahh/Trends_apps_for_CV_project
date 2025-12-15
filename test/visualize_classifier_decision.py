#!/usr/bin/env python3
"""
Visualize Random Forest Decision Boundaries

Create additional visualizations to understand how the classifier
separates real from fake videos in the feature space.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json


def load_results():
    """Load results from previous classification"""
    results_path = Path("test/artifact_classifier_results/artifact_classifier_results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def recreate_data_and_model():
    """Recreate the dataset and trained model"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
    from audio.phonemes2artifact_features import PhonemeArtifactExtractor
    from test_artifact_classifier import process_dataset, create_feature_matrix
    
    print("Recreating dataset...")
    dataset_path = Path("dataset/trump_biden/processed")
    output_dir = Path("test/artifact_classifier_results")
    
    # Process dataset
    all_features, labels = process_dataset(dataset_path, output_dir)
    
    # Create feature matrix
    X, video_ids, common_phonemes = create_feature_matrix(all_features)
    y = np.array([labels[vid] for vid in video_ids])
    
    # Train/test split
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, video_ids, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    return X, y, video_ids, clf, X_train, X_test, y_train, y_test, ids_train, ids_test


def plot_2d_projection(X, y, video_ids, method='PCA', output_path=None):
    """
    Plot 2D projection of feature space
    
    Args:
        method: 'PCA' or 'TSNE'
    """
    print(f"\nCreating {method} 2D projection...")
    
    if method == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        variance = reducer.explained_variance_ratio_
        title = f'PCA Projection (Variance: {variance[0]:.1%} + {variance[1]:.1%} = {variance.sum():.1%})'
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_2d = reducer.fit_transform(X)
        title = 't-SNE Projection'
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    colors = {0: 'red', 1: 'green'}
    labels_text = {0: 'Fake', 1: 'Real'}
    
    for label in [0, 1]:
        mask = y == label
        ax.scatter(
            X_2d[mask, 0], X_2d[mask, 1],
            c=colors[label], label=labels_text[label],
            alpha=0.7, s=200, edgecolors='black', linewidth=2
        )
        
        # Add video IDs as annotations
        for i, vid in enumerate(np.array(video_ids)[mask]):
            ax.annotate(
                vid, 
                (X_2d[mask, 0][i], X_2d[mask, 1][i]),
                fontsize=9, ha='center', va='bottom'
            )
    
    ax.set_xlabel(f'{method} Component 1', fontsize=14)
    ax.set_ylabel(f'{method} Component 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=14, markerscale=1.5)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
    
    return X_2d


def plot_decision_confidence(clf, X_test, y_test, ids_test, output_path=None):
    """
    Plot prediction confidence for test samples
    """
    print("\nCreating confidence plot...")
    
    # Get prediction probabilities
    y_pred_proba = clf.predict_proba(X_test)
    y_pred = clf.predict(X_test)
    
    # Calculate confidence (max probability)
    confidence = np.max(y_pred_proba, axis=1)
    
    # Sort by confidence
    sort_idx = np.argsort(confidence)[::-1]
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    x_pos = np.arange(len(ids_test))
    colors = []
    
    for i in sort_idx:
        if y_pred[i] == y_test[i]:
            # Correct prediction
            colors.append('green' if y_test[i] == 1 else 'lightgreen')
        else:
            # Wrong prediction
            colors.append('red')
    
    bars = ax.bar(x_pos, confidence[sort_idx], color=colors, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Video Sample', fontsize=12)
    ax.set_ylabel('Prediction Confidence', fontsize=12)
    ax.set_title('Classifier Confidence per Video (Sorted)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ids_test[i] for i in sort_idx], rotation=45, ha='right')
    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Correct (Real)'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Correct (Fake)'),
        Patch(facecolor='red', edgecolor='black', label='Wrong')
    ]
    ax.legend(handles=legend_elements, fontsize=10, loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")


def plot_feature_space_3d(X, y, video_ids, output_path=None):
    """
    Plot 3D PCA projection
    """
    print("\nCreating 3D PCA projection...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    pca = PCA(n_components=3, random_state=42)
    X_3d = pca.fit_transform(X)
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {0: 'red', 1: 'green'}
    labels_text = {0: 'Fake', 1: 'Real'}
    
    for label in [0, 1]:
        mask = y == label
        ax.scatter(
            X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
            c=colors[label], label=labels_text[label],
            alpha=0.7, s=200, edgecolors='black', linewidth=2
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})', fontsize=12)
    ax.set_title(f'3D PCA Projection (Total Variance: {pca.explained_variance_ratio_.sum():.1%})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12, markerscale=1.5)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")


def main():
    """Main function"""
    output_dir = Path("test/artifact_classifier_results")
    
    print("=" * 80)
    print("VISUALIZING CLASSIFIER DECISION BOUNDARIES")
    print("=" * 80)
    
    # Recreate data and model
    X, y, video_ids, clf, X_train, X_test, y_train, y_test, ids_train, ids_test = recreate_data_and_model()
    
    print(f"\nDataset: {len(X)} videos")
    print(f"  - Fake: {sum(y==0)}")
    print(f"  - Real: {sum(y==1)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # 1. PCA 2D projection
    plot_2d_projection(
        X, y, video_ids, 
        method='PCA',
        output_path=output_dir / 'classifier_pca_2d.png'
    )
    
    # 2. t-SNE 2D projection
    plot_2d_projection(
        X, y, video_ids, 
        method='TSNE',
        output_path=output_dir / 'classifier_tsne_2d.png'
    )
    
    # 3. Confidence plot
    plot_decision_confidence(
        clf, X_test, y_test, ids_test,
        output_path=output_dir / 'classifier_confidence.png'
    )
    
    # 4. 3D PCA projection
    plot_feature_space_3d(
        X, y, video_ids,
        output_path=output_dir / 'classifier_pca_3d.png'
    )
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"Plots saved to: {output_dir}")
    print("\nGenerated visualizations:")
    print("  1. classifier_pca_2d.png    - 2D PCA projection showing separation")
    print("  2. classifier_tsne_2d.png   - 2D t-SNE projection")
    print("  3. classifier_confidence.png - Prediction confidence per video")
    print("  4. classifier_pca_3d.png    - 3D PCA projection")
    print("=" * 80)


if __name__ == "__main__":
    main()
