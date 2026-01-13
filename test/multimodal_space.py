import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# CONFIGURATION - Modify these values as needed
# =============================================================================

# Mode: "dataset" | "test-impostor"
MODE = "dataset"

# Speaker ID for training (e.g., "s1", "s10", "s2")
SPEAKER_ID = "s1"

# Train/test split ratio (0.8 = 80% train, 20% test)
TRAIN_RATIO = 0.8

# Maximum samples to load (None = all, or set a number like 50 for faster testing)
MAX_SAMPLES = None

# Ridge regression regularization parameter (optimized by grid search)
LAMBDA_REG = 10.0

# Threshold settings (optimized by grid search)
THRESHOLD_SIGMA = 2.0  # Number of standard deviations
THRESHOLD_MULTIPLIER = 2.0  # Minimum threshold = mean_error * this multiplier

# Output path for trained model (None = auto-generate based on speaker)
OUTPUT_PATH = None

# Impostor speaker ID for testing (used in "test-impostor" mode)
IMPOSTOR_SPEAKER_ID = "s2"

# Path to trained model (used in "test-impostor" mode)
MODEL_PATH = "signatures/s1/multimodal_model_full.npz"

# Base path to dataset
DATASET_BASE_PATH = "../dataset/output"

# =============================================================================

# Multimodal Compatibility Space Class
class MultimodalCompatibilitySpace:
    """
    Creates a compatibility space between audio and video embeddings
    using per-phoneme ridge regression.
    """
    
    def __init__(self, lambda_reg: float = 1.0):
        """
        Initialize the multimodal space.
        
        Args:
            lambda_reg: Regularization parameter for ridge regression
        """
        self.lambda_reg = lambda_reg
        
        # Per-phoneme linear maps: W_p such that v_p ≈ W_p · a_p
        self.W_maps: Dict[str, np.ndarray] = {}
        
        # Centroids for threshold computation
        self.centroids_audio: Dict[str, np.ndarray] = {}
        self.centroids_video: Dict[str, np.ndarray] = {}
        
        # Dynamic thresholds based on training variance
        self.thresholds: Dict[str, float] = {}
        
        # Training statistics
        self.training_stats: Dict[str, dict] = {}
        
        # Global threshold
        self.global_threshold: float = 0.0
    
    def _ridge_regression(self, A: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Compute ridge regression: W = argmin ||WA - V||² + λ||W||²
        
        Closed form solution: W = V·A^T·(A·A^T + λI)^(-1)
        
        Args:
            A: Audio embeddings matrix (n_samples, audio_dim)
            V: Video embeddings matrix (n_samples, video_dim)
            
        Returns:
            W: Linear map (video_dim, audio_dim)
        """
        # A: (n, audio_dim) -> A^T: (audio_dim, n)
        # V: (n, video_dim) -> V^T: (video_dim, n)
        
        A_T = A.T  # (audio_dim, n)
        V_T = V.T  # (video_dim, n)
        
        # A·A^T: (audio_dim, audio_dim)
        AAT = A_T @ A
        
        # Add regularization: A·A^T + λI
        identity = np.eye(AAT.shape[0])
        AAT_reg = AAT + self.lambda_reg * identity
        
        # (A·A^T + λI)^(-1)
        AAT_inv = np.linalg.inv(AAT_reg)
        
        # W = V·A^T·(A·A^T + λI)^(-1)
        # W: (video_dim, audio_dim)
        W = V_T @ A @ AAT_inv
        
        return W
    
    def train(
        self,
        audio_embeddings: Dict[str, np.ndarray],
        video_embeddings: Dict[str, np.ndarray],
        min_samples: int = 2
    ) -> Dict[str, dict]:
        """
        Train per-phoneme linear maps using POI data.
        
        Args:
            audio_embeddings: Dict mapping phoneme -> array of audio embeddings
            video_embeddings: Dict mapping phoneme -> array of video embeddings
            min_samples: Minimum samples required to train a phoneme map
            
        Returns:
            Dictionary with training statistics
        """
        print(f"Training Multimodal Compatibility Space")
        
        # Find common phonemes
        common_phonemes = set(audio_embeddings.keys()) & set(video_embeddings.keys())
        
        if not common_phonemes:
            print("Error: No common phonemes between audio and video!")
            return {}
        
        print(f"Common phonemes: {len(common_phonemes)}")
        print(f"Regularization λ: {self.lambda_reg}")
        
        trained_phonemes = 0
        skipped_phonemes = 0
        all_distances = []
        
        for phoneme in sorted(common_phonemes):
            audio_emb = audio_embeddings[phoneme]
            video_emb = video_embeddings[phoneme]
            
            # Ensure 2D arrays
            if audio_emb.ndim == 1:
                audio_emb = audio_emb.reshape(1, -1)
            if video_emb.ndim == 1:
                video_emb = video_emb.reshape(1, -1)
            
            n_audio = audio_emb.shape[0]
            n_video = video_emb.shape[0]
            
            # Need matching pairs - use mean if different counts
            if n_audio != n_video:
                # Average to single embedding for training
                audio_emb = np.mean(audio_emb, axis=0, keepdims=True)
                video_emb = np.mean(video_emb, axis=0, keepdims=True)
            
            n_samples = audio_emb.shape[0]
            
            if n_samples < min_samples:
                # For single samples, we can still learn a map
                pass
            
            # Compute centroids
            self.centroids_audio[phoneme] = np.mean(audio_emb, axis=0)
            self.centroids_video[phoneme] = np.mean(video_emb, axis=0)
            
            # Train ridge regression: v ≈ W·a
            try:
                W = self._ridge_regression(audio_emb, video_emb)
                self.W_maps[phoneme] = W
                
                # Compute reconstruction errors for threshold
                predicted_video = audio_emb @ W.T  # (n, video_dim)
                errors = np.linalg.norm(predicted_video - video_emb, axis=1)
                
                mean_error = np.mean(errors)
                std_error = np.std(errors) if len(errors) > 1 else mean_error * 0.5
                
                # Threshold calculation with multiple strategies:
                # 1. Statistical: mean + N*std
                # 2. Multiplicative: mean * multiplier (handles few-sample cases better)
                stat_threshold = mean_error + THRESHOLD_SIGMA * std_error
                mult_threshold = mean_error * THRESHOLD_MULTIPLIER
                
                # Use the larger of the two (more permissive)
                self.thresholds[phoneme] = max(stat_threshold, mult_threshold)
                
                all_distances.extend(errors.tolist())
                
                self.training_stats[phoneme] = {
                    "n_samples": n_samples,
                    "mean_error": float(mean_error),
                    "std_error": float(std_error),
                    "threshold": float(self.thresholds[phoneme])
                }
                
                trained_phonemes += 1
                
            except Exception as e:
                print(f"  {phoneme:8s}: failed ({e})")
                skipped_phonemes += 1
        
        # Compute global threshold
        if all_distances:
            mean_dist = np.mean(all_distances)
            std_dist = np.std(all_distances)
            stat_global = mean_dist + THRESHOLD_SIGMA * std_dist
            mult_global = mean_dist * THRESHOLD_MULTIPLIER
            self.global_threshold = max(stat_global, mult_global)
        
        print(f"Training completed!")
        print(f"  Phonemes trained: {trained_phonemes}")
        print(f"  Phonemes skipped: {skipped_phonemes}")
        print(f"  Global threshold: {self.global_threshold:.4f}")
        
        return self.training_stats
    
    def predict(self, audio_embedding: np.ndarray, phoneme: str) -> Optional[np.ndarray]:
        """
        Predict video embedding from audio embedding for a phoneme.
        
        Args:
            audio_embedding: Audio embedding vector (audio_dim,) or (1, audio_dim)
            phoneme: Phoneme label
            
        Returns:
            Predicted video embedding or None if phoneme not trained
        """
        if phoneme not in self.W_maps:
            return None
        
        W = self.W_maps[phoneme]
        
        if audio_embedding.ndim == 1:
            audio_embedding = audio_embedding.reshape(1, -1)
        
        predicted = audio_embedding @ W.T
        return predicted.squeeze()
    
    def compute_compatibility_score(
        self,
        audio_embedding: np.ndarray,
        video_embedding: np.ndarray,
        phoneme: str
    ) -> Tuple[float, float, bool]:
        """
        Compute compatibility score between audio and video embeddings.
        
        Args:
            audio_embedding: Audio embedding vector
            video_embedding: Video embedding vector  
            phoneme: Phoneme label
            
        Returns:
            Tuple of (prediction_error, threshold, is_compatible)
        """
        predicted_video = self.predict(audio_embedding, phoneme)
        
        if predicted_video is None:
            return float('inf'), 0.0, False
        
        # Compute L2 distance between predicted and actual
        error = float(np.linalg.norm(predicted_video - video_embedding))
        threshold = self.thresholds.get(phoneme, self.global_threshold)
        
        is_compatible = error <= threshold
        
        return error, threshold, is_compatible
    
    def verify(
        self,
        test_audio_embeddings: Dict[str, np.ndarray],
        test_video_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, any]:
        """
        Verify if test audio and video are from the same person as POI.
        
        Args:
            test_audio_embeddings: Audio embeddings from test video
            test_video_embeddings: Video embeddings from test video
            
        Returns:
            Verification results dictionary
        """
        common_phonemes = set(test_audio_embeddings.keys()) & set(test_video_embeddings.keys()) & set(self.W_maps.keys())
        
        if not common_phonemes:
            return {
                "verdict": "INSUFFICIENT DATA",
                "confidence": 0.0,
                "compatible_phonemes": 0,
                "total_phonemes": 0
            }
        
        results = {}
        compatible_count = 0
        total_error = 0.0
        phoneme_details = []

        print(f"Analyzing {len(common_phonemes)} common phonemes:")
        print(f"{'Phoneme':<10} {'Error':>10} {'Threshold':>10} {'Status':<15}")
        
        for phoneme in sorted(common_phonemes):
            audio_emb = test_audio_embeddings[phoneme]
            video_emb = test_video_embeddings[phoneme]
            
            # Average if multiple
            if audio_emb.ndim == 2:
                audio_emb = np.mean(audio_emb, axis=0)
            if video_emb.ndim == 2:
                video_emb = np.mean(video_emb, axis=0)
            
            error, threshold, is_compatible = self.compute_compatibility_score(
                audio_emb, video_emb, phoneme
            )
            
            total_error += error
            
            if is_compatible:
                compatible_count += 1
                status = "COMPATIBLE"
            else:
                status = "MISMATCH"
            
            print(f"{phoneme:<10} {error:>10.4f} {threshold:>10.4f} {status:<15}")
            
            phoneme_details.append({
                "phoneme": phoneme,
                "error": error,
                "threshold": threshold,
                "compatible": is_compatible
            })
        
        # Compute final verdict
        total_phonemes = len(common_phonemes)
        compatibility_ratio = compatible_count / total_phonemes if total_phonemes > 0 else 0
        avg_error = total_error / total_phonemes if total_phonemes > 0 else float('inf')
        
        # Determine verdict based on compatibility ratio
        if compatibility_ratio >= 0.7:
            verdict = "SAME PERSON"
            confidence = min(85 + (compatibility_ratio - 0.7) * 50, 99)
        elif compatibility_ratio >= 0.5:
            verdict = "LIKELY SAME PERSON"
            confidence = 50 + (compatibility_ratio - 0.5) * 70
        elif compatibility_ratio >= 0.3:
            verdict = "UNCERTAIN"
            confidence = 25 + (compatibility_ratio - 0.3) * 50
        else:
            verdict = "DIFFERENT PERSON"
            confidence = compatibility_ratio * 100
        
        print(f"VERDICT: {verdict}")
        print(f"Confidence:          {confidence:.1f}%")
        print(f"Compatible phonemes: {compatible_count}/{total_phonemes} ({compatibility_ratio*100:.1f}%)")
        print(f"Average error:       {avg_error:.4f}")
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "compatible_phonemes": compatible_count,
            "total_phonemes": total_phonemes,
            "compatibility_ratio": compatibility_ratio,
            "average_error": avg_error,
            "phoneme_details": phoneme_details
        }
    
    def save(self, output_path: str):
        """Save the trained model to a .npz file."""
        data = {
            "lambda_reg": np.array([self.lambda_reg]),
            "global_threshold": np.array([self.global_threshold]),
            "phonemes": np.array(list(self.W_maps.keys())),
        }
        
        # Save per-phoneme data
        for phoneme, W in self.W_maps.items():
            data[f"W_{phoneme}"] = W
            data[f"threshold_{phoneme}"] = np.array([self.thresholds.get(phoneme, 0)])
            data[f"centroid_audio_{phoneme}"] = self.centroids_audio.get(phoneme, np.array([]))
            data[f"centroid_video_{phoneme}"] = self.centroids_video.get(phoneme, np.array([]))
        
        np.savez(output_path, **data)
        print(f"Model saved to: {output_path}")
    
    def load(self, model_path: str):
        """Load a trained model from a .npz file."""
        data = np.load(model_path, allow_pickle=True)
        
        self.lambda_reg = float(data["lambda_reg"][0])
        self.global_threshold = float(data["global_threshold"][0])
        
        phonemes = data["phonemes"]
        
        for phoneme in phonemes:
            phoneme = str(phoneme)
            self.W_maps[phoneme] = data[f"W_{phoneme}"]
            self.thresholds[phoneme] = float(data[f"threshold_{phoneme}"][0])
            
            if f"centroid_audio_{phoneme}" in data:
                self.centroids_audio[phoneme] = data[f"centroid_audio_{phoneme}"]
            if f"centroid_video_{phoneme}" in data:
                self.centroids_video[phoneme] = data[f"centroid_video_{phoneme}"]
        
        print(f"Model loaded from: {model_path}")
        print(f"  Phonemes: {len(self.W_maps)}")
        print(f"  Global threshold: {self.global_threshold:.4f}")


def load_audio_embeddings(npz_path: str) -> Dict[str, np.ndarray]:
    """Load audio embeddings from .npz file."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {}
    
    for phoneme in data.files:
        emb = data[phoneme]
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embeddings[phoneme] = emb
    
    return embeddings


def load_video_embeddings(json_path: str) -> Dict[str, np.ndarray]:
    """Load video embeddings from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    
    embeddings = {}
    for phoneme, info in data.items():
        vec = np.array(info["vector"])
        embeddings[phoneme] = vec.reshape(1, -1)
    
    return embeddings


def load_video_embeddings_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """Load video embeddings from .npz file (per-sample format)."""
    data = np.load(npz_path, allow_pickle=True)
    embeddings = {}
    
    for phoneme in data.files:
        emb = data[phoneme]
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        embeddings[phoneme] = emb
    
    return embeddings


class DatasetLoader:
    """
    Load audio and video embeddings from dataset folders for multi-sample training.
    """
    
    def __init__(self, base_path: str = "../dataset/output"):
        self.base_path = Path(base_path)
    
    def get_available_speakers(self) -> List[str]:
        """Get list of available speaker IDs."""
        speakers = set()
        
        # Look for mfa_workspace_sX folders
        for folder in self.base_path.iterdir():
            if folder.is_dir() and folder.name.startswith("mfa_workspace_s"):
                speaker_id = folder.name.replace("mfa_workspace_", "")
                speakers.add(speaker_id)
        
        return sorted(speakers)
    
    def load_speaker_data(
        self,
        speaker_id: str,
        max_samples: Optional[int] = None
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], List[str]]:
        """
        Load all audio and video embeddings for a speaker.
        
        Args:
            speaker_id: Speaker ID (e.g., "s1", "s10")
            max_samples: Maximum number of samples to load (None = all)
            
        Returns:
            Tuple of (audio_embeddings, video_embeddings, sample_names)
            Each embedding dict maps phoneme -> list of vectors
        """
        audio_folder = self.base_path / f"mfa_workspace_{speaker_id}"
        textgrid_folder = self.base_path / f"mfa_output_phonemes_{speaker_id}"
        video_folder = self.base_path / f"video_embeddings_{speaker_id}"
        
        if not audio_folder.exists():
            raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
        if not textgrid_folder.exists():
            raise FileNotFoundError(f"TextGrid folder not found: {textgrid_folder}")
        if not video_folder.exists():
            raise FileNotFoundError(f"Video embeddings folder not found: {video_folder}")
        
        # Find all samples (by video embedding files)
        video_files = sorted(video_folder.glob("*.npz"))
        
        if max_samples:
            video_files = video_files[:max_samples]
        
        print(f"Loading {len(video_files)} samples for speaker {speaker_id}...")
        
        # Accumulate embeddings
        audio_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        video_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
        sample_names = []
        
        # Import audio extractor
        from src.audio.phonemes2emb import PhonemeEmbeddingExtractor
        audio_extractor = PhonemeEmbeddingExtractor()
        
        for video_file in video_files:
            sample_name = video_file.stem
            
            # Check for corresponding audio and textgrid
            audio_file = audio_folder / f"{sample_name}.wav"
            textgrid_file = textgrid_folder / f"{sample_name}.TextGrid"
            
            if not audio_file.exists() or not textgrid_file.exists():
                continue
            
            # Load video embeddings
            video_emb = load_video_embeddings_npz(str(video_file))
            
            # Extract audio embeddings
            audio_results = audio_extractor.process_file(str(audio_file), str(textgrid_file))
            
            if not audio_results:
                continue
            
            # Organize audio embeddings by phoneme
            audio_emb: Dict[str, List[np.ndarray]] = defaultdict(list)
            for item in audio_results:
                audio_emb[item["phoneme"]].append(item["vector"])
            
            # Find common phonemes for this sample
            common = set(audio_emb.keys()) & set(video_emb.keys())
            
            if not common:
                continue
            
            # Accumulate
            for phoneme in common:
                # Average audio if multiple in same sample
                audio_vec = np.mean(audio_emb[phoneme], axis=0)
                audio_embeddings[phoneme].append(audio_vec)
                
                # Average video if multiple in same sample
                video_vec = video_emb[phoneme]
                if video_vec.ndim == 2:
                    video_vec = np.mean(video_vec, axis=0)
                video_embeddings[phoneme].append(video_vec)
            
            sample_names.append(sample_name)
            
            if len(sample_names) % 20 == 0:
                print(f"  Loaded {len(sample_names)} samples...")
        
        print(f"Loaded {len(sample_names)} samples with {len(audio_embeddings)} phonemes")
        
        return dict(audio_embeddings), dict(video_embeddings), sample_names
    
    def train_test_split(
        self,
        audio_embeddings: Dict[str, List[np.ndarray]],
        video_embeddings: Dict[str, List[np.ndarray]],
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray],
        Dict[str, np.ndarray]
    ]:
        """
        Split embeddings into train and test sets.
        
        Args:
            audio_embeddings: Dict mapping phoneme -> list of audio vectors
            video_embeddings: Dict mapping phoneme -> list of video vectors
            train_ratio: Fraction of data for training
            seed: Random seed
            
        Returns:
            Tuple of (train_audio, train_video, test_audio, test_video)
        """
        np.random.seed(seed)
        
        train_audio: Dict[str, List[np.ndarray]] = defaultdict(list)
        train_video: Dict[str, List[np.ndarray]] = defaultdict(list)
        test_audio: Dict[str, List[np.ndarray]] = defaultdict(list)
        test_video: Dict[str, List[np.ndarray]] = defaultdict(list)
        
        for phoneme in audio_embeddings:
            audio_list = audio_embeddings[phoneme]
            video_list = video_embeddings[phoneme]
            
            n_samples = len(audio_list)
            n_train = max(1, int(n_samples * train_ratio))
            
            # Shuffle indices
            indices = np.random.permutation(n_samples)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            for i in train_idx:
                train_audio[phoneme].append(audio_list[i])
                train_video[phoneme].append(video_list[i])
            
            for i in test_idx:
                test_audio[phoneme].append(audio_list[i])
                test_video[phoneme].append(video_list[i])
        
        # Convert to numpy arrays
        def to_arrays(d: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
            return {k: np.array(v) for k, v in d.items() if v}
        
        return (
            to_arrays(train_audio),
            to_arrays(train_video),
            to_arrays(test_audio),
            to_arrays(test_video)
        )


def train_from_dataset(
    speaker_id: str,
    base_path: str = "../dataset/output",
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
    lambda_reg: float = 1.0,
    output_path: Optional[str] = None
) -> Tuple[MultimodalCompatibilitySpace, Dict, Dict]:
    """
    Train a multimodal compatibility space from dataset folders.
    
    Args:
        speaker_id: Speaker ID (e.g., "s1", "s10")
        base_path: Path to dataset/output folder
        train_ratio: Fraction for training (rest for testing)
        max_samples: Maximum samples to use (None = all)
        lambda_reg: Ridge regression regularization
        output_path: Where to save the model
        
    Returns:
        Tuple of (trained_space, train_stats, test_results)
    """
    print(f"TRAINING FROM DATASET - Speaker {speaker_id}")
    
    # Load data
    loader = DatasetLoader(base_path)
    audio_emb, video_emb, sample_names = loader.load_speaker_data(speaker_id, max_samples)
    
    print(f"\nTotal samples: {len(sample_names)}")
    print(f"Phonemes: {len(audio_emb)}")
    
    # Get sample counts per phoneme
    sample_counts = {p: len(audio_emb[p]) for p in audio_emb}
    print(f"Samples per phoneme: min={min(sample_counts.values())}, max={max(sample_counts.values())}")
    
    # Split train/test
    train_audio, train_video, test_audio, test_video = loader.train_test_split(
        audio_emb, video_emb, train_ratio=train_ratio
    )
    
    print(f"\nTrain phonemes: {len(train_audio)}")
    print(f"Test phonemes: {len(test_audio)}")
    
    # Train
    space = MultimodalCompatibilitySpace(lambda_reg=lambda_reg)
    train_stats = space.train(train_audio, train_video, min_samples=1)
    
    # Test on held-out data
    print("VALIDATION ON HELD-OUT SAMPLES (Same Person)")
    
    if test_audio and test_video:
        test_results = space.verify(test_audio, test_video)
    else:
        print("No test data available (all data used for training)")
        test_results = {}
    
    # Save model
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        space.save(output_path)
    
    return space, train_stats, test_results


def test_different_speaker(
    model_path: str,
    speaker_id: str,
    base_path: str = "../dataset/output",
    max_samples: int = 20
) -> Dict:
    """
    Test the model on a DIFFERENT speaker (should fail verification).
    
    Args:
        model_path: Path to trained model
        speaker_id: Speaker ID to test against
        base_path: Path to dataset
        max_samples: Number of samples to test
        
    Returns:
        Verification results
    """
    print(f"TESTING ON DIFFERENT SPEAKER: {speaker_id}")
    
    # Load model
    space = MultimodalCompatibilitySpace()
    space.load(model_path)
    
    # Load different speaker's data
    loader = DatasetLoader(base_path)
    audio_emb, video_emb, sample_names = loader.load_speaker_data(speaker_id, max_samples)
    
    # Convert to numpy arrays
    audio_arrays = {k: np.array(v) for k, v in audio_emb.items()}
    video_arrays = {k: np.array(v) for k, v in video_emb.items()}
    
    # Verify (should fail)
    results = space.verify(audio_arrays, video_arrays)
    
    return results


def main():
    print("MULTIMODAL COMPATIBILITY SPACE")
    print(f"Mode:       {MODE}")
    print(f"Speaker:    {SPEAKER_ID}")
    print(f"Lambda:     {LAMBDA_REG}")
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Max samples: {MAX_SAMPLES or 'All'}")
    print("=" * 80)
    print()
    
    if MODE == "dataset":
        # Train from dataset with multi-sample support
        output = OUTPUT_PATH or f"signatures/{SPEAKER_ID}/multimodal_model_full.npz"
        
        space, train_stats, test_results = train_from_dataset(
            speaker_id=SPEAKER_ID,
            base_path=DATASET_BASE_PATH,
            train_ratio=TRAIN_RATIO,
            max_samples=MAX_SAMPLES,
            lambda_reg=LAMBDA_REG,
            output_path=output
        )
        
        print(f"\n✓ Model saved to: {output}")
        
        # Optionally test on impostor
        if IMPOSTOR_SPEAKER_ID:
            print("\n")
            impostor_results = test_different_speaker(
                model_path=output,
                speaker_id=IMPOSTOR_SPEAKER_ID,
                max_samples=20
            )
    
    elif MODE == "test-impostor":
        results = test_different_speaker(
            model_path=MODEL_PATH,
            speaker_id=IMPOSTOR_SPEAKER_ID,
            max_samples=MAX_SAMPLES or 20
        )


if __name__ == "__main__":
    main()
