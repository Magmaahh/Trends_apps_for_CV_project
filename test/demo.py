"""
End-to-End Multimodal Identity Verification Demo

This script provides a complete workflow from RAW videos to verification verdict:
1. Auto-preprocessing (if needed): extract audio, MFA alignment, embeddings
2. Training multimodal space on reference videos
3. Testing on test videos
4. Verdict: SAME/DIFFERENT PERSON

Usage:
    Modify the CONFIGURATION section below and run:
    python demo.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import from multimodal_space
from multimodal_space import (
    MultimodalCompatibilitySpace,
    load_audio_embeddings,
    load_video_embeddings_npz
)

# Import preprocessing functions
sys.path.insert(0, str(project_root / "dataset/trump_biden"))
from process_videos import (
    extract_audio,
    run_mfa_alignment,
    extract_audio_embeddings,
    extract_video_embeddings
)

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose person to test: "trump" or "biden"
PERSON = "biden"

# Path to reference videos (for training) - REAL videos
REFERENCE_VIDEO_FOLDER = f"../dataset/trump_biden/{PERSON}"

# Path to test videos - FAKE videos (same person)
TEST_VIDEO_FOLDER = f"../dataset/trump_biden/{PERSON}"

# Filter reference videos: Use ONLY REAL videos (IDs 08-15)
# Trump Real: t-08 to t-15
# Biden Real: b-08 to b-15
REFERENCE_FILTER = [f"{PERSON[0]}-{i:02d}" for i in range(9, 16)]

# Filter test videos: Use ONLY FAKE videos (IDs 00-07)
# Trump Fake: t-00 to t-07
# Biden Fake: b-00 to b-07
#TEST_FILTER = [f"{PERSON[0]}-{i:02d}" for i in range(8, 8)] # REAL
TEST_FILTER = [f"{PERSON[0]}-{i:02d}" for i in range(0, 8)] # FAKE

# Output directory for processed data
PROCESSED_DIR = "../dataset/trump_biden/processed"

# Ridge regression regularization
LAMBDA_REG = 1.0

# Auto-preprocess videos if embeddings not found
AUTO_PREPROCESS = False

# =============================================================================


def get_video_files(folder: Path, filter_list: List[str] = None) -> List[Path]:
    """Get video files from folder, optionally filtered."""
    video_files = sorted(folder.glob("*.mp4"))
    
    if filter_list:
        video_files = [
            vf for vf in video_files 
            if vf.stem in filter_list
        ]
    
    return video_files


def get_processed_paths(video_path: Path, processed_base: Path) -> Dict[str, Path]:
    """Get paths for processed files."""
    video_id = video_path.stem
    person = video_path.parent.name
    
    processed_dir = processed_base / person / video_id
    
    return {
        "dir": processed_dir,
        "audio": processed_dir / f"{video_id}.wav",
        "textgrid": processed_dir / f"{video_id}.TextGrid",
        "audio_emb": processed_dir / f"{video_id}_audio.npz",
        "video_emb": processed_dir / f"{video_id}_video.npz"
    }


def check_processed(paths: Dict[str, Path]) -> bool:
    """Check if video is already processed."""
    return paths["audio_emb"].exists() and paths["video_emb"].exists()


def preprocess_video(video_path: Path, transcript_path: Path, processed_paths: Dict[str, Path]) -> bool:
    """
    Preprocess a single video: extract audio, align, extract embeddings.
    
    Returns:
        True if successful
    """
    video_id = video_path.stem
    print(f"  🎬 Processing {video_id}...")
    
    # Create directory
    processed_paths["dir"].mkdir(parents=True, exist_ok=True)
    
    # 1. Extract audio
    if not processed_paths["audio"].exists():
        print(f"    📢 Extracting audio...")
        if not extract_audio(video_path, processed_paths["audio"]):
            print(f"    ❌ Audio extraction failed")
            return False
        print(f"    ✅ Audio extracted")
    else:
        print(f"    ⏭️  Audio already exists")
    
    # 2. MFA alignment
    # Check for existing TextGrid in various locations
    mfa_aligned_tg = processed_paths["dir"] / "mfa_workspace" / "aligned" / f"{video_id}.TextGrid"
    mfa_aligned_tg_unique = processed_paths["dir"] / f"mfa_workspace_{video_id}" / "aligned" / f"{video_id}.TextGrid"
    
    if not processed_paths["textgrid"].exists():
        # Check if it's in the aligned folder from a previous run
        found_existing = None
        for existing_tg in [mfa_aligned_tg, mfa_aligned_tg_unique]:
            if existing_tg.exists():
                found_existing = existing_tg
                break
        
        if found_existing:
            print(f"    📝 Found existing alignment, copying...")
            import shutil
            shutil.copy(found_existing, processed_paths["textgrid"])
            print(f"    ✅ TextGrid copied")
        else:
            print(f"    📝 MFA alignment...")
            tg = run_mfa_alignment(
                video_id,
                processed_paths["audio"],
                transcript_path,
                processed_paths["dir"]
            )
            if not tg and not processed_paths["textgrid"].exists():
                print(f"    ❌ MFA alignment failed")
                return False
            print(f"    ✅ MFA alignment complete")
    else:
        print(f"    ⏭️  TextGrid already exists")
    
    # 3. Extract audio embeddings
    if not processed_paths["audio_emb"].exists():
        print(f"    🎵 Extracting audio embeddings...")
        if not extract_audio_embeddings(
            processed_paths["audio"],
            processed_paths["textgrid"],
            processed_paths["audio_emb"]
        ):
            print(f"    ❌ Audio embeddings extraction failed")
            return False
    
    # 4. Extract video embeddings  
    if not processed_paths["video_emb"].exists():
        print(f"    📹 Extracting video embeddings...")
        if not extract_video_embeddings(
            video_path,
            processed_paths["textgrid"],
            processed_paths["video_emb"]
        ):
            print(f"    ❌ Video embeddings extraction failed")
            return False
    
    print(f"  ✅ {video_id} processed successfully")
    return True


def load_embeddings_from_folder(
    video_folder: Path,
    processed_base: Path,
    transcript_base: Path,
    filter_list: List[str] = None,
    auto_preprocess: bool = True
) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[np.ndarray]], List[str]]:
    """
    Load embeddings from a folder of videos.
    Auto-preprocesses if embeddings not found.
    
    Returns:
        Tuple of (audio_embeddings, video_embeddings, sample_names)
    """
    video_files = get_video_files(video_folder, filter_list)
    
    print(f"\n📂 Loading from {video_folder.name}/")
    print(f"   Found {len(video_files)} videos")
    
    audio_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    video_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)
    sample_names = []
    
    for video_path in video_files:
        video_id = video_path.stem
        transcript_path = transcript_base / f"{video_id}.txt"
        
        if not transcript_path.exists():
            print(f"  ⚠️  {video_id}: Transcript not found, skipping")
            continue
        
        # Get processed paths
        paths = get_processed_paths(video_path, processed_base)
        
        # Check if already processed
        if not check_processed(paths):
            if not auto_preprocess:
                print(f"  ⏭️  {video_id}: Not processed, skipping")
                continue
            
            # Preprocess
            if not preprocess_video(video_path, transcript_path, paths):
                continue
        else:
            print(f"  ✓ {video_id}: Using cached embeddings")
        
        # Load embeddings
        audio_emb = load_audio_embeddings(str(paths["audio_emb"]))
        video_emb = load_video_embeddings_npz(str(paths["video_emb"]))
        
        # Accumulate by phoneme
        for phoneme in set(audio_emb.keys()) & set(video_emb.keys()):
            audio_vec = audio_emb[phoneme]
            video_vec = video_emb[phoneme]
            
            # Average if multiple
            if audio_vec.ndim == 2:
                audio_vec = np.mean(audio_vec, axis=0)
            if video_vec.ndim == 2:
                video_vec = np.mean(video_vec, axis=0)
            
            audio_embeddings[phoneme].append(audio_vec)
            video_embeddings[phoneme].append(video_vec)
        
        sample_names.append(video_id)
    
    print(f"✓ Loaded {len(sample_names)} samples with {len(audio_embeddings)} phonemes")
    
    return dict(audio_embeddings), dict(video_embeddings), sample_names


def end_to_end_verification(
    reference_folder: str,
    test_folder: str,
    reference_filter: List[str] = None,
    test_filter: List[str] = None,
    processed_dir: str = "../dataset/trump_biden/processed",
    transcript_dir: str = "../dataset/trump_biden/transcripts",
    lambda_reg: float = 1.0,
    auto_preprocess: bool = True
) -> Dict:
    """
    End-to-end multimodal identity verification.
    
    Args:
        reference_folder: Path to reference videos (for training)
        test_folder: Path to test videos
        reference_filter: Filter reference videos (None = all)
        test_filter: Filter test videos (None = all)
        processed_dir: Directory for processed data
        transcript_dir: Directory with transcripts
        lambda_reg: Ridge regression regularization
        auto_preprocess: Auto-preprocess if embeddings not found
        
    Returns:
        Verification results dictionary
    """
    print("=" * 80)
    print("🎬 END-TO-END MULTIMODAL IDENTITY VERIFICATION")
    print("=" * 80)
    
    ref_folder = Path(reference_folder)
    test_folder_path = Path(test_folder)
    processed_base = Path(processed_dir)
    transcript_base = Path(transcript_dir)
    
    # 1. Load reference embeddings
    print("\n" + "=" * 80)
    print("📚 STEP 1: Loading Reference Data")
    print("=" * 80)
    
    ref_audio, ref_video, ref_names = load_embeddings_from_folder(
        ref_folder,
        processed_base,
        transcript_base,
        reference_filter,
        auto_preprocess
    )
    
    if not ref_audio or not ref_video:
        print("❌ Error: No reference data loaded")
        return {}
    
    # 2. Train multimodal space
    print("\n" + "=" * 80)
    print("🎓 STEP 2: Training Multimodal Space")
    print("=" * 80)
    
    # Convert to arrays
    ref_audio_arrays = {k: np.array(v) for k, v in ref_audio.items()}
    ref_video_arrays = {k: np.array(v) for k, v in ref_video.items()}
    
    space = MultimodalCompatibilitySpace(lambda_reg=lambda_reg)
    space.train(ref_audio_arrays, ref_video_arrays, min_samples=1)
    
    # 3. Load test embeddings
    print("\n" + "=" * 80)
    print("🧪 STEP 3: Loading Test Data")
    print("=" * 80)
    
    test_audio, test_video, test_names = load_embeddings_from_folder(
        test_folder_path,
        processed_base,
        transcript_base,
        test_filter,
        auto_preprocess
    )
    
    if not test_audio or not test_video:
        print("❌ Error: No test data loaded")
        return {}
    
    # Convert to arrays
    test_audio_arrays = {k: np.array(v) for k, v in test_audio.items()}
    test_video_arrays = {k: np.array(v) for k, v in test_video.items()}
    
    # 4. Verify
    print("\n" + "=" * 80)
    print("🔍 STEP 4: Verification")
    print("=" * 80)
    
    results = space.verify(test_audio_arrays, test_video_arrays)
    
    # Add metadata
    results["reference_samples"] = len(ref_names)
    results["test_samples"] = len(test_names)
    results["reference_folder"] = str(ref_folder)
    results["test_folder"] = str(test_folder_path)
    
    return results


def main():
    print("=" * 80)
    print("🎯 MULTIMODAL IDENTITY VERIFICATION DEMO")
    print("=" * 80)
    print()
    print(f"Reference: {REFERENCE_VIDEO_FOLDER}")
    print(f"Test:      {TEST_VIDEO_FOLDER}")
    print(f"Preprocess: {'Yes' if AUTO_PREPROCESS else 'No'}")
    print("=" * 80)
    
    # Run verification
    results = end_to_end_verification(
        reference_folder=REFERENCE_VIDEO_FOLDER,
        test_folder=TEST_VIDEO_FOLDER,
        reference_filter=REFERENCE_FILTER,
        test_filter=TEST_FILTER,
        processed_dir=PROCESSED_DIR,
        transcript_dir=Path(REFERENCE_VIDEO_FOLDER).parent / "transcripts",
        lambda_reg=LAMBDA_REG,
        auto_preprocess=AUTO_PREPROCESS
    )
    
    # Save results
    if results:
        output_file = Path(__file__).parent / "demo_results.json"
        with open(output_file, "w") as f:
            # Convert numpy types to native Python types
            def convert_value(v):
                if isinstance(v, (np.ndarray,)):
                    return v.tolist()
                elif isinstance(v, (np.floating, np.float32, np.float64)):
                    return float(v)
                elif isinstance(v, (np.integer, np.int32, np.int64)):
                    return int(v)
                elif isinstance(v, (np.bool_, bool)):
                    return bool(v)
                return v
            
            json_results = {}
            for k, v in results.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    # Handle phoneme_details list of dicts
                    json_results[k] = [
                        {kk: convert_value(vv) for kk, vv in item.items()}
                        for item in v
                    ]
                else:
                    json_results[k] = convert_value(v)
            
            json.dump(json_results, f, indent=2)
        
        print(f"\n📄 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
