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


from compare_audio import compute_confidence_and_verdict, compare_embeddings, print_results
from compare_video import compare_video
from create_signature import create_video_signature, create_audio_signature


# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose person to test: "trump" or "biden"
PERSON = "trump"

# Path to reference videos (for training) - REAL videos
REFERENCE_VIDEO_FOLDER = f"../dataset/trump_biden/{PERSON}"

# Path to test videos - FAKE videos (same person)
TEST_VIDEO_FOLDER = f"../dataset/trump_biden/{PERSON}"

# Filter reference videos: Use ONLY REAL videos (IDs 08-15)
# Trump Real: t-08 to t-15
# Biden Real: b-08 to b-15
REFERENCE_FILTER = [f"{PERSON[0]}-{i:02d}" for i in range(9, 15)]

# Filter test videos: FAKE (00-07) + REAL Held-out (08, 15)
FAKE_IDS = [f"{PERSON[0]}-{i:02d}" for i in range(0, 8)]
REAL_HELD_OUT_IDS = [f"{PERSON[0]}-{i:02d}" for i in [8, 15]]
TEST_FILTER = FAKE_IDS + REAL_HELD_OUT_IDS

# Output directory for processed data
PROCESSED_DIR = "../dataset/trump_biden/processed"

# Ridge regression regularization
LAMBDA_REG = 10.0

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
    print(f"Processing {video_id}...")
    
    # Create directory
    processed_paths["dir"].mkdir(parents=True, exist_ok=True)
    
    # Extract audio
    if not processed_paths["audio"].exists():
        if not extract_audio(video_path, processed_paths["audio"]):
            print(f"Audio extraction failed")
            return False
    
    # MFA alignment
    if not processed_paths["textgrid"].exists():
        # Check for existing TextGrid
        found_existing = None
        mfa_aligned_tg = processed_paths["dir"] / "mfa_workspace" / "aligned" / f"{video_id}.TextGrid"
        mfa_aligned_tg_unique = processed_paths["dir"] / f"mfa_workspace_{video_id}" / "aligned" / f"{video_id}.TextGrid"
        
        for existing_tg in [mfa_aligned_tg, mfa_aligned_tg_unique]:
            if existing_tg.exists():
                found_existing = existing_tg
                break
        
        if found_existing:
            import shutil
            shutil.copy(found_existing, processed_paths["textgrid"])
        else:
            tg = run_mfa_alignment(
                video_id,
                processed_paths["audio"],
                transcript_path,
                processed_paths["dir"]
            )
            if not tg and not processed_paths["textgrid"].exists():
                print(f"MFA alignment failed")
                return False
    
    # Extract audio embeddings
    if not processed_paths["audio_emb"].exists():
        if not extract_audio_embeddings(
            processed_paths["audio"],
            processed_paths["textgrid"],
            processed_paths["audio_emb"]
        ):
            print(f"Audio embeddings extraction failed")
            return False
    
    # Extract video embeddings  
    if not processed_paths["video_emb"].exists():
        if not extract_video_embeddings(
            video_path,
            processed_paths["textgrid"],
            processed_paths["video_emb"]
        ):
            print(f"Video embeddings extraction failed")
            return False
    
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
    
    print(f"Loading from {video_folder.name}/")
    print(f"Found {len(video_files)} videos")
    
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
            print(f"{video_id}: Using cached embeddings")
        
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
    
    print(f"Loaded {len(sample_names)} samples with {len(audio_embeddings)} phonemes")
    
    return dict(audio_embeddings), dict(video_embeddings), sample_names


def load_single_video_embeddings(
    video_path: Path,
    processed_base: Path
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load embeddings for a single video.
    
    Returns:
        Tuple of (audio_embeddings, video_embeddings)
    """
    paths = get_processed_paths(video_path, processed_base)
    
    if not check_processed(paths):
        return {}, {}
    
    audio_emb = load_audio_embeddings(str(paths["audio_emb"]))
    video_emb = load_video_embeddings_npz(str(paths["video_emb"]))
    
    return audio_emb, video_emb


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
    Tests each video INDIVIDUALLY and aggregates results.
    
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
        Verification results dictionary with per-video results
    """
    ref_folder = Path(reference_folder)
    test_folder_path = Path(test_folder)
    processed_base = Path(processed_dir)
    transcript_base = Path(transcript_dir)
    
    # 1. Load reference embeddings
    print(f"Loading Reference Data")
    
    ref_audio, ref_video, ref_names = load_embeddings_from_folder(
        ref_folder,
        processed_base,
        transcript_base,
        reference_filter,
        auto_preprocess
    )
    
    if not ref_audio or not ref_video:
        print("Error: No reference data loaded")
        return {}
    
    # 2. Train multimodal space
    print(f"Training Multimodal Space")
    
    # Convert to arrays
    ref_audio_arrays = {k: np.array(v) for k, v in ref_audio.items()}
    ref_video_arrays = {k: np.array(v) for k, v in ref_video.items()}
    
    space = MultimodalCompatibilitySpace(lambda_reg=lambda_reg)
    space.train(ref_audio_arrays, ref_video_arrays, min_samples=1)

    # Test EACH video individually
    print(f"Per-Video Testing")
    
    video_files = get_video_files(test_folder_path, test_filter)
    
    per_video_results = []
    all_compatibility_ratios = []
    all_confidences = []
    
    for video_path in video_files:
        video_id = video_path.stem
        
        # Load embeddings for this single video
        audio_emb, video_emb = load_single_video_embeddings(video_path, processed_base)
        
        if not audio_emb or not video_emb:
            print(f"  ⏭️  {video_id}: Not processed, skipping")
            continue
        
        print(f"\n  📹 Testing {video_id}...")
        
        # Verify this single video (silently)
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        result = space.verify(audio_emb, video_emb)
        
        sys.stdout = old_stdout
        
        # Store result
        per_video_results.append({
            "video_id": video_id,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "compatibility_ratio": result["compatibility_ratio"],
            "compatible_phonemes": result["compatible_phonemes"],
            "total_phonemes": result["total_phonemes"],
            "average_error": result["average_error"]
        })
        
        all_compatibility_ratios.append(result["compatibility_ratio"])
        all_confidences.append(result["confidence"])
        
        # Print summary for this video
        status = "🟢" if result["confidence"] > 50 else "🟡" if result["confidence"] > 25 else "🔴"
        print(f"     {status} {result['verdict']}: {result['confidence']:.1f}% ({result['compatible_phonemes']}/{result['total_phonemes']} phonemes)")
    
    if not per_video_results:
        print("No videos tested")
        return {}
    
    avg_compatibility = np.mean(all_compatibility_ratios)
    avg_confidence = np.mean(all_confidences)
    std_confidence = np.std(all_confidences)
    
    # Determine final verdict based on average
    if avg_compatibility >= 0.7:
        final_verdict = "SAME PERSON"
    elif avg_compatibility >= 0.5:
        final_verdict = "LIKELY SAME PERSON"
    elif avg_compatibility >= 0.3:
        final_verdict = "UNCERTAIN"
    else:
        final_verdict = "DIFFERENT PERSON"
    
    # Count verdicts
    verdict_counts = {}
    for r in per_video_results:
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    
    print(f"FINAL VERDICT: {final_verdict}")
    print(f"Videos tested:       {len(per_video_results)}")
    print(f"Average confidence:  {avg_confidence:.1f}% (±{std_confidence:.1f}%)")
    print(f"Average compatibility: {avg_compatibility*100:.1f}%")
    print()
    print("Verdict breakdown:")
    for v, count in sorted(verdict_counts.items()):
        print(f"  - {v}: {count} videos")
    
    return {
        "final_verdict": final_verdict,
        "average_confidence": float(avg_confidence),
        "std_confidence": float(std_confidence),
        "average_compatibility_ratio": float(avg_compatibility),
        "videos_tested": len(per_video_results),
        "verdict_breakdown": verdict_counts,
        "per_video_results": per_video_results,
        "reference_samples": len(ref_names),
        "reference_folder": str(ref_folder),
        "test_folder": str(test_folder_path)
    }


def main():
    print("MULTIMODAL IDENTITY VERIFICATION DEMO")
    print(f"Reference: {REFERENCE_VIDEO_FOLDER}")
    print(f"Test:      {TEST_VIDEO_FOLDER}")
    print(f"Preprocess: {'Yes' if AUTO_PREPROCESS else 'No'}")
    
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
        
        print(f"Results saved to: {output_file}")

    
    # ==========================================================================
    # AUDIO & VIDEO COMPARISON - Test ALL FAKE videos
    # ==========================================================================
    
    processed_base = project_root / f"dataset/trump_biden/processed/{PERSON}"
    sig_path = Path(__file__).parent / f"signatures/{PERSON}"
    sig_path.mkdir(parents=True, exist_ok=True)
    
    # Create signatures if needed
    reference_audio_sig = sig_path / "audio_signature.npz"
    reference_video_sig = sig_path / "video_signature.npz"
    
    # Create audio signature
    if not reference_audio_sig.exists():
        print("\nCreating audio signature from reference embeddings...")
        all_audio_embeddings = {}
        for ref_id in REFERENCE_FILTER:
            ref_audio_npz = processed_base / ref_id / f"{ref_id}_audio.npz"
            if ref_audio_npz.exists():
                emb = load_audio_embeddings(str(ref_audio_npz))
                for phoneme, vec in emb.items():
                    if phoneme not in all_audio_embeddings:
                        all_audio_embeddings[phoneme] = []
                    if vec.ndim == 2:
                        all_audio_embeddings[phoneme].extend(vec)
                    else:
                        all_audio_embeddings[phoneme].append(vec)
        if all_audio_embeddings:
            aggregated = {p: np.mean(v, axis=0) for p, v in all_audio_embeddings.items()}
            np.savez(reference_audio_sig, **aggregated)
            print(f"Audio signature: {len(aggregated)} phonemes")
    
    # Create video signature
    if not reference_video_sig.exists():
        print("Creating video signature from reference embeddings...")
        all_video_embeddings = {}
        for ref_id in REFERENCE_FILTER:
            ref_video_npz = processed_base / ref_id / f"{ref_id}_video.npz"
            if ref_video_npz.exists():
                emb = load_video_embeddings_npz(str(ref_video_npz))
                for phoneme, vec in emb.items():
                    if phoneme not in all_video_embeddings:
                        all_video_embeddings[phoneme] = []
                    if vec.ndim == 2:
                        all_video_embeddings[phoneme].extend(vec)
                    else:
                        all_video_embeddings[phoneme].append(vec)
        if all_video_embeddings:
            aggregated = {p: np.mean(v, axis=0) for p, v in all_video_embeddings.items()}
            np.savez(reference_video_sig, **aggregated)
            print(f"Video signature: {len(aggregated)} phonemes")
    
    
    # Test IDs
    fake_ids = FAKE_IDS
    real_ids = REAL_HELD_OUT_IDS  # REAL not in signature (signature uses 09-14)
    
    # ==========================================================================
    # AUDIO COMPARISON
    # ==========================================================================
    print(f"\nAUDIO COMPARISON")
    print(f"{'Video':<10} | {'Type':<6} | {'Global Sim':<10} | {'≥0.9':<7} | {'Verdict'}")
    
    audio_results_fake = []
    audio_results_real = []
    
    # Test REAL samples first (not in signature)
    for test_id in real_ids:
        test_audio_npz = processed_base / test_id / f"{test_id}_audio.npz"
        if reference_audio_sig.exists() and test_audio_npz.exists():
            similarities, global_sim, _, _ = compare_embeddings(reference_audio_sig, test_audio_npz)
            metrics = compute_confidence_and_verdict(similarities)
            
            audio_results_real.append({
                "video_id": test_id,
                "global_sim": global_sim,
                "excellent_pct": metrics["excellent_percentage"],
                "verdict": metrics["verdict"]
            })
            
            status = ""
            print(f"{test_id:<10} | {'REAL':<6} | {global_sim:<10.4f} | {status} {metrics['excellent_percentage']:>5.1f}% | {metrics['verdict']}")
        else:
            print(f"{test_id:<10} | {'REAL':<6} | {'N/A':<10} | {'N/A':<7} | Not found")
    
    print("-" * 70)
    
    # Test FAKE samples
    for test_id in fake_ids:
        test_audio_npz = processed_base / test_id / f"{test_id}_audio.npz"
        if reference_audio_sig.exists() and test_audio_npz.exists():
            similarities, global_sim, _, _ = compare_embeddings(reference_audio_sig, test_audio_npz)
            metrics = compute_confidence_and_verdict(similarities)
            
            audio_results_fake.append({
                "video_id": test_id,
                "global_sim": global_sim,
                "excellent_pct": metrics["excellent_percentage"],
                "verdict": metrics["verdict"]
            })
            
            status = ""
            print(f"{test_id:<10} | {'FAKE':<6} | {global_sim:<10.4f} | {status} {metrics['excellent_percentage']:>5.1f}% | {metrics['verdict']}")
        else:
            print(f"{test_id:<10} | {'FAKE':<6} | {'N/A':<10} | {'N/A':<7} | Not found")
    
    # Audio summary
    print("-" * 70)
    if audio_results_real:
        avg_real = np.mean([r["excellent_pct"] for r in audio_results_real])
        print(f"{'AVG REAL':<10} |        |            | {avg_real:>5.1f}% |")
    if audio_results_fake:
        avg_fake = np.mean([r["excellent_pct"] for r in audio_results_fake])
        print(f"{'AVG FAKE':<10} |        |            | {avg_fake:>5.1f}% |")
    
    # ==========================================================================
    # VIDEO COMPARISON
    # ==========================================================================
    print(f"\nVIDEO COMPARISON")
    print(f"{'Video':<10} | {'Type':<6} | {'Global Sim':<10} | {'≥0.9':<7} | {'Verdict'}")
    
    video_results_fake = []
    video_results_real = []
    
    # Test REAL samples first (not in signature)
    for test_id in real_ids:
        test_video_npz = processed_base / test_id / f"{test_id}_video.npz"
        if reference_video_sig.exists() and test_video_npz.exists():
            similarities, global_sim, _, _ = compare_embeddings(reference_video_sig, test_video_npz)
            metrics = compute_confidence_and_verdict(similarities)
            
            video_results_real.append({
                "video_id": test_id,
                "global_sim": global_sim,
                "excellent_pct": metrics["excellent_percentage"],
                "verdict": metrics["verdict"]
            })
            
            status = ""
            print(f"{test_id:<10} | {'REAL':<6} | {global_sim:<10.4f} | {status} {metrics['excellent_percentage']:>5.1f}% | {metrics['verdict']}")
        else:
            print(f"{test_id:<10} | {'REAL':<6} | {'N/A':<10} | {'N/A':<7} | Not found")
    
    print("-" * 70)
    
    # Test FAKE samples
    for test_id in fake_ids:
        test_video_npz = processed_base / test_id / f"{test_id}_video.npz"
        if reference_video_sig.exists() and test_video_npz.exists():
            similarities, global_sim, _, _ = compare_embeddings(reference_video_sig, test_video_npz)
            metrics = compute_confidence_and_verdict(similarities)
            
            video_results_fake.append({
                "video_id": test_id,
                "global_sim": global_sim,
                "excellent_pct": metrics["excellent_percentage"],
                "verdict": metrics["verdict"]
            })
            
            status = ""
            print(f"{test_id:<10} | {'FAKE':<6} | {global_sim:<10.4f} | {status} {metrics['excellent_percentage']:>5.1f}% | {metrics['verdict']}")
        else:
            print(f"{test_id:<10} | {'FAKE':<6} | {'N/A':<10} | {'N/A':<7} | Not found")
    
    # Video summary
    print("-" * 70)
    if video_results_real:
        avg_real = np.mean([r["excellent_pct"] for r in video_results_real])
        print(f"{'AVG REAL':<10} |        |            | {avg_real:>5.1f}% |")
    if video_results_fake:
        avg_fake = np.mean([r["excellent_pct"] for r in video_results_fake])
        print(f"{'AVG FAKE':<10} |        |            | {avg_fake:>5.1f}% |")
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    # FINAL SUMMARY
    print(f"\nSUMMARY: REAL vs FAKE DETECTION")
    print("Legend: ≥70% | 40-70% | 20-40% | <20%")
    print()
    
    if audio_results_real and audio_results_fake:
        real_same_audio = sum(1 for r in audio_results_real if "SAME" in r["verdict"])
        fake_same_audio = sum(1 for r in audio_results_fake if "SAME" in r["verdict"])
        print(f"AUDIO      - REAL: {real_same_audio}/{len(audio_results_real)} as SAME | FAKE: {fake_same_audio}/{len(audio_results_fake)} as SAME")
    
    if video_results_real and video_results_fake:
        real_same_video = sum(1 for r in video_results_real if "SAME" in r["verdict"])
        fake_same_video = sum(1 for r in video_results_fake if "SAME" in r["verdict"])
        print(f"VIDEO      - REAL: {real_same_video}/{len(video_results_real)} as SAME | FAKE: {fake_same_video}/{len(video_results_fake)} as SAME")
        
    # Multimodal results from end_to_end_verification
    if results and "per_video_results" in results:
        mm_results = results["per_video_results"]
        # Separate Real and Fake based on IDs
        mm_real = [r for r in mm_results if r["video_id"] in real_ids]
        mm_fake = [r for r in mm_results if r["video_id"] in fake_ids]
        
        if mm_real or mm_fake:
            if mm_real:
                real_same_mm = sum(1 for r in mm_real if "SAME" in r["verdict"])
                real_str = f"REAL: {real_same_mm}/{len(mm_real)} as SAME"
            else:
                real_str = "REAL: N/A"
                
            if mm_fake:
                fake_same_mm = sum(1 for r in mm_fake if "SAME" in r["verdict"])
                fake_str = f"FAKE: {fake_same_mm}/{len(mm_fake)} as SAME"
            else:
                fake_str = "FAKE: N/A"
                
            print(f"MULTIMODAL - {real_str} | {fake_str}")

    print("\nExpected: REAL should be classified as SAME, FAKE as DIFFERENT")

if __name__ == "__main__":
    main()
