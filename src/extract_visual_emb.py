"""
Visual Embedding Extraction Script for Gold Standard Profile Creation

Creates a "gold standard" identity profile from a collection of authentic videos of a person.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from video.pipeline import VideoPipeline
from video.utils import aggregate_embeddings


def run_mfa_batch(wav_dir: Path, mfa_dict: Path, mfa_acoustic: Path, output_dir: Path) -> Path:
    """
    Run Montreal Forced Aligner in batch mode on multiple audio files.
    
    This function processes all wav+lab file pairs in a directory at once,
    which is more efficient than processing them one by one.
    
    Args:
        wav_dir: Directory containing .wav and .lab files
        mfa_dict: Path to MFA pronunciation dictionary
        mfa_acoustic: Path to MFA acoustic model
        output_dir: Directory where TextGrid files will be saved
        
    Returns:
        Path to output directory containing generated TextGrid files
        
    Note: MFA expects each audio file to have a corresponding .lab file
    with the same name containing the transcript.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["mfa", "align", str(wav_dir), str(mfa_dict), str(mfa_acoustic), str(output_dir)]
    subprocess.run(cmd, check=True)
    return output_dir


def prepare_lab_files(raw_dir: Path, tmp_dir: Path) -> Path:
    """
    Prepare .lab transcript files from GRID dataset .align files.
    
    The GRID audiovisual corpus uses .align files for word-level alignments.
    MFA requires .lab files (plain text transcripts). This function converts
    .align files to .lab format and copies corresponding audio files.
    
    Process:
    1. Find all .align files in the raw directory
    2. For each .align file:
       - Extract transcript text (uppercase)
       - Create corresponding .lab file
       - Copy associated .wav file to temp directory
    
    Args:
        raw_dir: Directory containing raw GRID dataset files (.align, .wav, .mpg)
        tmp_dir: Temporary directory for processed files
        
    Returns:
        Path to directory containing prepared .wav and .lab files
        
    Raises:
        FileNotFoundError: If no .align files are found
    """
    lab_dir = tmp_dir
    
    # Find all .align files (word-level alignments from GRID dataset)
    align_files = list(raw_dir.glob("*.align")) + list(raw_dir.rglob("*.align"))
    if not align_files:
        raise FileNotFoundError(f"No .align files found in {raw_dir}")
    
    # Process each align file
    for align_file in align_files:
        stem = align_file.stem
        wav_src = raw_dir / f"{stem}.wav"
        
        # Skip if corresponding audio file doesn't exist
        if not wav_src.exists():
            continue
        
        # Convert .align to .lab (plain text transcript)
        lab_text = align_file.read_text(encoding="utf-8", errors="ignore").strip().upper()
        (lab_dir / f"{stem}.lab").write_text(lab_text, encoding="utf-8")
        
        # Copy audio file to temp directory
        shutil.copy2(wav_src, lab_dir / wav_src.name)
    
    return lab_dir


def extract_visual_emb(
    raw_dir: Path, mfa_dict: Path, mfa_acoustic: Path, output_json: Path, device: str = "auto"
) -> None:
    """
    Args:
        raw_dir: Directory containing raw video files (.mpg/.mp4), audio (.wav),
                and alignment files (.align) from GRID dataset
        mfa_dict: Path to MFA pronunciation dictionary
        mfa_acoustic: Path to MFA acoustic model
        output_json: Path where the gold standard JSON will be saved
        device: Computing device for neural network ("auto", "cuda", or "cpu")
        
    Output JSON Format:
        {
            "IH1": [128-D normalized vector],
            "AE1": [128-D normalized vector],
            ...
        }
        
    Example:
        extract_visual_emb(
            Path("dataset/raw/speaker1"),
            Path("models/mfa_dict.txt"),
            Path("models/mfa_acoustic.zip"),
            Path("output/speaker1_profile.json")
        )
    """
    # Use temporary directory for intermediate files
    with tempfile.TemporaryDirectory(prefix="forge_") as tmp:
        tmp_dir = Path(tmp)
        
        # Prepare .lab transcript files from GRID .align files
        wav_lab_dir = prepare_lab_files(raw_dir, tmp_dir)
        
        # Run MFA batch alignment to get TextGrid files with phoneme timestamps
        tg_dir = tmp_dir / "aligned"
        run_mfa_batch(wav_lab_dir, mfa_dict, mfa_acoustic, tg_dir)

        # Initialize video processing pipeline
        pipeline = VideoPipeline(mfa_dict, mfa_acoustic, device=device)
        embedding_files: List[Path] = []

        # Process each video to extract phoneme embeddings
        tg_files = list(tg_dir.glob("*.TextGrid"))
        for tg_file in tqdm(tg_files, desc="Processing videos"):
            stem = tg_file.stem
            
            # Find corresponding video file (try .mpg first, then .mp4)
            video_path = raw_dir / f"{stem}.mpg"
            if not video_path.exists():
                video_path = raw_dir / f"{stem}.mp4"
            if not video_path.exists():
                continue
            
            # Extract embeddings for this video
            # Returns: {"IH1": [emb1, emb2], "AE1": [emb3], ...}
            embeddings = pipeline.process_single_video(str(video_path))
            
            if embeddings:
                # Save embeddings to temporary .npz file
                # Stack multiple occurrences of same phoneme into 2D array
                npz_path = tmp_dir / f"{stem}.npz"
                np.savez_compressed(npz_path, **{k: np.stack(v) for k, v in embeddings.items()})
                embedding_files.append(npz_path)

        # Aggregate embeddings from all videos into gold standard
        gold = aggregate_embeddings(embedding_files)
        
        # Save gold standard profile as JSON
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(gold, indent=2), encoding="utf-8")
        print(f"Gold standard profile saved to {output_json}")
        print(f"Total phonemes in profile: {len(gold)}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the gold standard profile creation script.
    
    Returns:
        Parsed arguments namespace containing:
        - raw_dir: Directory with raw GRID dataset files
        - mfa_dict: Path to MFA pronunciation dictionary
        - mfa_acoustic: Path to MFA acoustic model
        - output: Path for output JSON file
        - device: Computing device for neural networks
    """
    parser = argparse.ArgumentParser(
        description="Create a gold standard identity profile from a folder of authentic videos."
    )
    parser.add_argument(
        "raw_dir", 
        type=Path, 
        help="Raw folder with video+audio+align files (e.g., dataset/raw/speaker1)"
    )
    parser.add_argument(
        "--mfa-dict", 
        type=Path, 
        required=True, 
        help="Path to MFA pronunciation dictionary"
    )
    parser.add_argument(
        "--mfa-acoustic", 
        type=Path, 
        required=True, 
        help="Path to MFA acoustic model"
    )
    parser.add_argument(
        "-o", 
        "--output", 
        type=Path, 
        default=Path("dataset/output/gold_store/identity.json"), 
        help="Output JSON file path"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto", 
        help='Torch device (e.g., "cuda" or "cpu")'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_visual_emb(args.raw_dir, args.mfa_dict, args.mfa_acoustic, args.output, device=args.device)
