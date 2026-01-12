"""
Extract Test Features Script

Extracts both video and audio embeddings from test videos for the multimodal identity matcher.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

# Import existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.audio.phonemes2emb import PhonemeEmbeddingExtractor
from src.video import VideoFeatureExtractor
from src.video.inference.utils import parse_textgrid, load_video_frames


class TestFeatureExtractor:
    """
    Extract both video and audio features from test videos.
    """
    
    def __init__(
        self,
        mfa_dict: str = "english_us_arpa",
        mfa_acoustic: str = "english_us_arpa",
        device: str = "auto"
    ):
        """
        Initialize extractors.
        
        Args:
            mfa_dict: MFA dictionary name (installed model)
            mfa_acoustic: MFA acoustic model name (installed model)
            device: Device for neural networks ("auto", "cuda", "cpu")
        """
        self.mfa_dict = mfa_dict
        self.mfa_acoustic = mfa_acoustic
        self.device = device
        
        # Initialize extractors
        self.video_extractor = VideoFeatureExtractor(device=device)
        self.audio_extractor = PhonemeEmbeddingExtractor()
        
        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def extract_audio(self, video_path: Path, output_dir: Path) -> Path:
        """
        Extract audio from video using ffmpeg.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save audio file
            
        Returns:
            Path to extracted audio file (.wav)
        """
        audio_path = output_dir / f"{video_path.stem}.wav"
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-ac", "1",  # mono
            "-ar", "16000",  # 16kHz sample rate
            "-y",  # overwrite
            str(audio_path)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0 or not audio_path.exists():
            raise RuntimeError(f"Failed to extract audio from {video_path.name}")
        
        return audio_path
    
    def create_lab_file(self, audio_path: Path, transcript: str) -> Path:
        """
        Create .lab file with transcript for MFA.
        
        Args:
            audio_path: Path to audio file
            transcript: Transcript text
            
        Returns:
            Path to created .lab file
        """
        lab_path = audio_path.parent / f"{audio_path.stem}.lab"
        lab_path.write_text(transcript.upper(), encoding="utf-8")
        return lab_path
    
    def run_mfa_alignment(self, work_dir: Path, output_dir: Path) -> Path:
        """
        Run MFA alignment to generate TextGrid with phoneme timestamps.
        
        Args:
            work_dir: Directory containing .wav and .lab files
            output_dir: Directory to save TextGrid files
            
        Returns:
            Path to output directory with TextGrid files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "mfa",
            "align",
            str(work_dir),
            self.mfa_dict,
            self.mfa_acoustic,
            str(output_dir)
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"MFA alignment failed: {result.stderr.decode()}")
        
        return output_dir
    
    def extract_mouth_crops(
        self,
        frames: List[np.ndarray],
        target_size: int = 112
    ) -> List[np.ndarray]:
        """
        Extract and crop mouth region from frames.
        
        Args:
            frames: List of video frames (BGR)
            target_size: Target size for crops (default: 112)
            
        Returns:
            List of cropped mouth images (grayscale, normalized)
        """
        crops = []
        
        for frame in frames:
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                continue
            
            # Get face landmarks
            landmarks = results.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            
            # Get mouth region coordinates
            # Using landmarks around mouth: 61, 291 (corners), 0, 17 (top/bottom)
            xs = [landmarks[i].x * w for i in [61, 291, 0, 17]]
            ys = [landmarks[i].y * h for i in [61, 291, 0, 17]]
            
            # Calculate crop region (2x mouth width)
            cx, cy = np.mean(xs), np.mean(ys)
            size = int((max(xs) - min(xs)) * 2)
            
            x1 = max(0, int(cx - size))
            y1 = max(0, int(cy - size))
            x2 = min(w, int(cx + size))
            y2 = min(h, int(cy + size))
            
            # Crop and resize
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crop = cv2.resize(crop, (target_size, target_size))
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop = crop.astype(np.float32) / 255.0
            
            crops.append(crop)
        
        return crops
    
    def extract_video_embeddings(
        self,
        video_path: Path,
        textgrid_path: Path,
        output_path: Path
    ) -> Dict[str, np.ndarray]:
        """
        Extract video embeddings for each phoneme.
        
        Args:
            video_path: Path to video file
            textgrid_path: Path to TextGrid file with phoneme alignment
            output_path: Path to save embeddings JSON
            
        Returns:
            Dictionary mapping phonemes to mean embeddings
        """
        # Parse TextGrid
        intervals = parse_textgrid(str(textgrid_path), tier_name="phones")
        if not intervals:
            raise RuntimeError(f"No phoneme intervals found in {textgrid_path.name}")
        
        # Load video frames
        frames, fps = load_video_frames(video_path)
        total_frames = len(frames)
        
        # Extract embeddings for each phoneme
        phoneme_embeddings: Dict[str, List[np.ndarray]] = {}
        
        for interval in intervals:
            phoneme = interval["phoneme"]
            start_f = int(interval["start"] * fps)
            end_f = int(np.ceil(interval["end"] * fps))
            
            # Ensure minimum 4 frames
            if end_f - start_f < 4:
                missing = 4 - (end_f - start_f)
                pad_before = missing // 2
                pad_after = missing - pad_before
                start_f = max(0, start_f - pad_before)
                end_f = min(total_frames, end_f + pad_after)
            
            # Extract frames for this phoneme
            phoneme_frames = frames[start_f:end_f]
            
            # Extract mouth crops
            crops = self.extract_mouth_crops(phoneme_frames)
            
            if len(crops) < 4:
                continue
            
            # Extract embedding using video extractor
            embedding = self.video_extractor.process_video_interval(
                str(video_path),
                start_f,
                end_f
            )
            
            if embedding is not None:
                phoneme_embeddings.setdefault(phoneme, []).append(embedding)
        
        # Aggregate to mean (gold standard format)
        gold_format = {}
        for phoneme, embs in phoneme_embeddings.items():
            if embs:
                mean_emb = np.mean(np.stack(embs), axis=0)
                gold_format[phoneme] = {
                    "vector": mean_emb.tolist(),
                    "count": len(embs)
                }
        
        # Save as JSON
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(gold_format, f, indent=2)
        
        return gold_format
    
    def extract_audio_embeddings(
        self,
        audio_path: Path,
        textgrid_path: Path,
        output_path: Path
    ) -> Dict[str, np.ndarray]:
        """
        Extract audio embeddings for each phoneme.
        
        Args:
            audio_path: Path to audio file
            textgrid_path: Path to TextGrid file with phoneme alignment
            output_path: Path to save embeddings NPZ
            
        Returns:
            Dictionary mapping phonemes to embeddings
        """
        # Extract embeddings using audio extractor
        results = self.audio_extractor.process_file(
            str(audio_path),
            str(textgrid_path)
        )
        
        if not results:
            raise RuntimeError(f"No audio embeddings extracted from {audio_path.name}")
        
        # Aggregate by phoneme (take mean if multiple occurrences)
        phoneme_dict = defaultdict(list)
        for item in results:
            phoneme_dict[item["phoneme"]].append(item["vector"])
        
        # Calculate mean for each phoneme
        voice_profile = {}
        for phoneme, vectors in phoneme_dict.items():
            voice_profile[phoneme] = np.mean(vectors, axis=0).astype(np.float32)
        
        # Save as NPZ
        np.savez(output_path, **voice_profile)
        
        return voice_profile
    
    def process_single_video(
        self,
        video_path: Path,
        transcript: str,
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Process a single video: extract all features.
        
        Args:
            video_path: Path to video file
            transcript: Transcript text
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary with paths to generated files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        video_stem = video_path.stem
        
        print(f"Processing: {video_path.name}")
        
        # Create temp directory for MFA
        with tempfile.TemporaryDirectory(prefix="mfa_") as tmpdir:
            work_dir = Path(tmpdir)
            
            # Step 1: Extract audio
            audio_path = self.extract_audio(video_path, work_dir)
            
            # Copy to output dir
            final_audio_path = output_dir / f"{video_stem}.wav"
            final_audio_path.write_bytes(audio_path.read_bytes())
            
            # Step 2: Create .lab file
            lab_path = self.create_lab_file(audio_path, transcript)
            final_lab_path = output_dir / f"{video_stem}.lab"
            final_lab_path.write_text(lab_path.read_text())
            
            # Step 3: Run MFA alignment
            mfa_output_dir = work_dir / "aligned"
            self.run_mfa_alignment(work_dir, mfa_output_dir)
            
            textgrid_path = mfa_output_dir / f"{video_stem}.TextGrid"
            if not textgrid_path.exists():
                raise RuntimeError(f"TextGrid not generated for {video_stem}")
            
            # Copy to output dir
            final_textgrid_path = output_dir / f"{video_stem}.TextGrid"
            final_textgrid_path.write_text(textgrid_path.read_text())
            
            # Step 4: Extract video embeddings
            video_emb_path = output_dir / f"{video_stem}_visual.json"
            video_embs = self.extract_video_embeddings(
                video_path,
                textgrid_path,
                video_emb_path
            )
            
            # Step 5: Extract audio embeddings
            audio_emb_path = output_dir / f"{video_stem}.npz"
            audio_embs = self.extract_audio_embeddings(
                final_audio_path,
                final_textgrid_path,
                audio_emb_path
            )
        
        print(f"Completed: {video_path.name}")
        
        return {
            "video": video_path,
            "audio": final_audio_path,
            "lab": final_lab_path,
            "textgrid": final_textgrid_path,
            "video_embeddings": video_emb_path,
            "audio_embeddings": audio_emb_path
        }
    
    def process_batch(
        self,
        video_dir: Path,
        transcripts_file: Path,
        output_dir: Path
    ) -> List[Dict[str, Path]]:
        """
        Process multiple videos from a directory.
        
        Args:
            video_dir: Directory containing video files
            transcripts_file: File with video-transcript pairs
            output_dir: Directory to save all outputs
            
        Returns:
            List of result dictionaries for each video
        """
        # Parse transcripts file
        transcripts = {}
        with transcripts_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    video_name, transcript = parts
                    transcripts[video_name.strip()] = transcript.strip()
        
        if not transcripts:
            raise ValueError(f"No transcripts found in {transcripts_file}")
        
        print(f"\nBatch processing: {len(transcripts)} videos")
        
        # Find video files
        video_files = []
        for video_name in transcripts.keys():
            video_path = video_dir / video_name
            if not video_path.exists():
                print(f"Warning: Video not found: {video_name}")
                continue
            video_files.append(video_path)
        
        if not video_files:
            raise RuntimeError("No valid video files found")
        
        # Process each video
        results = []
        for video_path in tqdm(video_files, desc="Processing videos"):
            transcript = transcripts[video_path.name]
            try:
                result = self.process_single_video(
                    video_path,
                    transcript,
                    output_dir / video_path.stem
                )
                results.append(result)
            except Exception as e:
                print(f"Error processing {video_path.name}: {e}")
                continue
        
        print(f"Batch complete: {len(results)}/{len(video_files)} videos processed")
        return results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract video and audio features from test videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single video with inline transcript
  python extract_test_features.py video.mp4 "BIN BLUE AT A FIVE AGAIN" -o test/output/
  
  # Batch processing
  python extract_test_features.py video_dir/ --transcripts transcripts.txt -o test/batch/
  
Transcripts file format (for batch):
  video1.mp4|BIN BLUE AT A FIVE AGAIN
  video2.mp4|SET RED BY B ZERO NOW
  video3.mp4|PLACE WHITE IN C EIGHT PLEASE
        """
    )
    
    parser.add_argument(
        "input",
        type=Path,
        help="Video file or directory containing videos"
    )
    
    parser.add_argument(
        "transcript",
        type=str,
        nargs="?",
        help="Transcript text (for single video mode)"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output directory for extracted features"
    )
    
    parser.add_argument(
        "--transcripts",
        type=Path,
        help="Path to transcripts file (for batch mode)"
    )
    
    parser.add_argument(
        "--mfa-dict",
        type=str,
        default="english_us_arpa",
        help="MFA dictionary name (default: english_us_arpa)"
    )
    
    parser.add_argument(
        "--mfa-acoustic",
        type=str,
        default="english_us_arpa",
        help="MFA acoustic model name (default: english_us_arpa)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for neural networks (default: auto)"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input not found: {args.input}")
        return
    
    # Determine mode (single video vs batch)
    is_batch = args.input.is_dir()
    
    if is_batch:
        if not args.transcripts:
            print("Error: --transcripts required for batch mode")
            return
        if not args.transcripts.exists():
            print(f"Error: Transcripts file not found: {args.transcripts}")
            return
    else:
        if not args.transcript:
            print("Error: transcript required for single video mode")
            return
    
    # Initialize extractor
    extractor = TestFeatureExtractor(
        mfa_dict=args.mfa_dict,
        mfa_acoustic=args.mfa_acoustic,
        device=args.device
    )
    
    # Process
    try:
        if is_batch:
            results = extractor.process_batch(
                args.input,
                args.transcripts,
                args.output
            )
        else:
            result = extractor.process_single_video(
                args.input,
                args.transcript,
                args.output
            )
            results = [result]
        
        # Summary
        print(f"Videos processed: {len(results)}")
        print(f"Output directory: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
