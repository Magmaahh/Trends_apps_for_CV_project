from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import whisper  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    whisper = None

from .core import VideoFeatureExtractor
from .utils import parse_textgrid


class VideoPipeline:
    """
    End-to-end pipeline for processing videos and extracting phoneme-level embeddings.
    
    This pipeline automates the entire process from raw video to phoneme embeddings:
    
    Pipeline Steps:
    1. Extract audio from video (using ffmpeg)
    2. Transcribe audio to text (using Whisper, or use provided transcript)
    3. Align transcript to phonemes with timestamps (using Montreal Forced Aligner - MFA)
    4. For each phoneme interval, extract visual embedding from mouth movements
    
    Why this pipeline?
    - Automatic: No manual annotation needed
    - Phoneme-level: Analyzes specific speech sounds, not just whole words
    - Temporal alignment: Knows exactly when each phoneme is pronounced
    - Visual features: Captures how the mouth moves during each phoneme
    
    Use Cases:
    - Creating gold standard profiles from authentic videos
    - Processing test videos for deepfake detection
    """

    def __init__(self, mfa_dictionary_path: Path, mfa_acoustic_model_path: Path, device: str = "auto") -> None:
        """
        Initialize the pipeline with MFA models.
        
        Args:
            mfa_dictionary_path: Path to MFA pronunciation dictionary
                                (maps words to phoneme sequences)
            mfa_acoustic_model_path: Path to MFA acoustic model
                                    (trained model for phoneme alignment)
            device: Computing device for neural network inference
        """
        self.mfa_dictionary_path = Path(mfa_dictionary_path)
        self.mfa_acoustic_model_path = Path(mfa_acoustic_model_path)
        self.device = device
        self.extractor = VideoFeatureExtractor(device=device)

    def _transcribe(self, audio_path: Path, transcript_text: Optional[str]) -> str:
        """
        Get transcript text either from provided text or by transcribing audio.
        
        Args:
            audio_path: Path to audio file
            transcript_text: Optional pre-provided transcript
            
        Returns:
            Transcript text (uppercase)
        """
        if transcript_text:
            return transcript_text
        if whisper is None:
            raise RuntimeError("Whisper not installed: specify transcript_text or install openai-whisper.")
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        return result.get("text", "").strip()

    def _run_mfa(self, workdir: Path, wav_path: Path, transcript: str) -> Path:
        """
        Run Montreal Forced Aligner to get phoneme-level timestamps.
        
        MFA (Montreal Forced Aligner) is a tool that:
        1. Takes audio + transcript as input
        2. Uses acoustic models to detect phonemes in the audio
        3. Outputs a TextGrid file with precise timestamps for each phoneme
        
        Example TextGrid output:
        - "HELLO" might be aligned as:
          - HH: 0.0s - 0.1s
          - EH: 0.1s - 0.2s
          - L: 0.2s - 0.3s
          - OW: 0.3s - 0.5s
        
        Args:
            workdir: Temporary directory for MFA processing
            wav_path: Path to audio file
            transcript: Text transcript (what was said)
            
        Returns:
            Path to generated TextGrid file with phoneme alignments
        """
        # Create .lab file (MFA's expected format for transcripts)
        lab_path = workdir / f"{wav_path.stem}.lab"
        lab_path.write_text(transcript.upper(), encoding="utf-8")
        
        # Run MFA alignment
        # MFA expects wav + lab files in same folder
        cmd = [
            "mfa",
            "align",
            str(workdir),  # Input directory with wav + lab files
            str(self.mfa_dictionary_path),  # Pronunciation dictionary
            str(self.mfa_acoustic_model_path),  # Acoustic model
            str(workdir / "aligned"),  # Output directory
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check that TextGrid was created
        tg_path = workdir / "aligned" / f"{wav_path.stem}.TextGrid"
        if not tg_path.exists():
            raise FileNotFoundError(f"TextGrid not found after MFA: {tg_path}")
        return tg_path

    def process_single_video(self, video_path: str, transcript_text: Optional[str] = None) -> Dict[str, List[np.ndarray]]:
        """
        Process a single video end-to-end to extract phoneme embeddings.
        
        This is the main entry point for the pipeline. It orchestrates all steps:
        1. Audio extraction
        2. Transcription (if needed)
        3. Phoneme alignment
        4. Visual embedding extraction
        
        Args:
            video_path: Path to input video file
            transcript_text: Optional transcript (if None, will use Whisper to transcribe)
            
        Returns:
            Dictionary mapping phoneme labels to lists of embedding vectors.
            Example: {
                "IH1": [emb1, emb2],  # "IH1" appeared twice in video
                "AE1": [emb3],         # "AE1" appeared once
                ...
            }
            Each embedding is a 128-D numpy array representing mouth movements
            during that phoneme's pronunciation.
        """
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video not found: {video}")

        with tempfile.TemporaryDirectory(prefix="dfcv_") as tmpdir:
            workdir = Path(tmpdir)
            wav_path = workdir / f"{video.stem}.wav"

            # 1) Extract audio (mono 16k) with ffmpeg
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(video),
                "-ac",
                "1",
                "-ar",
                "16000",
                str(wav_path),
            ]
            proc = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0 or not wav_path.exists():
                raise RuntimeError("Audio extraction failed: check ffmpeg and video file.")

            # 2) Transcribe
            transcript = self._transcribe(wav_path, transcript_text)
            if not transcript:
                raise RuntimeError("Empty transcript: provide transcript_text or check Whisper.")

            # 3) MFA align
            tg_path = self._run_mfa(workdir, wav_path, transcript)

            # 4) Parse TextGrid
            intervals = parse_textgrid(str(tg_path), tier_name="phones")
            if not intervals:
                raise RuntimeError("No phoneme intervals found in TextGrid.")

            # 5) Embedding
            from .utils import load_video_frames

            frames, fps = load_video_frames(video)
            embeddings: Dict[str, List[np.ndarray]] = {}
            total_frames = len(frames)
            for interval in intervals:
                start_f = int(interval["start"] * fps)
                end_f = int(np.ceil(interval["end"] * fps))
                start_f, end_f = self._ensure_min_frames(start_f, end_f, total_frames)
                emb = self.extractor.process_video_interval(str(video), start_f, end_f)
                if emb is None:
                    continue
                phoneme = interval["phoneme"]
                embeddings.setdefault(phoneme, []).append(emb)

            return embeddings

    def _ensure_min_frames(self, start: int, end: int, total: int, min_frames: int = 4) -> tuple[int, int]:
        """
        Ensure a frame interval has at least min_frames by padding if necessary.
        
        Some phonemes are very short (e.g., stop consonants like "T" or "P").
        The ResNet3D model needs a minimum number of frames to work properly.
        If an interval is too short, we expand it by adding frames before/after.
        
        Args:
            start: Start frame index
            end: End frame index
            total: Total number of frames in video
            min_frames: Minimum required frames (default: 4)
            
        Returns:
            Tuple of (adjusted_start, adjusted_end) with at least min_frames duration
            
        Example:
            If phoneme spans frames 10-11 (only 1 frame), and min_frames=4:
            - Missing: 3 frames
            - Pad before: 1 frame (start becomes 9)
            - Pad after: 2 frames (end becomes 13)
            - Result: frames 9-13 (4 frames total)
        """
        duration = end - start
        if duration >= min_frames:
            return start, end
        
        # Calculate how many frames we need to add
        missing = min_frames - duration
        pad_before = missing // 2
        pad_after = missing - pad_before
        
        # Expand interval, respecting video boundaries
        new_start = max(0, start - pad_before)
        new_end = min(total, end + pad_after)
        return new_start, new_end
