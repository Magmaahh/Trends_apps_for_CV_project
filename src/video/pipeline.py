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
    """One-click pipeline: transcribe -> MFA align -> crop -> embed."""

    def __init__(self, mfa_dictionary_path: Path, mfa_acoustic_model_path: Path, device: str = "auto") -> None:
        self.mfa_dictionary_path = Path(mfa_dictionary_path)
        self.mfa_acoustic_model_path = Path(mfa_acoustic_model_path)
        self.device = device
        self.extractor = VideoFeatureExtractor(device=device)

    def _transcribe(self, audio_path: Path, transcript_text: Optional[str]) -> str:
        if transcript_text:
            return transcript_text
        if whisper is None:
            raise RuntimeError("Whisper non installato: specifica transcript_text oppure installa openai-whisper.")
        model = whisper.load_model("base")
        result = model.transcribe(str(audio_path))
        return result.get("text", "").strip()

    def _run_mfa(self, workdir: Path, wav_path: Path, transcript: str) -> Path:
        lab_path = workdir / f"{wav_path.stem}.lab"
        lab_path.write_text(transcript.upper(), encoding="utf-8")
        # MFA expects wav + lab in same folder
        cmd = [
            "mfa",
            "align",
            str(workdir),
            str(self.mfa_dictionary_path),
            str(self.mfa_acoustic_model_path),
            str(workdir / "aligned"),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        tg_path = workdir / "aligned" / f"{wav_path.stem}.TextGrid"
        if not tg_path.exists():
            raise FileNotFoundError(f"TextGrid non trovato dopo MFA: {tg_path}")
        return tg_path

    def process_single_video(self, video_path: str, transcript_text: Optional[str] = None) -> Dict[str, List[np.ndarray]]:
        """
        Process a single video end-to-end.
        Returns dict phoneme -> list of embedding vectors.
        """
        video = Path(video_path)
        if not video.exists():
            raise FileNotFoundError(f"Video non trovato: {video}")

        with tempfile.TemporaryDirectory(prefix="dfcv_") as tmpdir:
            workdir = Path(tmpdir)
            wav_path = workdir / f"{video.stem}.wav"

            # 1) Estrai audio (mono 16k) con ffmpeg
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
                raise RuntimeError("Estrazione audio fallita: controlla ffmpeg e il file video.")

            # 2) Trascrivi
            transcript = self._transcribe(wav_path, transcript_text)
            if not transcript:
                raise RuntimeError("Transcript vuoto: fornisci transcript_text o verifica Whisper.")

            # 3) MFA align
            tg_path = self._run_mfa(workdir, wav_path, transcript)

            # 4) Parse TextGrid
            intervals = parse_textgrid(str(tg_path), tier_name="phones")
            if not intervals:
                raise RuntimeError("Nessun intervallo fonema trovato in TextGrid.")

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
        duration = end - start
        if duration >= min_frames:
            return start, end
        missing = min_frames - duration
        pad_before = missing // 2
        pad_after = missing - pad_before
        new_start = max(0, start - pad_before)
        new_end = min(total, end + pad_after)
        return new_start, new_end
