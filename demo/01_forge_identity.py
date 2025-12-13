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

from src.video.pipeline import VideoPipeline
from src.video.utils import aggregate_embeddings


def run_mfa_batch(wav_dir: Path, mfa_dict: Path, mfa_acoustic: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["mfa", "align", str(wav_dir), str(mfa_dict), str(mfa_acoustic), str(output_dir)]
    subprocess.run(cmd, check=True)
    return output_dir


def prepare_lab_files(raw_dir: Path, tmp_dir: Path) -> Path:
    lab_dir = tmp_dir
    align_files = list(raw_dir.glob("*.align")) + list(raw_dir.rglob("*.align"))
    if not align_files:
        raise FileNotFoundError(f"Nessun file .align trovato in {raw_dir}")
    for align_file in align_files:
        stem = align_file.stem
        wav_src = raw_dir / f"{stem}.wav"
        if not wav_src.exists():
            continue
        lab_text = align_file.read_text(encoding="utf-8", errors="ignore").strip().upper()
        (lab_dir / f"{stem}.lab").write_text(lab_text, encoding="utf-8")
        shutil.copy2(wav_src, lab_dir / wav_src.name)
    return lab_dir


def forge_identity(
    raw_dir: Path, mfa_dict: Path, mfa_acoustic: Path, output_json: Path, device: str = "auto"
) -> None:
    with tempfile.TemporaryDirectory(prefix="forge_") as tmp:
        tmp_dir = Path(tmp)
        wav_lab_dir = prepare_lab_files(raw_dir, tmp_dir)
        tg_dir = tmp_dir / "aligned"
        run_mfa_batch(wav_lab_dir, mfa_dict, mfa_acoustic, tg_dir)

        pipeline = VideoPipeline(mfa_dict, mfa_acoustic, device=device)
        embedding_files: List[Path] = []

        tg_files = list(tg_dir.glob("*.TextGrid"))
        for tg_file in tqdm(tg_files, desc="Video"):
            stem = tg_file.stem
            video_path = raw_dir / f"{stem}.mpg"
            if not video_path.exists():
                video_path = raw_dir / f"{stem}.mp4"
            if not video_path.exists():
                continue
            embeddings = pipeline.process_single_video(str(video_path))
            if embeddings:
                npz_path = tmp_dir / f"{stem}.npz"
                np.savez_compressed(npz_path, **{k: np.stack(v) for k, v in embeddings.items()})
                embedding_files.append(npz_path)

        gold = aggregate_embeddings(embedding_files)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(gold, indent=2), encoding="utf-8")
        print(f"Identity salvata in {output_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crea un file identità (gold) da una cartella raw.")
    parser.add_argument("raw_dir", type=Path, help="Cartella raw con video+audio+align (es. dataset/init/s1)")
    parser.add_argument("--mfa-dict", type=Path, required=True, help="Percorso dizionario MFA")
    parser.add_argument("--mfa-acoustic", type=Path, required=True, help="Percorso modello acustico MFA")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("dataset/output/gold_store/identity.json"), help="Percorso JSON output"
    )
    parser.add_argument("--device", type=str, default="auto", help='Dispositivo Torch (es. "cuda" o "cpu")')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    forge_identity(args.raw_dir, args.mfa_dict, args.mfa_acoustic, args.output, device=args.device)
