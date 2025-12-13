from __future__ import annotations

import argparse
from pathlib import Path

from src.video.matcher import IdentityMatcher
from src.video.pipeline import VideoPipeline


def audit_video(video_path: Path, gold_json: Path, mfa_dict: Path, mfa_acoustic: Path, threshold: float, device: str):
    pipeline = VideoPipeline(mfa_dict, mfa_acoustic, device=device)
    matcher = IdentityMatcher(gold_json, threshold=threshold)

    embeddings = pipeline.process_single_video(str(video_path))
    if not embeddings:
        raise RuntimeError("Nessun embedding estratto: controlla video/audio/volto.")

    report = matcher.compare(embeddings)
    verdict = report["verdict"]
    score = report["global_score"]
    if verdict == "REAL":
        print(f"✅ REAL (score={score:.3f} >= threshold={threshold})")
    else:
        print(f"❌ FAKE (score={score:.3f} < threshold={threshold})")
    print("Dettagli per fonema:", report.get("per_phoneme_best", {}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valuta un video contro una identità gold.")
    parser.add_argument("video", type=Path, help="Percorso al video da auditare")
    parser.add_argument("gold_json", type=Path, help="File JSON gold")
    parser.add_argument("--mfa-dict", type=Path, required=True, help="Percorso dizionario MFA")
    parser.add_argument("--mfa-acoustic", type=Path, required=True, help="Percorso modello acustico MFA")
    parser.add_argument("--threshold", type=float, default=0.75, help="Soglia di similarità")
    parser.add_argument("--device", type=str, default="auto", help='Dispositivo Torch (es. "cuda" o "cpu")')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    audit_video(args.video, args.gold_json, args.mfa_dict, args.mfa_acoustic, args.threshold, args.device)
