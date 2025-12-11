from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def fmt(status: bool, message: str) -> str:
    prefix = f"{GREEN}OK{RESET}" if status else f"{RED}FAIL{RESET}"
    return f"[{prefix}] {message}"


def check_dirs() -> Tuple[bool, str]:
    required = ["src", "scripts", "data"]
    missing = [d for d in required if not (ROOT / d).exists()]
    return (len(missing) == 0, f"Directories present: {', '.join(required)}" if not missing else f"Missing: {', '.join(missing)}")


def check_import() -> Tuple[bool, str]:
    try:
        from src.deepfake_cv import VideoFeatureExtractor  # noqa: F401

        return True, "Import src.deepfake_cv.VideoFeatureExtractor"
    except Exception as exc:
        return False, f"Import failed: {exc}"


def check_deps() -> Tuple[bool, str]:
    missing = []
    cuda = "unknown"
    for pkg in ["torch", "mediapipe", "cv2"]:
        try:
            importlib.import_module(pkg)
        except Exception:
            missing.append(pkg)
    try:
        import torch  # type: ignore

        cuda = "available" if torch.cuda.is_available() else "not available"
    except Exception:
        pass
    if missing:
        return False, f"Missing packages: {', '.join(missing)}; CUDA: {cuda}"
    return True, f"torch/mediapipe/cv2 present; CUDA: {cuda}"


if __name__ == "__main__":
    checks = [check_dirs, check_import, check_deps]
    for check in checks:
        ok, msg = check()
        print(fmt(ok, msg))
