import os
import sys
import argparse
import subprocess
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_preparation.config import DATASET_OUTPUT
from data_preparation.pipeline import build_gold_dictionary_for_speaker

def run_gold_creation(adapter_path):
    print("\n=== STEP 1: Building Gold Dictionaries with Adapter ===")
    if not os.path.exists(DATASET_OUTPUT):
        print(f"ERRORE: {DATASET_OUTPUT} non esiste.")
        return

    # Find speaker folders
    embedding_folders = [d for d in os.listdir(DATASET_OUTPUT) 
                         if os.path.isdir(os.path.join(DATASET_OUTPUT, d)) 
                         and d.startswith("video_embeddings_s")]
    
    speakers = []
    for folder in embedding_folders:
        parts = folder.split("_")
        if len(parts) >= 3:
            spk = parts[-1] 
            speakers.append(spk)
    
    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    
    print(f"Found {len(speakers)} speakers.")
    
    for spk in tqdm(speakers):
        build_gold_dictionary_for_speaker(spk, adapter_path=adapter_path)

def run_cross_check(target, adapter_path):
    print(f"\n=== STEP 2: Running Cross-Check for {target} ===")
    cmd = [
        sys.executable, "demo/step6_cross_check.py",
        "--target", target,
        "--adapter", adapter_path
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to trained adapter model (.pth)")
    parser.add_argument("--target", type=str, default="s1", help="Target speaker for cross-check")
    args = parser.parse_args()

    # 1. Re-build Gold Dictionaries
    run_gold_creation(args.adapter)

    # 2. Run Cross-Check
    run_cross_check(args.target, args.adapter)

if __name__ == "__main__":
    main()
