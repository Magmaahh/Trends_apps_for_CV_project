import os
import sys
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from tqdm import tqdm

# Add project root to path if run as script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    __package__ = "src.video.inference"

from .utils import DATASET_OUTPUT, GOLD_STORE_DIR, DEVICE
from .model import VideoEmbeddingAdapter
from src.video.training.dataset_prep import build_gold_dictionary_for_speaker

# --- CROSS CHECK LOGIC ---

# Loads gold dictionary for a speaker
def load_gold_dictionary(speaker_id):
    path = os.path.join(GOLD_STORE_DIR, f"{speaker_id}_gold_dictionary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# Performs cross-identity verification test
def cross_identity_test(target_speaker="s1", adapter_path=None):
    print(f"\n--- CROSS IDENTITY CHECK: Target {target_speaker} ---")
    
    # Load Target Gold Dictionary
    target_gold = load_gold_dictionary(target_speaker)
    if not target_gold:
        print(f"Error: Gold dictionary for {target_speaker} not found.")
        return

    gold_vectors = {k: np.array(v["vector"]) for k, v in target_gold.items()}
    
    # Find Impostors
    embedding_folders = [d for d in os.listdir(DATASET_OUTPUT) 
                         if os.path.isdir(os.path.join(DATASET_OUTPUT, d)) 
                         and d.startswith("video_embeddings_s")]
    
    print(f"Checking against {len(embedding_folders)} potential impostors...")

    # Load Adapter
    adapter = None
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        adapter = VideoEmbeddingAdapter().to(DEVICE)
        adapter.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
        adapter.eval()

    results = {}

    for folder in tqdm(embedding_folders):
        impostor_id = folder.split("_")[-1]
        
        emb_path = os.path.join(DATASET_OUTPUT, folder)
        files = [f for f in os.listdir(emb_path) if f.endswith(".npz")]
        
        # Limit files for speed
        files = files[:50] 
        
        video_scores = []
        
        for filename in files:
            data = np.load(os.path.join(emb_path, filename))
            
            for phoneme, embeddings in data.items():
                if phoneme in gold_vectors:
                    target_vec = gold_vectors[phoneme]
                    
                    # Apply adapter if present
                    if adapter:
                        tensor_in = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
                        with torch.no_grad():
                            embeddings = adapter(tensor_in).cpu().numpy()

                    for vec in embeddings:
                        sim = 1 - cosine(vec, target_vec)
                        video_scores.append(sim)
        
        if video_scores:
            avg_score = np.mean(video_scores)
            results[impostor_id] = avg_score

    # 3. Visualization & Report
    print("\RESULTS:")
    sorted_res = sorted(results.items(), key=lambda x: int(x[0][1:]) if x[0][1:].isdigit() else 999)
    
    impostor_scores = []
    self_score = 0
    
    for spk, score in sorted_res:
        status = "BLOCKED" if score < 0.75 else "PASSED"
        if spk == target_speaker:
            status = "SELF"
            self_score = score
        else:
            impostor_scores.append(score)
            
        print(f"{spk:<10} | {score:.4f}     | {status}")

    avg_impostor = np.mean(impostor_scores) if impostor_scores else 0
    security_gap = self_score - avg_impostor
    
    print("-" * 40)
    print(f"AVERAGE IMPOSTORS: {avg_impostor:.4f}")
    print(f"SECURITY GAP:   {security_gap:.4f}")

    # Plot
    plt.figure(figsize=(12, 6))
    speakers = [x[0] for x in sorted_res]
    scores = [x[1] for x in sorted_res]
    colors = ['blue' if s == target_speaker else ('red' if sc > 0.75 else 'green') for s, sc in zip(speakers, scores)]
    
    sns.barplot(x=speakers, y=scores, palette=colors)
    plt.axhline(y=0.75, color='r', linestyle='--', label='Threshold (0.75)')
    plt.title(f"Identity Verification (Target: {target_speaker})")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cross_check_result.png")
    print("\nPlot saved to cross_check_result.png")


# --- MAIN VERIFICATION FLOW ---

# Runs gold dictionary creation for all speakers
def run_gold_creation(adapter_path):
    print("\n=== STEP 1: Building Gold Dictionaries with Adapter ===")
    if not os.path.exists(DATASET_OUTPUT):
        print(f"ERROR: {DATASET_OUTPUT} does not exist.")
        return

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, required=True, help="Path to trained adapter model (.pth)")
    parser.add_argument("--target", type=str, default="s1", help="Target speaker for cross-check")
    args = parser.parse_args()

    # Re-build Gold Dictionaries
    run_gold_creation(args.adapter)

    # Run Cross-Check
    cross_identity_test(args.target, args.adapter)

if __name__ == "__main__":
    main()
