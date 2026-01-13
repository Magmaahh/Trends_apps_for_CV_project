import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from scipy.spatial.distance import cosine

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.video.inference.pipeline import VideoPipeline
from src.video.inference.model import VideoEmbeddingAdapter
from src.video.inference.utils import DEVICE

def load_signature(path):
    with open(path, "r") as f:
        return json.load(f)

def compare_video(test_video_path, signature_path, adapter_path=None):
    print(f"Video Verification")
    print(f"Test Video: {test_video_path}")
    print(f"Signature:  {signature_path}")
    
    # Load Signature
    if not os.path.exists(signature_path):
        print("Error: Signature file not found.")
        return
    signature = load_signature(signature_path)
    gold_vectors = {k: np.array(v["vector"]) for k, v in signature.items()}
    
    # Load Adapter (Optional)
    adapter = None
    if adapter_path and os.path.exists(adapter_path):
        print(f"Using Adapter: {adapter_path}")
        adapter = VideoEmbeddingAdapter().to(DEVICE)
        adapter.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
        adapter.eval()
    
    # Process Test Video
    # Using default MFA paths on Mac
    home = Path.home()
    mfa_dict = home / "Documents/MFA/pretrained_models/dictionary/english_us_arpa.dict"
    mfa_model = home / "Documents/MFA/pretrained_models/acoustic/english_us_arpa.zip"
    
    if not mfa_dict.exists() or not mfa_model.exists():
        print("Error: MFA models not found in dataset/output/mfa_data/")
        return

    pipeline = VideoPipeline(mfa_dict, mfa_model, device="auto")
    
    try:
        print("Processing video (Extracting -> Transcribing -> Aligning -> Embedding)...")
        embeddings = pipeline.process_single_video(test_video_path)
    except Exception as e:
        print(f"Error processing video: {e}")
        return

    # Compare
    scores = []
    
    print("\nPhoneme Comparison:")
    print(f"{'Phoneme':<10} | {'Count':<5} | {'Avg Similarity':<15}")
    print("-" * 35)
    
    for phoneme, vecs in embeddings.items():
        if phoneme in gold_vectors:
            target_vec = gold_vectors[phoneme]
            
            # Apply adapter
            if adapter:
                tensor_in = torch.tensor(vecs, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    vecs = adapter(tensor_in).cpu().numpy()
            
            phoneme_scores = []
            for vec in vecs:
                sim = 1 - cosine(vec, target_vec)
                phoneme_scores.append(sim)
                scores.append(sim)
            
            avg_p_score = np.mean(phoneme_scores)
            print(f"{phoneme:<10} | {len(vecs):<5} | {avg_p_score:.4f}")

    if not scores:
        print("\nNo common phonemes found between video and signature.")
        return

    final_score = np.mean(scores)
    threshold = 0.75 # Default threshold
    
    print("-" * 35)
    print(f"FINAL SCORE: {final_score:.4f}")
    
    if final_score >= threshold:
        print("VERIFICATION PASSED (Identity Match)")
    else:
        print("VERIFICATION FAILED (Potential Impostor)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test video file")
    parser.add_argument("--signature", required=True, help="Path to gold signature JSON")
    parser.add_argument("--adapter", help="Path to trained adapter model")
    
    args = parser.parse_args()
    
    compare_video(args.test, args.signature, args.adapter)
