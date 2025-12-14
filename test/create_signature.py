import os
# Fix OpenMP library conflict on macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path

# Add project root to path to enable absolute imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.audio.phonemes2emb import PhonemeEmbeddingExtractor
import numpy as np

AUDIO_FOLDER = os.path.join(os.path.dirname(__file__), "../dataset/output/mfa_workspace_s10")          # Folder with .wav files
TEXTGRID_FOLDER = os.path.join(os.path.dirname(__file__), "../dataset/output/mfa_output_phonemes_s10")   # Folder with .TextGrid files
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "signatures/s10/audio.npz")

def create_audio_signature():
    extractor = PhonemeEmbeddingExtractor()
    print("=== Voice Profile Extractor ===")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if folders exist
    if os.path.exists(AUDIO_FOLDER) and os.path.exists(TEXTGRID_FOLDER):
        # Extract the voice profile
        stats = extractor.extract_voice_profile(AUDIO_FOLDER, TEXTGRID_FOLDER, OUTPUT_FILE)
        
        if stats:
            print("\n--- Final statistics ---")
            print(f"Unique phonemes found: {', '.join(stats['phoneme_labels'][:10])}...")
            
            # Verify the created file
            if os.path.exists(OUTPUT_FILE):
                print(f"\n--- Output file verification ---")
                with np.load(OUTPUT_FILE) as data:
                    print(f"File {OUTPUT_FILE} contains {len(data.files)} phonemes:")
                    for phoneme in sorted(data.files)[:5]:
                        print(f"  {phoneme}: shape {data[phoneme].shape}, dtype {data[phoneme].dtype}")
    else:
        print(f"Folders not found.")


from src.video.inference.pipeline import VideoPipeline
from src.video.inference.utils import aggregate_embeddings, DEVICE

# Paths for Video Signature
VIDEO_GOLD_FOLDER = os.path.join(os.path.dirname(__file__), "samples/s1") 
VIDEO_SIGNATURE_FILE = os.path.join(os.path.dirname(__file__), "signatures/s1/video_gold.json")

def create_video_signature(source_folder=VIDEO_GOLD_FOLDER, output_file=VIDEO_SIGNATURE_FILE):
    print("\n=== Video Profile Extractor ===")
    
    # Initialize Pipeline
    # Note: You need MFA models downloaded. Assuming they are in dataset/output/mfa_data
    mfa_dict = Path("dataset/output/mfa_data/english_us_arpa.dict")
    mfa_model = Path("dataset/output/mfa_data/english_us_arpa.zip")
    
    if not mfa_dict.exists() or not mfa_model.exists():
        print("Error: MFA models not found. Please run 'mfa model download dictionary english_us_arpa' and 'mfa model download acoustic english_us_arpa'.")
        return

    pipeline = VideoPipeline(mfa_dict, mfa_model, device="auto")
    
    # Process videos
    if not os.path.exists(source_folder):
        print(f"Source folder {source_folder} not found.")
        return

    video_files = [f for f in os.listdir(source_folder) if f.endswith(('.mp4', '.mpg'))][:10] # Limit to 10 for speed
    all_embeddings = {} # phoneme -> list of vectors

    print(f"Processing {len(video_files)} videos from {source_folder}...")
    
    for v_file in video_files:
        path = os.path.join(source_folder, v_file)
        print(f"  - {v_file}...")
        try:
            # Process video (auto-transcribe if needed)
            embeddings = pipeline.process_single_video(path)
            
            for phoneme, vecs in embeddings.items():
                all_embeddings.setdefault(phoneme, []).extend(vecs)
        except Exception as e:
            print(f"    Error processing {v_file}: {e}")

    # Aggregate to create Gold Standard
    print("Aggregating embeddings...")
    gold_profile = {}
    for phoneme, vecs in all_embeddings.items():
        if len(vecs) < 5: # Min samples
            continue
        
        # Mean vector
        mean_vec = np.mean(vecs, axis=0)
        # Normalize
        mean_vec /= np.linalg.norm(mean_vec)
        
        gold_profile[phoneme] = {
            "vector": mean_vec.tolist(),
            "count": len(vecs)
        }

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    import json
    with open(output_file, "w") as f:
        json.dump(gold_profile, f, indent=4)
    
    print(f"Video signature saved to {output_file}")
    print(f"Profile contains {len(gold_profile)} phonemes.")

if __name__ == "__main__":
    # create_audio_signature()
    create_video_signature()