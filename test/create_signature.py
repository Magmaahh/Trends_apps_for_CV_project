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


def create_video_signature():
    pass

if __name__ == "__main__":
    
    create_audio_signature()
    
    # create_video_signature()