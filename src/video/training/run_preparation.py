import sys
import os

# Add project root to path if run as script
if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
    __package__ = "src.video.training"

from .utils import load_speakers, load_embedding_speakers
from .dataset_prep import prepare_speaker_for_mfa, extract_embeddings_for_speaker, build_gold_dictionary_for_speaker

def main():
    # MFA preparation
    print("=== STEP 1: PREPARE MFA ===")
    speakers = load_speakers()
    for spk in speakers:
        print(f"Processing {spk}...")
        prepare_speaker_for_mfa(spk)

    # Note: MFA alignment (Step 2) is run via shell script run_mfa.sh

    # Embedding extraction
    print("\n=== STEP 3: EXTRACT EMBEDDINGS ===")
    for spk in speakers:
        extract_embeddings_for_speaker(spk)

    # Gold dictionary creation
    print("\n=== STEP 4: BUILD GOLD DICTIONARIES ===")
    processed_speakers = load_embedding_speakers()
    for spk in processed_speakers:
        print(f"Building Gold Dictionary for {spk}...")
        build_gold_dictionary_for_speaker(spk)

if __name__ == "__main__":
    main()
