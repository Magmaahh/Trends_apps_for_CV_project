"""
Main entry point of the pipeline.

Pipeline stages:
1. Prepare MFA input for each speaker
2. Extract visual embeddings from aligned videos
3. Build gold phoneme dictionaries
"""

from .training_utils import load_speakers, load_embedding_speakers
from .training_pipeline import prepare_speaker_for_mfa, extract_embeddings_for_speaker, build_gold_dictionary_for_speaker

def main():
    # MFA preparation
    speakers = load_speakers()
    for spk in speakers:
        prepare_speaker_for_mfa(spk)

    # Embedding extraction and dictionary building
    embedding_speakers = load_embedding_speakers()
    for spk in embedding_speakers:
        extract_embeddings_for_speaker(spk)
        build_gold_dictionary_for_speaker(spk)

if __name__ == "__main__":
    main()