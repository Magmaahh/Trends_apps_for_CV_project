# Fix OpenMP duplicate library issue on macOS
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import textgrid
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from pathlib import Path
from collections import defaultdict

# Single File Embedding Extractor Class
class SingleFileEmbeddingExtractor:
    """Extract embeddings from a single audio file"""
    
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        print(f"Loading model: {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()  # Evaluation mode (no dropout)
        print("Model loaded successfully")
    
    def extract_embeddings(self, audio_path, textgrid_path):
        """
        Extract phoneme embeddings from a single audio + TextGrid file pair.
        
        Args:
            audio_path: Path to the audio file (.wav)
            textgrid_path: Path to the TextGrid file (.TextGrid)
            
        Returns:
            Dictionary mapping phoneme labels to lists of embedding vectors
            Returns None if extraction fails
        """
        audio_path = Path(audio_path)
        textgrid_path = Path(textgrid_path)
        
        print(f"\nProcessing: {audio_path.name}")
        
        # Load audio (Wav2Vec2 requires 16kHz)
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            print(f"Audio loaded: {len(audio)/sr:.2f}s at {sr}Hz")
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None
        
        # Load TextGrid
        try:
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
            print(f"TextGrid loaded: {len(tg)} tiers")
        except Exception as e:
            print(f"Error loading TextGrid: {e}")
            return None
        
        # Find the phoneme tier
        phone_tier = None
        for tier in tg:
            if tier.name == "phones":
                phone_tier = tier
                break
        
        # Fallback to second tier if "phones" not found
        if phone_tier is None and len(tg) > 1:
            phone_tier = tg[1]
            print(f"Tier 'phones' not found, using tier '{phone_tier.name}'")
        
        if phone_tier is None:
            print(f"No valid phoneme tier found in TextGrid")
            return None
        
        print(f"Using tier: {phone_tier.name} ({len(phone_tier)} intervals)")
        
        # Extract embeddings using Wav2Vec2
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Shape: (1, num_frames, 768) -> (num_frames, 768)
        full_embeddings = outputs.last_hidden_state.squeeze(0)
        
        # Calculate time to frame mapping
        num_frames = full_embeddings.shape[0]
        duration_seconds = len(audio) / sr
        seconds_per_frame = duration_seconds / num_frames
        
        print(f"Extracted {num_frames} frames ({full_embeddings.shape[1]}-dimensional)")
        
        # Extract embeddings for each phoneme
        phoneme_embeddings = defaultdict(list)
        extracted_count = 0
        skipped_count = 0
        
        for interval in phone_tier:
            phoneme_label = interval.mark
            
            # Skip empty labels and silences
            if not phoneme_label or phoneme_label in ["", "sil", "sp"]:
                skipped_count += 1
                continue
            
            # Calculate frame indices
            start_idx = int(interval.minTime / seconds_per_frame)
            end_idx = int(interval.maxTime / seconds_per_frame)
            
            # Handle very short phonemes
            if start_idx == end_idx:
                end_idx += 1
            
            # Bounds check
            if start_idx >= num_frames:
                skipped_count += 1
                continue
            end_idx = min(end_idx, num_frames)
            
            # Extract and pool phoneme frames
            phoneme_tensor = full_embeddings[start_idx:end_idx, :]
            phoneme_vector = torch.mean(phoneme_tensor, dim=0).numpy()
            
            phoneme_embeddings[phoneme_label].append(phoneme_vector)
            extracted_count += 1
        
        print(f"Extracted {extracted_count} phonemes ({skipped_count} skipped)")
        print(f"Found {len(phoneme_embeddings)} unique phoneme types")
        
        return phoneme_embeddings
    
    def save_signature(
        self,
        audio_path,
        textgrid_path,
        output_path,
        average_duplicates=False
    ):
        """
        Extract embeddings and save as a voice signature (.npz file).
        
        Args:
            audio_path: Path to the audio file (.wav)
            textgrid_path: Path to the TextGrid file (.TextGrid)
            output_path: Path where to save the .npz file
            average_duplicates: If True, average multiple occurrences of the same phoneme.
                              If False, keep all occurrences as a 2D array.
        
        Returns:
            Dictionary with statistics about the extraction, or None if failed
        """
        # Extract embeddings
        phoneme_embeddings = self.extract_embeddings(audio_path, textgrid_path)
        
        if phoneme_embeddings is None:
            return None
        
        # Prepare data for saving
        signature_data = {}
        
        print(f"\nPreparing signature:")
        for phoneme_label, embeddings_list in sorted(phoneme_embeddings.items()):
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            
            if average_duplicates:
                # Average all occurrences into a single vector
                signature_data[phoneme_label] = np.mean(embeddings_array, axis=0)
                print(f"  {phoneme_label:8s}: {len(embeddings_list)} occurrences -> averaged to 1D array ({embeddings_array.shape[1]},)")
            else:
                # Keep all occurrences
                if len(embeddings_list) == 1:
                    signature_data[phoneme_label] = embeddings_array[0]
                    print(f"  {phoneme_label:8s}: 1 occurrence -> 1D array ({embeddings_array.shape[1]},)")
                else:
                    signature_data[phoneme_label] = embeddings_array
                    print(f"  {phoneme_label:8s}: {len(embeddings_list)} occurrences -> 2D array {embeddings_array.shape}")
        
        # Save to .npz file
        output_path = Path(output_path)
        np.savez(str(output_path), **signature_data)
        
        print(f"\n{'='*80}")
        print(f"Signature saved to: {output_path}")
        print(f"{'='*80}")
        
        # Return statistics
        total_occurrences = sum(len(embs) for embs in phoneme_embeddings.values())
        
        stats = {
            "output_file": str(output_path),
            "unique_phonemes": len(phoneme_embeddings),
            "total_phoneme_occurrences": total_occurrences,
            "phoneme_labels": sorted(phoneme_embeddings.keys()),
            "averaged": average_duplicates
        }
        
        return stats


def main():
    """Main function."""
    
    # Paths
    audio_file = "../../test/s2/bbaf1n.wav"
    textgrid_file = "../../test/s2/bbaf1n.TextGrid"
    output_file = "../../test/s2/voice_sig.npz"
    
    # Set to True to average multiple occurrences of the same phoneme
    # Set to False to keep all occurrences (like the original test files)
    average_duplicates = False

    print("="*80)
    print("SINGLE FILE TO SIGNATURE CONVERTER")
    print("="*80)
    
    # Check if input files exist
    audio_path = Path(audio_file)
    textgrid_path = Path(textgrid_file)
    
    if not audio_path.exists():
        print(f"\nError: Audio file not found: {audio_path}")
        print("\nPlease update the 'audio_file' path in the script.")
        return
    
    if not textgrid_path.exists():
        print(f"\nError: TextGrid file not found: {textgrid_path}")
        print("\nPlease update the 'textgrid_file' path in the script.")
        return
    
    print(f"\nInput files:")
    print(f"  Audio:    {audio_path}")
    print(f"  TextGrid: {textgrid_path}")
    print(f"  Output:   {output_file}")
    print(f"  Average duplicates: {average_duplicates}")
    
    # Create extractor and process
    extractor = SingleFileEmbeddingExtractor()
    
    stats = extractor.save_signature(
        audio_path,
        textgrid_path,
        output_file,
        average_duplicates=average_duplicates
    )
    
    if stats:
        print(f"\n{'='*80}")
        print("EXTRACTION STATISTICS")
        print(f"{'='*80}")
        print(f"Unique phonemes: {stats['unique_phonemes']}")
        print(f"Total occurrences: {stats['total_phoneme_occurrences']}")
        print(f"Averaged: {stats['averaged']}")
        print(f"\nPhoneme labels:")
        for i, phoneme in enumerate(stats['phoneme_labels']):
            if i % 10 == 0 and i > 0:
                print()
            print(f"  {phoneme}", end="")
        print()
        
        # Verify the output file
        print(f"\n{'='*80}")
        print("FILE VERIFICATION")
        print(f"{'='*80}")
        with np.load(output_file) as data:
            print(f"File contains {len(data.files)} phonemes:")
            for phoneme in sorted(data.files):
                arr = data[phoneme]
                print(f"  {phoneme:8s}: shape {arr.shape}, dtype {arr.dtype}")
        
        print(f"\nSuccess! You can now use this signature for comparison.")
        print(f"\nExample:")
        print(f"  python src/compare_npz_similarity.py {output_file} test/s1/bbaf2n.npz")
    else:
        print("\nExtraction failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
