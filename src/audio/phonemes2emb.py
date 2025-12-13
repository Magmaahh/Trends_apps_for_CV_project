import textgrid
import librosa
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import os
from pathlib import Path
from collections import defaultdict

class PhonemeEmbeddingExtractor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        """
        Initialize the model and processor once.
        """
        print(f"Loading model: {model_name}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval() # Set model to evaluation mode (no dropout, etc.)
        print("Model loaded.")

    def process_file(self, audio_path, textgrid_path):
        """
        Takes file paths as input and returns a list of dictionaries.
        Each dictionary represents a phoneme with its embedding.
        """
        
        # 1. Audio Loading and Resampling
        # Wav2Vec2 requires audio at 16kHz
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return []

        # 2. TextGrid Loading
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
        except Exception as e:
            print(f"Error loading TextGrid {textgrid_path}: {e}")
            return []

        # Find the "phones" tier (or use the second tier if not found by name)
        phone_tier = None
        for tier in tg:
            if tier.name == "phones":
                phone_tier = tier
                break
        
        # Fallback: if "phones" not found, try to use the second tier (index 1)
        if phone_tier is None and len(tg) > 1:
            phone_tier = tg[1]
            print(f"Warning: Tier 'phones' not found by name. Using tier '{phone_tier.name}'.")

        if phone_tier is None:
            print("Error: Unable to find a valid tier in the TextGrid.")
            return []

        # 3. Model Inference (Get frame-by-frame embeddings)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # shape: (1, num_frames, 768) -> remove batch dimension -> (num_frames, 768)
        full_embeddings = outputs.last_hidden_state.squeeze(0)
        
        # 4. Calculate Time -> Frame ratio
        # We need to know how many seconds correspond to a single model frame
        num_frames = full_embeddings.shape[0]
        duration_seconds = len(audio) / sr
        seconds_per_frame = duration_seconds / num_frames
        
        extracted_data = []

        # 5. Iterate over phonemes and slice
        for interval in phone_tier:
            phoneme_label = interval.mark
            
            # Filter empty labels or silences (often marked as "", "sil", or "sp")
            if not phoneme_label or phoneme_label in ["", "sil", "sp"]:
                continue
            
            # Calculate start/end indices in the embedding tensor
            start_idx = int(interval.minTime / seconds_per_frame)
            end_idx = int(interval.maxTime / seconds_per_frame)
            
            # Correction for very short phonemes (if start == end, take at least 1 frame)
            if start_idx == end_idx:
                end_idx += 1
            
            # Bounds check (to avoid crash if textgrid slightly exceeds audio)
            if start_idx >= num_frames:
                continue
            end_idx = min(end_idx, num_frames)

            # Extract tensor slice
            phoneme_tensor = full_embeddings[start_idx:end_idx, :]
            
            # Mean Pooling: calculate the mean of frames to get a single vector for this phoneme
            # Result: vector of dimension (768,)
            phoneme_vector = torch.mean(phoneme_tensor, dim=0).numpy()
            
            extracted_data.append({
                "phoneme": phoneme_label,     # E.g.: "IH1"
                "start_time": interval.minTime,
                "end_time": interval.maxTime,
                "vector": phoneme_vector      # numpy array (768,)
            })
            
        return extracted_data

    def extract_voice_profile(self, audio_folder, textgrid_folder, output_path):
        """
        Extracts a voice profile from a folder of audio and TextGrid files.
        
        Args:
            audio_folder: Path to the folder containing audio files (.wav)
            textgrid_folder: Path to the folder containing TextGrid files (.TextGrid)
            output_path: Path to the output .npz file to save the voice profile
        
        Returns:
            dict: Dictionary with extraction statistics
        """
        audio_folder = Path(audio_folder)
        textgrid_folder = Path(textgrid_folder)
        
        # Find all .wav files
        audio_files = sorted(list(audio_folder.glob("*.wav")))
        
        if not audio_files:
            print(f"Error: No .wav files found in {audio_folder}")
            return None
        
        print(f"\nFound {len(audio_files)} audio files in {audio_folder}")
        
        # Dictionary to accumulate embeddings: {phoneme_label: [list of embeddings]}
        phoneme_embeddings = defaultdict(list)
        
        processed_files = 0
        skipped_files = 0
        total_phonemes = 0
        
        # Process each audio file
        for audio_file in audio_files:
            # Find the corresponding TextGrid file
            textgrid_file = textgrid_folder / f"{audio_file.stem}.TextGrid"
            
            if not textgrid_file.exists():
                print(f"Warning: TextGrid file not found for {audio_file.name}, skipping this file.")
                skipped_files += 1
                continue
            
            print(f"Processing: {audio_file.name} + {textgrid_file.name}")
            
            # Extract embeddings for this file
            results = self.process_file(str(audio_file), str(textgrid_file))
            
            if not results:
                print(f"  No phonemes extracted from {audio_file.name}")
                skipped_files += 1
                continue
            
            # Accumulate embeddings for each phoneme
            for item in results:
                phoneme_label = item["phoneme"]
                embedding = item["vector"]
                phoneme_embeddings[phoneme_label].append(embedding)
                total_phonemes += 1
            
            processed_files += 1
            print(f"  Extracted {len(results)} phonemes")
        
        if not phoneme_embeddings:
            print("\nError: No phonemes extracted from any file!")
            return None
        
        # Calculate mean embedding for each phoneme
        print(f"\n--- Computing mean embeddings ---")
        voice_profile = {}
        
        for phoneme_label, embeddings_list in phoneme_embeddings.items():
            # Convert list of arrays to a 2D array: (num_occurrences, 768)
            embeddings_array = np.array(embeddings_list)
            
            # Calculate mean along axis 0 to get a single embedding (768,)
            mean_embedding = np.mean(embeddings_array, axis=0).astype(np.float32)
            
            voice_profile[phoneme_label] = mean_embedding
            
            print(f"  {phoneme_label}: {len(embeddings_list)} occurrences -> mean embedding (768,)")
        
        # Save voice profile in .npz format
        np.savez(output_path, **voice_profile)
        
        print(f"\n=== Extraction completed ===")
        print(f"Files processed: {processed_files}/{len(audio_files)}")
        print(f"Files skipped: {skipped_files}")
        print(f"Total phonemes extracted: {total_phonemes}")
        print(f"Unique phonemes in voice profile: {len(voice_profile)}")
        print(f"Voice profile saved to: {output_path}")
        
        return {
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "total_phonemes": total_phonemes,
            "unique_phonemes": len(voice_profile),
            "phoneme_labels": sorted(voice_profile.keys())
        }

# --- USAGE EXAMPLE ---

if __name__ == "__main__":
    
    extractor = PhonemeEmbeddingExtractor()
    print("=== Voice Profile from folders ===")
    
    # Define input folders and output file
    audio_folder = os.path.join(os.path.dirname(__file__), "../../dataset/output/mfa_workspace_s1")          # Folder with .wav files
    textgrid_folder = os.path.join(os.path.dirname(__file__), "../../dataset/output/mfa_output_phonemes_s1")   # Folder with .TextGrid files
    output_file = "voice_profile_s1.npz" #TODO salvare in una cartella specifica
    
    # Check if folders exist
    if os.path.exists(audio_folder) and os.path.exists(textgrid_folder):
        # Extract the voice profile
        stats = extractor.extract_voice_profile(audio_folder, textgrid_folder, output_file)
        
        if stats:
            print("\n--- Final statistics ---")
            print(f"Unique phonemes found: {', '.join(stats['phoneme_labels'][:10])}...")
            
            # Verify the created file
            if os.path.exists(output_file):
                print(f"\n--- Output file verification ---")
                with np.load(output_file) as data:
                    print(f"File {output_file} contains {len(data.files)} phonemes:")
                    for phoneme in sorted(data.files)[:5]:
                        print(f"  {phoneme}: shape {data[phoneme].shape}, dtype {data[phoneme].dtype}")
    else:
        print(f"Folders not found. To use this method:")
        print(f"1. Create the folder '{audio_folder}' with .wav files")
        print(f"2. Create the folder '{textgrid_folder}' with corresponding .TextGrid files")
        print(f"3. Run the script again")
        print("\nNote: Audio and TextGrid files must have the same base name")
        print("Example: audio/speaker01.wav + textgrids/speaker01.TextGrid")
