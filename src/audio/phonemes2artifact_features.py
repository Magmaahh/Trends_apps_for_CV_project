#!/usr/bin/env python3
"""
Phoneme-Based Artifact Feature Extraction for Deepfake Detection

This module extracts authenticity-oriented features (NOT identity-oriented)
from audio at the phoneme level, designed to capture synthesis artifacts.

Features extracted per phoneme:
- MFCC statistics (mean, variance)
- LFCC (Low-Frequency Cepstral Coefficients)
- Phase-based features (variance, jitter)
- Harmonic features (HNR, harmonic stability)
- Formant features (F1, F2, F3 variance)
- Spectral features (flatness, centroid, rolloff)
- Energy modulation features

Usage:
    extractor = PhonemeArtifactExtractor()
    features = extractor.process_file("audio.wav", "audio.TextGrid")
"""

import textgrid
import librosa
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.fftpack import dct
from scipy.stats import kurtosis, skew
import parselmouth
from parselmouth.praat import call


class PhonemeArtifactExtractor:
    """
    Extract artifact-specific features from audio at phoneme level.
    
    Unlike Wav2Vec2 (identity-oriented), these features capture:
    - Spectral anomalies (MFCC/LFCC variance)
    - Phase inconsistencies (synthesis artifacts)
    - Harmonic irregularities (unnatural voice quality)
    - Formant instabilities (vocal tract modeling errors)
    """
    
    def __init__(self, sr=16000, n_mfcc=13, n_lfcc=13):
        """
        Initialize feature extractor
        
        Args:
            sr: Sample rate (16kHz for consistency with Wav2Vec2)
            n_mfcc: Number of MFCC coefficients
            n_lfcc: Number of LFCC coefficients
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_lfcc = n_lfcc
        
        print(f"PhonemeArtifactExtractor initialized (sr={sr}Hz)")
    
    def extract_mfcc_features(self, audio_segment):
        """
        Extract MFCC features (spectral shape)
        
        Args:
            audio_segment: Audio numpy array
            
        Returns:
            dict with MFCC statistics
        """
        # Compute MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio_segment, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc
        )
        
        # Statistics across time
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)
        mfcc_skew = skew(mfcc, axis=1)
        mfcc_kurt = kurtosis(mfcc, axis=1)
        
        # Delta and delta-delta (temporal dynamics)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        delta_mean = np.mean(mfcc_delta, axis=1)
        delta2_mean = np.mean(mfcc_delta2, axis=1)
        
        return {
            'mfcc_mean': mfcc_mean,
            'mfcc_var': mfcc_var,
            'mfcc_skew': mfcc_skew,
            'mfcc_kurt': mfcc_kurt,
            'mfcc_delta_mean': delta_mean,
            'mfcc_delta2_mean': delta2_mean
        }
    
    def extract_lfcc_features(self, audio_segment):
        """
        Extract LFCC features (low-frequency focus, better for deepfake)
        
        Args:
            audio_segment: Audio numpy array
            
        Returns:
            dict with LFCC statistics
        """
        # Linear frequency scale (vs mel scale in MFCC)
        # Compute STFT
        stft = librosa.stft(audio_segment)
        magnitude = np.abs(stft)
        
        # Linear filterbank (low-frequency emphasis)
        n_filters = self.n_lfcc
        filterbank = librosa.filters.mel(
            sr=self.sr, 
            n_fft=2048, 
            n_mels=n_filters,
            fmin=0,
            fmax=4000  # Focus on low frequencies
        )
        
        # Apply filterbank
        mel_spec = np.dot(filterbank, magnitude)
        
        # Log compression
        log_mel = np.log(mel_spec + 1e-10)
        
        # DCT (discrete cosine transform)
        lfcc = dct(log_mel, axis=0, norm='ortho')[:n_filters]
        
        # Statistics
        lfcc_mean = np.mean(lfcc, axis=1)
        lfcc_var = np.var(lfcc, axis=1)
        
        return {
            'lfcc_mean': lfcc_mean,
            'lfcc_var': lfcc_var
        }
    
    def extract_phase_features(self, audio_segment):
        """
        Extract phase-based features (synthesis artifacts)
        
        Args:
            audio_segment: Audio numpy array
            
        Returns:
            dict with phase statistics
        """
        # Compute STFT
        stft = librosa.stft(audio_segment)
        phase = np.angle(stft)
        
        # Phase variance (instability indicator)
        phase_var = np.var(phase)
        
        # Phase difference (temporal discontinuities)
        phase_diff = np.diff(phase, axis=1)
        phase_diff_var = np.var(phase_diff)
        
        # Instantaneous frequency deviation
        inst_freq = np.diff(np.unwrap(phase, axis=1), axis=1)
        inst_freq_var = np.var(inst_freq)
        
        # Group delay (phase derivative)
        group_delay = -np.gradient(np.unwrap(phase, axis=0), axis=0)
        group_delay_var = np.var(group_delay)
        
        return {
            'phase_var': float(phase_var),
            'phase_diff_var': float(phase_diff_var),
            'inst_freq_var': float(inst_freq_var),
            'group_delay_var': float(group_delay_var)
        }
    
    def extract_harmonic_features(self, audio_segment):
        """
        Extract harmonic features using Praat/Parselmouth
        
        Args:
            audio_segment: Audio numpy array
            
        Returns:
            dict with harmonic statistics
        """
        try:
            # Create Praat sound object
            sound = parselmouth.Sound(audio_segment, sampling_frequency=self.sr)
            
            # Harmonic-to-Noise Ratio (HNR)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            # Pitch (F0) statistics
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            f0_mean = call(pitch, "Get mean", 0, 0, "Hertz")
            f0_std = call(pitch, "Get standard deviation", 0, 0, "Hertz")
            
            # Handle undefined values
            hnr = float(hnr) if not np.isnan(hnr) else 0.0
            f0_mean = float(f0_mean) if not np.isnan(f0_mean) else 0.0
            f0_std = float(f0_std) if not np.isnan(f0_std) else 0.0
            
            return {
                'hnr': hnr,
                'f0_mean': f0_mean,
                'f0_std': f0_std
            }
            
        except Exception as e:
            print(f"Warning: Harmonic feature extraction failed: {e}")
            return {
                'hnr': 0.0,
                'f0_mean': 0.0,
                'f0_std': 0.0
            }
    
    def extract_formant_features(self, audio_segment):
        """
        Extract formant features (vocal tract resonances)
        
        Args:
            audio_segment: Audio numpy array
            
        Returns:
            dict with formant statistics
        """
        try:
            # Create Praat sound object
            sound = parselmouth.Sound(audio_segment, sampling_frequency=self.sr)
            
            # Extract formants
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            # Get F1, F2, F3 means
            f1_mean = call(formant, "Get mean", 1, 0, 0, "Hertz")
            f2_mean = call(formant, "Get mean", 2, 0, 0, "Hertz")
            f3_mean = call(formant, "Get mean", 3, 0, 0, "Hertz")
            
            # Get F1, F2, F3 standard deviations
            f1_std = call(formant, "Get standard deviation", 1, 0, 0, "Hertz")
            f2_std = call(formant, "Get standard deviation", 2, 0, 0, "Hertz")
            f3_std = call(formant, "Get standard deviation", 3, 0, 0, "Hertz")
            
            # Handle undefined values
            f1_mean = float(f1_mean) if not np.isnan(f1_mean) else 0.0
            f2_mean = float(f2_mean) if not np.isnan(f2_mean) else 0.0
            f3_mean = float(f3_mean) if not np.isnan(f3_mean) else 0.0
            f1_std = float(f1_std) if not np.isnan(f1_std) else 0.0
            f2_std = float(f2_std) if not np.isnan(f2_std) else 0.0
            f3_std = float(f3_std) if not np.isnan(f3_std) else 0.0
            
            return {
                'f1_mean': f1_mean,
                'f2_mean': f2_mean,
                'f3_mean': f3_mean,
                'f1_std': f1_std,
                'f2_std': f2_std,
                'f3_std': f3_std
            }
            
        except Exception as e:
            print(f"Warning: Formant feature extraction failed: {e}")
            return {
                'f1_mean': 0.0,
                'f2_mean': 0.0,
                'f3_mean': 0.0,
                'f1_std': 0.0,
                'f2_std': 0.0,
                'f3_std': 0.0
            }
    
    def extract_spectral_features(self, audio_segment):
        """
        Extract spectral features
        
        Args:
            audio_segment: Audio numpy array
            
        Returns:
            dict with spectral statistics
        """
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=self.sr)
        centroid_mean = np.mean(centroid)
        centroid_var = np.var(centroid)
        
        # Spectral flatness (noisiness)
        flatness = librosa.feature.spectral_flatness(y=audio_segment)
        flatness_mean = np.mean(flatness)
        flatness_var = np.var(flatness)
        
        # Spectral rolloff (high-frequency content)
        rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=self.sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_var = np.var(rolloff)
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=self.sr)
        bandwidth_mean = np.mean(bandwidth)
        bandwidth_var = np.var(bandwidth)
        
        # Zero-crossing rate (temporal feature)
        zcr = librosa.feature.zero_crossing_rate(audio_segment)
        zcr_mean = np.mean(zcr)
        zcr_var = np.var(zcr)
        
        return {
            'spectral_centroid_mean': float(centroid_mean),
            'spectral_centroid_var': float(centroid_var),
            'spectral_flatness_mean': float(flatness_mean),
            'spectral_flatness_var': float(flatness_var),
            'spectral_rolloff_mean': float(rolloff_mean),
            'spectral_rolloff_var': float(rolloff_var),
            'spectral_bandwidth_mean': float(bandwidth_mean),
            'spectral_bandwidth_var': float(bandwidth_var),
            'zcr_mean': float(zcr_mean),
            'zcr_var': float(zcr_var)
        }
    
    def extract_energy_features(self, audio_segment):
        """
        Extract energy modulation features
        
        Args:
            audio_segment: Audio numpy array
            
        Returns:
            dict with energy statistics
        """
        # RMS energy
        rms = librosa.feature.rms(y=audio_segment)
        rms_mean = np.mean(rms)
        rms_var = np.var(rms)
        
        # Energy envelope
        envelope = np.abs(signal.hilbert(audio_segment))
        envelope_var = np.var(envelope)
        
        # Energy modulation instability
        energy_diff = np.diff(rms[0])
        energy_instability = np.var(energy_diff)
        
        return {
            'rms_mean': float(rms_mean),
            'rms_var': float(rms_var),
            'envelope_var': float(envelope_var),
            'energy_instability': float(energy_instability)
        }
    
    def process_file(self, audio_path, textgrid_path):
        """
        Process audio file and extract artifact features per phoneme
        
        Args:
            audio_path: Path to audio file (.wav)
            textgrid_path: Path to TextGrid file with phoneme alignments
            
        Returns:
            List of dicts, one per phoneme, with all features
        """
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return []
        
        # Load TextGrid
        try:
            tg = textgrid.TextGrid.fromFile(textgrid_path)
        except Exception as e:
            print(f"Error loading TextGrid {textgrid_path}: {e}")
            return []
        
        # Find phones tier
        phone_tier = None
        for tier in tg:
            if tier.name == "phones":
                phone_tier = tier
                break
        
        if phone_tier is None and len(tg) > 1:
            phone_tier = tg[1]
        
        if phone_tier is None:
            print("Error: Unable to find phones tier")
            return []
        
        extracted_data = []
        
        # Process each phoneme
        for interval in phone_tier:
            phoneme_label = interval.mark
            
            # Skip silence/empty
            if not phoneme_label or phoneme_label in ["", "sil", "sp"]:
                continue
            
            # Extract audio segment
            start_sample = int(interval.minTime * sr)
            end_sample = int(interval.maxTime * sr)
            
            if start_sample >= len(audio) or end_sample > len(audio):
                continue
            
            audio_segment = audio[start_sample:end_sample]
            
            # Skip very short segments (need at least 512 samples for good STFT)
            if len(audio_segment) < 512:  # ~32ms minimum
                continue
            
            # Extract all features
            features = {
                'phoneme': phoneme_label,
                'start_time': interval.minTime,
                'end_time': interval.maxTime,
                'duration': interval.maxTime - interval.minTime
            }
            
            # MFCC features
            try:
                mfcc_feats = self.extract_mfcc_features(audio_segment)
                features.update(mfcc_feats)
            except Exception as e:
                print(f"Warning: MFCC extraction failed for {phoneme_label}: {e}")
            
            # LFCC features
            try:
                lfcc_feats = self.extract_lfcc_features(audio_segment)
                features.update(lfcc_feats)
            except Exception as e:
                print(f"Warning: LFCC extraction failed for {phoneme_label}: {e}")
            
            # Phase features
            try:
                phase_feats = self.extract_phase_features(audio_segment)
                features.update(phase_feats)
            except Exception as e:
                print(f"Warning: Phase extraction failed for {phoneme_label}: {e}")
            
            # Harmonic features
            try:
                harmonic_feats = self.extract_harmonic_features(audio_segment)
                features.update(harmonic_feats)
            except Exception as e:
                print(f"Warning: Harmonic extraction failed for {phoneme_label}: {e}")
            
            # Formant features
            try:
                formant_feats = self.extract_formant_features(audio_segment)
                features.update(formant_feats)
            except Exception as e:
                print(f"Warning: Formant extraction failed for {phoneme_label}: {e}")
            
            # Spectral features
            try:
                spectral_feats = self.extract_spectral_features(audio_segment)
                features.update(spectral_feats)
            except Exception as e:
                print(f"Warning: Spectral extraction failed for {phoneme_label}: {e}")
            
            # Energy features
            try:
                energy_feats = self.extract_energy_features(audio_segment)
                features.update(energy_feats)
            except Exception as e:
                print(f"Warning: Energy extraction failed for {phoneme_label}: {e}")
            
            extracted_data.append(features)
        
        return extracted_data


def main():
    """Test the artifact feature extractor"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python phonemes2artifact_features.py <audio.wav> <audio.TextGrid>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    textgrid_path = sys.argv[2]
    
    print(f"Processing: {audio_path}")
    print(f"TextGrid: {textgrid_path}")
    
    extractor = PhonemeArtifactExtractor()
    results = extractor.process_file(audio_path, textgrid_path)
    
    print(f"\nExtracted features for {len(results)} phonemes")
    
    if results:
        print("\nExample features for first phoneme:")
        first = results[0]
        print(f"Phoneme: {first['phoneme']}")
        print(f"Duration: {first['duration']:.3f}s")
        print(f"Features extracted: {len(first)-4}")  # -4 for metadata fields
        
        # Print some feature names
        feature_names = [k for k in first.keys() if k not in ['phoneme', 'start_time', 'end_time', 'duration']]
        print(f"\nFeature categories:")
        print(f"  - MFCC features: {sum(1 for k in feature_names if 'mfcc' in k)}")
        print(f"  - LFCC features: {sum(1 for k in feature_names if 'lfcc' in k)}")
        print(f"  - Phase features: {sum(1 for k in feature_names if 'phase' in k or 'inst_freq' in k or 'group_delay' in k)}")
        print(f"  - Harmonic features: {sum(1 for k in feature_names if 'hnr' in k or 'f0' in k)}")
        print(f"  - Formant features: {sum(1 for k in feature_names if k.startswith('f1') or k.startswith('f2') or k.startswith('f3'))}")
        print(f"  - Spectral features: {sum(1 for k in feature_names if 'spectral' in k or 'zcr' in k)}")
        print(f"  - Energy features: {sum(1 for k in feature_names if 'rms' in k or 'energy' in k or 'envelope' in k)}")
        
        print(f"\nTotal features per phoneme: {len(feature_names)}")


if __name__ == "__main__":
    main()
