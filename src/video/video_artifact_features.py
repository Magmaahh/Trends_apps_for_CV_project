#!/usr/bin/env python3
"""
Video Artifact Feature Extraction for Deepfake Detection

This module extracts authenticity-oriented features from video at the phoneme level,
designed to capture visual synthesis artifacts.

Features extracted per phoneme:
- Lip aperture dynamics (mean, velocity, acceleration, smoothness)
- Optical flow patterns (magnitude, direction consistency)
- Audio-video synchronization (lip-sync quality)

Unlike identity-oriented ResNet3D embeddings, these features capture:
- Unnatural temporal dynamics in lip movements
- Inconsistent optical flow patterns
- Audio-video misalignment typical of deepfakes
"""

import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from scipy.signal import savgol_filter
from scipy.stats import entropy


class VideoArtifactExtractor:
    """
    Extract artifact-specific features from video at phoneme level.
    
    Unlike ResNet3D (identity-oriented), these features capture:
    - Temporal dynamics of lip movements (velocity, acceleration)
    - Optical flow inconsistencies
    - Audio-video synchronization quality
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        # MediaPipe Face Mesh for landmark detection
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Lip landmark indices (outer lip contour)
        # Upper lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
        # Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
        self.upper_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        self.lower_lip_indices = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        
        print("VideoArtifactExtractor initialized")
    
    def compute_lip_aperture(self, landmarks, frame_shape) -> float:
        """
        Compute lip aperture (vertical distance between upper and lower lip)
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: (height, width) of frame
            
        Returns:
            Normalized lip aperture (0-1 range)
        """
        h, w = frame_shape[:2]
        
        # Get upper and lower lip center points
        upper_y = np.mean([landmarks[idx].y * h for idx in self.upper_lip_indices])
        lower_y = np.mean([landmarks[idx].y * h for idx in self.lower_lip_indices])
        
        # Vertical distance
        aperture = abs(lower_y - upper_y)
        
        # Normalize by face height (approximate)
        face_height = h * 0.3  # Approximate face region
        normalized_aperture = aperture / face_height
        
        return float(normalized_aperture)
    
    def extract_lip_dynamics_features(self, video_path: str, start_frame: int, end_frame: int) -> Dict:
        """
        Extract lip aperture dynamics features
        
        Returns dict with:
        - lip_aperture_mean, lip_aperture_std
        - lip_velocity_mean, lip_velocity_std
        - lip_acceleration_mean, lip_acceleration_std
        - lip_smoothness
        - lip_range
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._get_empty_lip_features()
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        apertures = []
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                aperture = self.compute_lip_aperture(landmarks, frame.shape)
                apertures.append(aperture)
            
            frame_idx += 1
        
        cap.release()
        
        if len(apertures) < 3:
            return self._get_empty_lip_features()
        
        apertures = np.array(apertures)
        
        # Smooth signal to reduce noise
        if len(apertures) >= 5:
            apertures_smooth = savgol_filter(apertures, window_length=min(5, len(apertures)), polyorder=2)
        else:
            apertures_smooth = apertures
        
        # Compute derivatives
        dt = 1.0 / fps
        velocity = np.gradient(apertures_smooth, dt)
        acceleration = np.gradient(velocity, dt)
        
        # Smoothness (jitter) - variance of acceleration
        smoothness = np.var(acceleration) if len(acceleration) > 0 else 0.0
        
        # Spectral entropy of velocity (frequency irregularity)
        if len(velocity) > 0:
            fft = np.abs(np.fft.fft(velocity))
            fft_norm = fft / (np.sum(fft) + 1e-10)
            velocity_entropy = entropy(fft_norm + 1e-10)
        else:
            velocity_entropy = 0.0
        
        return {
            'lip_aperture_mean': float(np.mean(apertures)),
            'lip_aperture_std': float(np.std(apertures)),
            'lip_velocity_mean': float(np.mean(np.abs(velocity))),
            'lip_velocity_std': float(np.std(velocity)),
            'lip_acceleration_mean': float(np.mean(np.abs(acceleration))),
            'lip_acceleration_std': float(np.std(acceleration)),
            'lip_smoothness': float(smoothness),
            'lip_range': float(np.max(apertures) - np.min(apertures)),
            'lip_velocity_entropy': float(velocity_entropy)
        }
    
    def extract_optical_flow_features(self, video_path: str, start_frame: int, end_frame: int) -> Dict:
        """
        Extract optical flow features from mouth region
        
        Returns dict with:
        - flow_mag_mean, flow_mag_std
        - flow_dir_consistency
        - flow_spatial_variation
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._get_empty_flow_features()
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read first frame
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return self._get_empty_flow_features()
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Get mouth ROI
        mouth_roi = self._get_mouth_roi(prev_frame)
        if mouth_roi is None:
            cap.release()
            return self._get_empty_flow_features()
        
        x, y, w, h = mouth_roi
        prev_roi = prev_gray[y:y+h, x:x+w]
        
        magnitudes = []
        directions = []
        spatial_vars = []
        
        frame_idx = start_frame + 1
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_roi = gray[y:y+h, x:x+w]
            
            # Compute optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, curr_roi,
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Flow magnitude and direction
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            magnitudes.append(np.mean(mag))
            directions.append(ang.flatten())
            spatial_vars.append(np.var(mag))
            
            prev_roi = curr_roi
            frame_idx += 1
        
        cap.release()
        
        if len(magnitudes) == 0:
            return self._get_empty_flow_features()
        
        # Direction consistency (entropy of direction histogram)
        all_directions = np.concatenate(directions)
        hist, _ = np.histogram(all_directions, bins=36, range=(0, 2*np.pi))
        hist_norm = hist / (np.sum(hist) + 1e-10)
        dir_entropy = entropy(hist_norm + 1e-10)
        dir_consistency = 1.0 / (1.0 + dir_entropy)  # Higher = more consistent
        
        return {
            'flow_mag_mean': float(np.mean(magnitudes)),
            'flow_mag_std': float(np.std(magnitudes)),
            'flow_dir_consistency': float(dir_consistency),
            'flow_spatial_variation': float(np.mean(spatial_vars))
        }
    
    def extract_lipsync_features(self, video_path: str, audio_path: str, 
                                  start_frame: int, end_frame: int,
                                  start_time: float, end_time: float) -> Dict:
        """
        Extract audio-video synchronization features
        
        Returns dict with:
        - lipsync_correlation
        - lipsync_lag
        - lipsync_quality
        """
        import librosa
        
        # Extract lip aperture from video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._get_empty_lipsync_features()
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        apertures = []
        frame_idx = start_frame
        
        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                aperture = self.compute_lip_aperture(landmarks, frame.shape)
                apertures.append(aperture)
            
            frame_idx += 1
        
        cap.release()
        
        if len(apertures) < 3:
            return self._get_empty_lipsync_features()
        
        # Extract audio energy envelope
        try:
            audio, sr = librosa.load(audio_path, sr=16000, offset=start_time, duration=end_time-start_time)
            
            # RMS energy envelope
            hop_length = int(sr / fps)
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            
            # Align lengths
            min_len = min(len(apertures), len(rms))
            apertures = np.array(apertures[:min_len])
            rms = rms[:min_len]
            
            if len(apertures) < 3:
                return self._get_empty_lipsync_features()
            
            # Cross-correlation
            correlation = np.correlate(apertures - np.mean(apertures), 
                                      rms - np.mean(rms), mode='same')
            correlation = correlation / (np.std(apertures) * np.std(rms) * len(apertures) + 1e-10)
            
            # Find lag (in frames)
            max_idx = np.argmax(np.abs(correlation))
            lag = max_idx - len(correlation) // 2
            lag_ms = (lag / fps) * 1000  # Convert to milliseconds
            
            # Max correlation coefficient
            max_corr = float(correlation[max_idx])
            
            # Quality score (penalize large lags)
            lag_penalty = np.exp(-abs(lag_ms) / 40.0)  # 40ms threshold
            quality = max_corr * lag_penalty
            
            return {
                'lipsync_correlation': max_corr,
                'lipsync_lag_ms': float(lag_ms),
                'lipsync_quality': float(quality)
            }
            
        except Exception as e:
            print(f"Warning: Lip-sync extraction failed: {e}")
            return self._get_empty_lipsync_features()
    
    def _get_mouth_roi(self, frame) -> Optional[Tuple[int, int, int, int]]:
        """Get mouth region of interest (x, y, w, h)"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Get mouth bounding box
        mouth_indices = self.upper_lip_indices + self.lower_lip_indices
        xs = [landmarks[idx].x * w for idx in mouth_indices]
        ys = [landmarks[idx].y * h for idx in mouth_indices]
        
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        
        # Add padding
        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def _get_empty_lip_features(self) -> Dict:
        """Return zero-filled lip features"""
        return {
            'lip_aperture_mean': 0.0,
            'lip_aperture_std': 0.0,
            'lip_velocity_mean': 0.0,
            'lip_velocity_std': 0.0,
            'lip_acceleration_mean': 0.0,
            'lip_acceleration_std': 0.0,
            'lip_smoothness': 0.0,
            'lip_range': 0.0,
            'lip_velocity_entropy': 0.0
        }
    
    def _get_empty_flow_features(self) -> Dict:
        """Return zero-filled optical flow features"""
        return {
            'flow_mag_mean': 0.0,
            'flow_mag_std': 0.0,
            'flow_dir_consistency': 0.0,
            'flow_spatial_variation': 0.0
        }
    
    def _get_empty_lipsync_features(self) -> Dict:
        """Return zero-filled lip-sync features"""
        return {
            'lipsync_correlation': 0.0,
            'lipsync_lag_ms': 0.0,
            'lipsync_quality': 0.0
        }
    
    def process_video_interval(self, video_path: str, audio_path: str,
                               start_frame: int, end_frame: int,
                               start_time: float, end_time: float,
                               phoneme: str) -> Dict:
        """
        Process a video interval and extract all artifact features
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            start_frame, end_frame: Frame interval
            start_time, end_time: Time interval (seconds)
            phoneme: Phoneme label
            
        Returns:
            Dictionary with all features + metadata
        """
        features = {
            'phoneme': phoneme,
            'start_frame': start_frame,
            'end_frame': end_frame,
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time
        }
        
        # Extract lip dynamics
        try:
            lip_features = self.extract_lip_dynamics_features(video_path, start_frame, end_frame)
            features.update(lip_features)
        except Exception as e:
            print(f"Warning: Lip dynamics extraction failed for {phoneme}: {e}")
            features.update(self._get_empty_lip_features())
        
        # Extract optical flow
        try:
            flow_features = self.extract_optical_flow_features(video_path, start_frame, end_frame)
            features.update(flow_features)
        except Exception as e:
            print(f"Warning: Optical flow extraction failed for {phoneme}: {e}")
            features.update(self._get_empty_flow_features())
        
        # Extract lip-sync
        try:
            lipsync_features = self.extract_lipsync_features(
                video_path, audio_path, start_frame, end_frame, start_time, end_time
            )
            features.update(lipsync_features)
        except Exception as e:
            print(f"Warning: Lip-sync extraction failed for {phoneme}: {e}")
            features.update(self._get_empty_lipsync_features())
        
        return features


def main():
    """Test the video artifact extractor"""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python video_artifact_features.py <video_path> <audio_path>")
        print("\nExample test on first 30 frames...")
        sys.exit(1)
    
    video_path = sys.argv[1]
    audio_path = sys.argv[2]
    
    print(f"Processing: {video_path}")
    print(f"Audio: {audio_path}")
    
    extractor = VideoArtifactExtractor()
    
    # Test on first 30 frames (1 second at 30fps)
    features = extractor.process_video_interval(
        video_path=video_path,
        audio_path=audio_path,
        start_frame=0,
        end_frame=30,
        start_time=0.0,
        end_time=1.0,
        phoneme="TEST"
    )
    
    print(f"\nExtracted features:")
    print(f"Phoneme: {features['phoneme']}")
    print(f"Duration: {features['duration']:.3f}s")
    print(f"\nFeature counts:")
    print(f"  - Lip dynamics: 9 features")
    print(f"  - Optical flow: 4 features")
    print(f"  - Lip-sync: 3 features")
    print(f"  Total: 16 features per phoneme")
    
    print(f"\nSample values:")
    for key, value in features.items():
        if key not in ['phoneme', 'start_frame', 'end_frame', 'start_time', 'end_time', 'duration']:
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
