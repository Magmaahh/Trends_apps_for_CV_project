from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from torchvision.models.video import R3D_18_Weights, r3d_18


class VideoFeatureExtractor:
    """
    Extract visual embeddings from mouth regions during speech.
    
    This class is the core component for analyzing how a person's mouth moves
    when pronouncing different phonemes. It uses:
    - MediaPipe Face Mesh: to detect and track facial landmarks (especially lips)
    - ResNet3D (R3D-18): a 3D CNN to extract spatio-temporal features from mouth movements
    
    The output is a fixed-size embedding vector that represents the visual characteristics
    of mouth movements during a specific time interval (typically corresponding to a phoneme).
    
    Key Concept:
    Each person has unique mouth movement patterns when speaking. By extracting embeddings
    for each phoneme, we can create a "visual voice profile" that's hard to fake convincingly.
    """

    def __init__(
        self,
        device: str = "auto",
        img_size: int = 112,
        embedding_dim: int = 128,
        min_frames: int = 4,
    ) -> None:
        """
        Initialize the feature extractor.
        
        Args:
            device: Computing device ("auto", "cuda", or "cpu")
            img_size: Size to resize mouth patches (112x112 pixels)
            embedding_dim: Dimension of output embedding vectors (128-D)
            min_frames: Minimum frames required to extract a valid embedding
        """
        self.device = self._resolve_device(device)
        self.img_size = img_size
        self.min_frames = min_frames
        self.model = self._load_model(embedding_dim).to(self.device)
        self.model.eval()

        # MediaPipe Face Mesh for detecting facial landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )
        # Key lip landmark indices from MediaPipe's 468-point face mesh
        # These points define the outer contour of the lips
        self.lip_landmarks: List[int] = [61, 291, 0, 17, 78, 308, 13, 14]

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, embedding_dim: int) -> nn.Module:
        """
        Load and adapt ResNet3D model for grayscale mouth video analysis.
        
        ResNet3D (R3D-18) is a 3D convolutional neural network designed for video understanding.
        It processes sequences of frames to capture temporal patterns (motion over time).
        
        Modifications:
        1. Convert first conv layer from RGB (3 channels) to grayscale (1 channel)
           - Original: expects RGB video input
           - Modified: expects grayscale mouth patches
        2. Replace final FC layer to output embeddings of desired dimension (128-D)
        
        Returns:
            Modified ResNet3D model ready for mouth movement analysis
        """
        # Load pretrained ResNet3D-18 model
        backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        
        # Modify first convolutional layer to accept grayscale input (1 channel instead of 3)
        old_conv = backbone.stem[0]
        new_conv = nn.Conv3d(
            1,  # Input: 1 channel (grayscale)
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Initialize new conv weights by averaging RGB weights
        with torch.no_grad():
            new_conv.weight[:] = torch.sum(old_conv.weight, dim=1, keepdim=True)
        backbone.stem[0] = new_conv
        
        # Replace final layer to output embedding_dim features (default 128)
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        return backbone

    def extract_mouth_patch(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract and normalize mouth region from a single video frame.
        
        Process:
        1. Detect face using MediaPipe Face Mesh
        2. Locate lip landmarks
        3. Calculate bounding box around mouth (with 1.8x padding for context)
        4. Crop, resize to 112x112, convert to grayscale
        5. Normalize pixel values to [0, 1]
        
        Args:
            frame: BGR video frame from OpenCV
            
        Returns:
            Normalized grayscale mouth patch (112x112) or None if face not detected
        """
        if frame is None or frame.size == 0:
            return None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape
        xs = [landmarks[idx].x * w for idx in self.lip_landmarks]
        ys = [landmarks[idx].y * h for idx in self.lip_landmarks]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        box_size = max(int(max(max(xs) - min(xs), max(ys) - min(ys)) * 1.8), self.img_size)
        half = box_size // 2

        x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
        x2, y2 = min(w, int(cx + half)), min(h, int(cy + half))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return (crop.astype(np.float32) / 255.0)

    def get_embedding(self, frames_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate embedding vector from a sequence of mouth patches.
        
        The ResNet3D model processes the temporal sequence of mouth images
        and outputs a single embedding vector that captures the motion pattern.
        
        Args:
            frames_tensor: Tensor of shape (1, 1, T, H, W) where:
                - 1st dim: batch size (always 1)
                - 2nd dim: channels (1 for grayscale)
                - T: number of frames (temporal dimension)
                - H, W: height and width (112x112)
        
        Returns:
            Embedding vector of shape (embedding_dim,) - typically 128-D
        """
        if frames_tensor.ndim != 5:
            raise ValueError(f"Expected tensor with 5 dims, got {frames_tensor.ndim}")
        frames_tensor = frames_tensor.to(self.device)
        with torch.no_grad():
            embedding = self.model(frames_tensor).squeeze(0).cpu().numpy()
        return embedding

    def process_video_interval(
        self, video_path: str, start_frame: int, end_frame: int
    ) -> Optional[np.ndarray]:
        """
        Extract embedding for a specific time interval in a video.
        
        This is the main method used during phoneme analysis. Given a video
        and a frame range (corresponding to when a phoneme is pronounced),
        it extracts mouth patches from each frame and generates an embedding.
        
        Process:
        1. Open video and seek to start_frame
        2. Extract mouth patch from each frame in the interval
        3. Stack patches into a temporal sequence
        4. Pass through ResNet3D to get embedding
        
        Args:
            video_path: Path to video file
            start_frame: First frame of the interval
            end_frame: Last frame of the interval
            
        Returns:
            Embedding vector (128-D) or None if:
            - Video cannot be opened
            - Face not detected in enough frames
            - Less than min_frames valid patches extracted
        """
        import os
        from pathlib import Path

        path = Path(video_path)
        if not path.exists() or start_frame >= end_frame:
            return None

        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or end_frame
            start = max(0, start_frame)
            end = min(end_frame, total_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            patches: List[np.ndarray] = []
            frame_idx = start
            while frame_idx < end:
                ret, frame = cap.read()
                if not ret:
                    break
                patch = self.extract_mouth_patch(frame)
                if patch is not None:
                    patches.append(patch)
                frame_idx += 1
            cap.release()

            if len(patches) < self.min_frames:
                return None

            frames_np = np.stack(patches).astype(np.float32)
            frames_tensor = torch.from_numpy(frames_np).unsqueeze(0).unsqueeze(0)
            return self.get_embedding(frames_tensor)
        except Exception:
            return None
