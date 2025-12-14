from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from torchvision.models.video import r3d_18, R3D_18_Weights

# --- NEURAL NETWORKS ---

class MouthEmbeddingResNet3D(nn.Module):
    """
    3D ResNet adapted for grayscale mouth video clips.
    Outputs a fixed-dimensional embedding (default 128).
    """
    def __init__(self, embedding_dim=128):
        super().__init__()

        # Load pretrained ResNet3D-18 model
        self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)

        # Modify first convolutional layer to accept grayscale input (1 channel instead of 3)
        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(
            1,  # Input: 1 channel (grayscale)
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        # Initialize new conv weights by averaging RGB weights
        with torch.no_grad():
            new_conv.weight[:] = torch.sum(old_conv.weight, dim=1, keepdim=True)

        self.backbone.stem[0] = new_conv
        
        # Replace final layer to output embedding_dim features
        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            embedding_dim
        )

    def forward(self, x):
        return self.backbone(x)

class VideoEmbeddingAdapter(nn.Module):
    """
    Simple MLP adapter to project embeddings into a metric space
    optimized for speaker discrimination (Triplet Loss).
    """
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)


# --- FEATURE EXTRACTOR ---

class VideoFeatureExtractor:
    """
    Extract visual embeddings from mouth regions during speech.
    Uses MediaPipe Face Mesh and MouthEmbeddingResNet3D.
    """

    def __init__(
        self,
        device: str = "auto",
        img_size: int = 112,
        embedding_dim: int = 128,
        min_frames: int = 4,
    ) -> None:
        self.device = self._resolve_device(device)
        self.img_size = img_size
        self.min_frames = min_frames
        
        # Load Model
        self.model = MouthEmbeddingResNet3D(embedding_dim=embedding_dim).to(self.device)
        self.model.eval()

        # MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )
        # Key lip landmark indices
        self.lip_landmarks: List[int] = [61, 291, 0, 17, 78, 308, 13, 14]

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def extract_mouth_patch(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract and normalize mouth region from a single video frame.
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
        """
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
