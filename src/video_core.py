from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights, r3d_18


class VideoFeatureExtractor:
    """
    Universal video feature extractor for mouth embeddings.
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
        self.model = self._load_model(embedding_dim).to(self.device)
        self.model.eval()

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True
        )
        # Landmarks covering the mouth region for stable bounding boxes.
        self.lip_landmarks: List[int] = [61, 291, 0, 17, 78, 308, 13, 14]

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self, embedding_dim: int) -> nn.Module:
        backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        old_conv = backbone.stem[0]
        new_conv = nn.Conv3d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:] = torch.sum(old_conv.weight, dim=1, keepdim=True)
        backbone.stem[0] = new_conv
        backbone.fc = nn.Linear(backbone.fc.in_features, embedding_dim)
        return backbone

    def extract_mouth_patch(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the mouth region and return a normalized grayscale patch (img_size x img_size).
        Returns None when no face is detected.
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

        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))
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
        Run inference on a tensor shaped (1, 1, T, H, W) and return a 128D numpy vector.
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
        Complete pipeline: open video, crop mouth patches for the interval, and return the embedding.
        Returns None when frames are insufficient or no face is detected.
        """
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
