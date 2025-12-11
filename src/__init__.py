"""Core library for deepfake detection video processing."""

from .video_core import VideoFeatureExtractor
from . import utils

__all__ = ["VideoFeatureExtractor", "utils"]
