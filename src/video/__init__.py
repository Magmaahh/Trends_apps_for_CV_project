"""deepfake_cv package exposing core pipeline components."""

from .core import VideoFeatureExtractor
from .pipeline import VideoPipeline
from .matcher import IdentityMatcher

__all__ = ["VideoFeatureExtractor", "VideoPipeline", "IdentityMatcher"]
