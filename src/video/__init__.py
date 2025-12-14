"""
deepfake_cv package: Deepfake detection system based on visual phoneme analysis.

This package provides tools to detect deepfake videos by analyzing mouth movements
during speech and comparing them against a known speaker's visual phoneme patterns.

Main Components:
- VideoFeatureExtractor: Extracts visual embeddings from mouth regions during phoneme pronunciation
- VideoPipeline: End-to-end pipeline for processing videos (transcription -> alignment -> embedding)
- IdentityMatcher: Compares extracted embeddings against a gold-standard identity profile

Workflow:
1. Extract a "voice profile" from authentic videos of a person (gold standard)
2. Process a test video to extract its visual phoneme embeddings
3. Compare the test embeddings against the gold standard to detect anomalies
4. Videos with low similarity scores are flagged as potential deepfakes
"""

from .core import VideoFeatureExtractor
from .pipeline import VideoPipeline
from .core import VideoFeatureExtractor
from .pipeline import VideoPipeline

__all__ = ["VideoFeatureExtractor", "VideoPipeline"]
