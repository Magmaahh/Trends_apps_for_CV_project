import os
from .config import DATASET_INIT, DATASET_OUTPUT

def load_speakers():
    """
    Returns a sorted list of speaker IDs found in the dataset.
    """
    audio_root = os.path.join(DATASET_INIT, "audio_25k")
    if not os.path.exists(audio_root):
        return []

    speakers = [
        d for d in os.listdir(audio_root)
        if os.path.isdir(os.path.join(audio_root, d)) and d.startswith("s")
    ]

    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    return speakers


def load_embedding_speakers():
    """
    Returns speaker IDs for which video embeddings exist.
    """
    folders = [
        d for d in os.listdir(DATASET_OUTPUT)
        if d.startswith("video_embeddings_s")
    ]

    speakers = [f.split("_")[-1] for f in folders]
    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    return speakers
