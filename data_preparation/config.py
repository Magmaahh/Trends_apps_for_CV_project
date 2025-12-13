import torch

# Dataset paths
DATASET_INIT = "dataset/init"
DATASET_OUTPUT = "dataset/output"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image/model parameters
IMG_SIZE = 112
EMBEDDING_DIM = 128
MIN_PHONEME_SAMPLES = 50

# Phonemes for visualization/analysis
TARGET_PHONEMES = ["B", "M", "AA1", "UW1", "F", "IY1"]

# GRID corpus grammar
GRID_GRAMMAR = {
    0: {'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set'},
    1: {'b': 'blue', 'g': 'green', 'r': 'red', 'w': 'white'},
    2: {'a': 'at', 'b': 'by', 'i': 'in', 'w': 'with'},
    3: {c: c.upper() for c in "abcdefghijklmnopqrstuvwxyz"},
    4: {
        '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight',
        '9': 'nine', 'z': 'zero'
    },
    5: {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}
}