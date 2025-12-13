import os
import json
import shutil
import numpy as np
import cv2
import mediapipe as mp
import torch
from pathlib import Path
from tqdm import tqdm

from data_preparation.config import *
from data_preparation.model import MouthEmbeddingResNet3D, VideoEmbeddingAdapter

# GRID filename decoding
def decode_grid_filename(stem: str):
    """
    Decode a 6-character GRID filename into a sentence.
    """
    if len(stem) != 6:
        return None

    words = []
    for idx, char in enumerate(stem.lower()):
        mapping = GRID_GRAMMAR.get(idx)
        if mapping is None or char not in mapping:
            return None
        words.append(mapping[char])

    return " ".join(words).upper()

# MFA preparation
def prepare_speaker_for_mfa(speaker_id):
    """
    Creates .lab and .wav pairs for MFA alignment.
    """
    # Setup paths
    audio_dir = os.path.join(DATASET_INIT, "audio_25k", speaker_id)
    align_dir = os.path.join(DATASET_OUTPUT, "alignments", speaker_id)
    output_dir = os.path.join(DATASET_OUTPUT, f"mfa_workspace_{speaker_id}")

    if not os.path.exists(audio_dir):
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    wav_files = Path(audio_dir).glob("*.wav")

    # Process each wav file
    for wav in wav_files:
        text = None
        file_id = wav.stem

        align_path = os.path.join(align_dir, f"{file_id}.align")
        if os.path.exists(align_path):
            words = []
            with open(align_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3 and parts[2] not in {"sil", "sp"}:
                        words.append(parts[2])
            if words:
                text = " ".join(words).upper()
        
        # Fallback to GRID decoding
        if text is None:
            text = decode_grid_filename(file_id)

        # Write .lab and copy .wav if text is available
        if text:
            with open(os.path.join(output_dir, f"{file_id}.lab"), "w") as f:
                f.write(text)
            shutil.copy(wav, os.path.join(output_dir, f"{file_id}.wav"))

# TextGrid parsing
def parse_textgrid_intervals(path, tier="phones"):
    """
    Parses a Praat TextGrid and extracts phoneme intervals.
    """
    intervals = []
    with open(path, encoding="utf-8") as f:
        lines = [l.strip() for l in f]

    active = False
    interval = {}

    for line in lines:
        if f'name = "{tier}"' in line:
            active = True
            continue
        if active and line.startswith('name = "'):
            break

        if line.startswith("intervals ["):
            interval = {}

        if active:
            if line.startswith("xmin"):
                interval["start"] = float(line.split("=")[1])
            elif line.startswith("xmax"):
                interval["end"] = float(line.split("=")[1])
            elif line.startswith("text"):
                text = line.split("=")[1].strip().replace('"', "")
                if text not in {"", "sil", "sp", "<eps>"}:
                    interval["text"] = text
                    intervals.append(interval.copy())

    return intervals

# Embedding extraction
def extract_embeddings_for_speaker(speaker_id):
    """
    Extracts visual phoneme embeddings for one speaker.
    """
    model = MouthEmbeddingResNet3D(EMBEDDING_DIM).to(DEVICE).eval()
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    )

    tg_dir = os.path.join(DATASET_OUTPUT, f"mfa_output_phonemes_{speaker_id}")
    vid_dir = os.path.join(DATASET_INIT, speaker_id)
    out_dir = os.path.join(DATASET_OUTPUT, f"video_embeddings_{speaker_id}")

    if not os.path.exists(tg_dir) or not os.path.exists(vid_dir):
        return

    os.makedirs(out_dir, exist_ok=True)

    # Check if already processed (folder level check)
    existing_npz = list(Path(out_dir).glob("*.npz"))
    if len(existing_npz) >= 200:
        print(f"SKIP: {speaker_id} già processato ({len(existing_npz)} embeddings trovati).")
        return

    tg_files = [f for f in os.listdir(tg_dir) if f.endswith(".TextGrid")]
    tg_files.sort()
    tg_files = tg_files[:200] # LIMIT TO 200 VIDEOS

    for tg_file in tqdm(tg_files, desc=speaker_id):

        file_id = tg_file.replace(".TextGrid", "")
        video_path = None
        for ext in (".mpg", ".mp4"):
            candidate = os.path.join(vid_dir, file_id + ext)
            if os.path.exists(candidate):
                video_path = candidate
                break
        if video_path is None:
            continue

        phonemes = parse_textgrid_intervals(os.path.join(tg_dir, tg_file))
        if not phonemes:
            continue

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        phoneme_embeddings = {}

        for p in phonemes:
            start = int(p["start"] * fps)
            end = int(p["end"] * fps)
            clip = frames[start:end]

            crops = []
            for frame in clip:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                if not res.multi_face_landmarks:
                    continue

                lm = res.multi_face_landmarks[0].landmark
                h, w, _ = frame.shape
                xs = [lm[i].x * w for i in [61, 291, 0, 17]]
                ys = [lm[i].y * h for i in [61, 291, 0, 17]]

                cx, cy = np.mean(xs), np.mean(ys)
                size = int((max(xs) - min(xs)) * 2)
                x1, y1 = max(0, int(cx - size)), max(0, int(cy - size))
                x2, y2 = min(w, int(cx + size)), min(h, int(cy + size))

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) / 255.0
                crops.append(crop)

            if len(crops) >= 4:
                x = torch.tensor(crops).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
                with torch.no_grad():
                    emb = model(x).cpu().numpy()[0]
                phoneme_embeddings.setdefault(p["text"], []).append(emb)

        if phoneme_embeddings:
            np.savez_compressed(
                os.path.join(out_dir, f"{file_id}.npz"),
                **{k: np.array(v) for k, v in phoneme_embeddings.items()}
            )

# Gold dictionary
def build_gold_dictionary_for_speaker(speaker_id, adapter_path=None):
    """
    Builds a mean gold embedding per phoneme for one speaker.
    """
    emb_dir = os.path.join(DATASET_OUTPUT, f"video_embeddings_{speaker_id}")
    gold_dir = os.path.join(DATASET_OUTPUT, "gold_store")
    os.makedirs(gold_dir, exist_ok=True)

    accumulator = {}
    
    # Load Adapter if provided
    adapter = None
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        adapter = VideoEmbeddingAdapter(EMBEDDING_DIM).to(DEVICE)
        adapter.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
        adapter.eval()

    for f in os.listdir(emb_dir):
        if not f.endswith(".npz"):
            continue
        data = np.load(os.path.join(emb_dir, f))
        for phoneme, vecs in data.items():
            # Apply adapter if present
            if adapter:
                tensor_in = torch.tensor(vecs, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    vecs = adapter(tensor_in).cpu().numpy()
            accumulator.setdefault(phoneme, []).extend(vecs)

    gold = {}
    for phoneme, vecs in accumulator.items():
        if len(vecs) < MIN_PHONEME_SAMPLES:
            continue
        mean = np.mean(vecs, axis=0)
        mean /= np.linalg.norm(mean)
        gold[phoneme] = {
            "vector": mean.tolist(),
            "count": len(vecs)
        }

    with open(os.path.join(gold_dir, f"{speaker_id}_gold_dictionary.json"), "w") as f:
        json.dump(gold, f, indent=4)