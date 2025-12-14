import os
import shutil
import numpy as np
import cv2
import mediapipe as mp
import torch
import json
from pathlib import Path
from tqdm import tqdm

from src.video.common.utils import (
    DATASET_INIT, DATASET_OUTPUT, GOLD_STORE_DIR, DEVICE, EMBEDDING_DIM, IMG_SIZE, MIN_PHONEME_SAMPLES,
    decode_grid_filename, parse_textgrid
)
from src.video.common.model import MouthEmbeddingResNet3D, VideoEmbeddingAdapter

def prepare_speaker_for_mfa(speaker_id):
    """
    Creates .lab and .wav pairs for MFA alignment (for Dataset).
    """
    audio_dir = os.path.join(DATASET_INIT, "audio_25k", speaker_id)
    align_dir = os.path.join(DATASET_OUTPUT, "alignments", speaker_id)
    output_dir = os.path.join(DATASET_OUTPUT, f"mfa_workspace_{speaker_id}")

    if not os.path.exists(audio_dir):
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    wav_files = Path(audio_dir).glob("*.wav")

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
        
        if text is None:
            text = decode_grid_filename(file_id)

        if text:
            with open(os.path.join(output_dir, f"{file_id}.lab"), "w") as f:
                f.write(text)
            shutil.copy(wav, os.path.join(output_dir, f"{file_id}.wav"))


def extract_embeddings_for_speaker(speaker_id):
    """
    Extracts visual phoneme embeddings for one speaker (Dataset).
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

    existing_npz = list(Path(out_dir).glob("*.npz"))
    if len(existing_npz) >= 200:
        print(f"SKIP: {speaker_id} già processato ({len(existing_npz)} embeddings trovati).")
        return

    tg_files = [f for f in os.listdir(tg_dir) if f.endswith(".TextGrid")]
    tg_files.sort()
    tg_files = tg_files[:200]

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

        phonemes = parse_textgrid(os.path.join(tg_dir, tg_file))
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
                phoneme_embeddings.setdefault(p["phoneme"], []).append(emb)

        if phoneme_embeddings:
            np.savez_compressed(
                os.path.join(out_dir, f"{file_id}.npz"),
                **{k: np.array(v) for k, v in phoneme_embeddings.items()}
            )

def build_gold_dictionary_for_speaker(speaker_id, adapter_path=None):
    """
    Builds a mean gold embedding per phoneme for one speaker.
    """
    emb_dir = os.path.join(DATASET_OUTPUT, f"video_embeddings_{speaker_id}")
    gold_dir = GOLD_STORE_DIR
    os.makedirs(gold_dir, exist_ok=True)

    accumulator = {}
    
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
