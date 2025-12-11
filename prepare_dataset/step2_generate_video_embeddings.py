import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
from torchvision.models.video import r3d_18
from tqdm import tqdm

# --- CONFIGURAZIONE ---
DATASET_INIT = "dataset/init"
DATASET_OUTPUT = "dataset/output"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 112

# --- PARSER TEXTGRID ARTIGIANALE ---
def parse_textgrid_intervals(tg_path, tier_name="phones"):
    """Legge il file TextGrid e restituisce lista di {text, start, end}"""
    intervals = []
    try:
        with open(tg_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]
            
        in_target_tier = False
        current_interval = {}
        
        for i, line in enumerate(lines):
            if f'name = "{tier_name}"' in line:
                in_target_tier = True
                continue
            
            if in_target_tier and line.startswith('name = "'):
                break
                
            if in_target_tier and line.startswith('intervals ['):
                current_interval = {}
            
            if in_target_tier:
                if line.startswith('xmin = '):
                    current_interval['start'] = float(line.split('=')[1].strip())
                elif line.startswith('xmax = '):
                    current_interval['end'] = float(line.split('=')[1].strip())
                elif line.startswith('text = '):
                    text = line.split('=')[1].strip().replace('"', '')
                    current_interval['text'] = text
                    
                    if 'start' in current_interval and 'end' in current_interval:
                        if text not in ["", "sil", "sp", "<eps>"]:
                            intervals.append(current_interval.copy())
    except Exception as e:
        print(f"Errore parsing TextGrid {tg_path}: {e}")
        
    return intervals

# --- MODELLO RESNET3D ---
class MouthEmbeddingResNet3D(nn.Module):
    def __init__(self, embedding_dim=128):
        super(MouthEmbeddingResNet3D, self).__init__()
        self.backbone = r3d_18(weights='R3D_18_Weights.DEFAULT')
        old_conv = self.backbone.stem[0]
        new_conv = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), 
                             padding=(1, 3, 3), bias=False)
        with torch.no_grad():
            new_conv.weight[:] = torch.sum(old_conv.weight, dim=1, keepdim=True)
        self.backbone.stem[0] = new_conv
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, embedding_dim)
        
    def forward(self, x):
        return self.backbone(x)

# --- INIZIALIZZAZIONE GLOBALE ---
print(f"Caricamento modello su {DEVICE}...")
model = MouthEmbeddingResNet3D(embedding_dim=128).to(DEVICE)
model.eval()

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True
)

def process_single_speaker(speaker_id):
    print(f"\n--- Processing Speaker: {speaker_id} ---")
    
    textgrid_folder = os.path.join(DATASET_OUTPUT, f"mfa_output_phonemes_{speaker_id}")
    video_folder = os.path.join(DATASET_INIT, speaker_id)
    output_folder = os.path.join(DATASET_OUTPUT, f"video_embeddings_{speaker_id}")
    
    if not os.path.exists(textgrid_folder):
        print(f"SKIP: Nessun output MFA trovato per {speaker_id} in {textgrid_folder}")
        return
    
    if not os.path.exists(video_folder):
        print(f"SKIP: Nessuna cartella video trovata per {speaker_id} in {video_folder}")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    # Lista dei file TextGrid
    tg_files = [f for f in os.listdir(textgrid_folder) if f.endswith('.TextGrid')]
    print(f"Trovati {len(tg_files)} allineamenti. Inizio elaborazione...")

    for tg_file in tqdm(tg_files, desc=f"Speaker {speaker_id}"):
        file_id = os.path.splitext(tg_file)[0] # es: bbaf2n
        tg_path = os.path.join(textgrid_folder, tg_file)
        
        # 1. Trova il video corrispondente
        video_path = os.path.join(video_folder, f"{file_id}.mpg")
        if not os.path.exists(video_path):
            video_path = os.path.join(video_folder, f"{file_id}.mp4")
            if not os.path.exists(video_path):
                continue

        # 2. Leggi i fonemi dal TextGrid
        phonemes = parse_textgrid_intervals(tg_path, tier_name="phones")
        if not phonemes: continue

        # 3. Carica il video in memoria
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 25.0 
        
        frames_buffer = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames_buffer.append(frame)
        cap.release()
        
        if not frames_buffer: continue

        # 4. Estrai embedding per ogni fonema
        phoneme_embeddings = {}
        
        for p in phonemes:
            ph_label = p['text']
            start_f = int(p['start'] * fps)
            end_f = int(p['end'] * fps)
            
            duration = end_f - start_f
            if duration < 5:
                missing = 5 - duration
                start_f = max(0, start_f - missing // 2)
                end_f = min(len(frames_buffer), end_f + missing // 2 + 1)

            raw_frames = frames_buffer[start_f:end_f]
            if not raw_frames: continue
            
            processed_crops = []
            
            for frame in raw_frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = mp_face_mesh.process(rgb)
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    h, w, _ = frame.shape
                    
                    xs = [lm[i].x * w for i in [61, 291, 0, 17]]
                    ys = [lm[i].y * h for i in [61, 291, 0, 17]]
                    
                    cx, cy = np.mean(xs), np.mean(ys)
                    box_size = int((max(xs) - min(xs)) * 2.0)
                    half = box_size // 2
                    
                    x1, y1 = max(0, int(cx - half)), max(0, int(cy - half))
                    x2, y2 = min(w, int(cx + half)), min(h, int(cy + half))
                    
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    processed_crops.append(crop / 255.0)
            
            if len(processed_crops) >= 4:
                tensor_in = torch.FloatTensor(np.array(processed_crops)).unsqueeze(0).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    emb = model(tensor_in).squeeze(0).cpu().numpy()
                
                if ph_label not in phoneme_embeddings:
                    phoneme_embeddings[ph_label] = []
                phoneme_embeddings[ph_label].append(emb)

        # 5. Salva su disco
        if phoneme_embeddings:
            final_dict = {k: np.array(v) for k, v in phoneme_embeddings.items()}
            np.savez_compressed(os.path.join(output_folder, f"{file_id}.npz"), **final_dict)

    print(f"Finito {speaker_id}! Embedding salvati in {output_folder}")

def main():
    # Trova tutte le cartelle s* in dataset/init
    if not os.path.exists(DATASET_INIT):
        print(f"ERRORE: {DATASET_INIT} non esiste.")
        return

    speakers = [d for d in os.listdir(DATASET_INIT) if os.path.isdir(os.path.join(DATASET_INIT, d)) and d.startswith('s')]
    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    
    print(f"Trovati {len(speakers)} potenziali speaker: {speakers}")
    
    for spk in speakers:
        process_single_speaker(spk)

if __name__ == "__main__":
    process_video_phonemes()