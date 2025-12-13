import os
import json
import numpy as np
import argparse
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_preparation.config import DATASET_OUTPUT, GOLD_STORE_DIR, DEVICE
from data_preparation.model import VideoEmbeddingAdapter
import torch
# --- CONFIGURAZIONE ---


def cross_identity_test(target_speaker="s1", adapter_path=None):
    print(f"--- TEST DI SICUREZZA: {target_speaker} vs TUTTI ---")
    
    # 1. Carichiamo il profilo del Target
    gold_path = os.path.join(GOLD_STORE_DIR, f"{target_speaker}_gold_dictionary.json")
    if not os.path.exists(gold_path):
        print(f"Errore: Manca il Gold Dictionary per {target_speaker}!")
        print(f"Hai lanciato lo step 4 per {target_speaker}?")
        return

    with open(gold_path, "r") as f:
        gold_dict = json.load(f)
    
    # Ottimizzazione lookup
    gold_vectors = {ph: np.array(data["vector"]) for ph, data in gold_dict.items()}
    print(f"Target {target_speaker} caricato: {len(gold_vectors)} fonemi.")

    # 2. Troviamo tutti gli altri speaker (Impostori)
    # Cerca cartelle video_embeddings_s*
    embedding_folders = [d for d in os.listdir(DATASET_OUTPUT) 
                         if os.path.isdir(os.path.join(DATASET_OUTPUT, d)) 
                         and d.startswith("video_embeddings_s")]
    
    impostor_scores = {} # { "s2": [score1, score2...], "s3": [...] }
    
    print(f"Confronto con {len(embedding_folders)} potenziali impostori...")

    # Load Adapter
    adapter = None
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        adapter = VideoEmbeddingAdapter().to(DEVICE)
        adapter.load_state_dict(torch.load(adapter_path, map_location=DEVICE))
        adapter.eval()

    for folder in tqdm(embedding_folders):
        speaker_id = folder.split("_")[-1] # video_embeddings_s2 -> s2
        
        # Saltiamo se stesso (o lo usiamo come baseline self-test)
        is_self = (speaker_id == target_speaker)
        
        input_folder = os.path.join(DATASET_OUTPUT, folder)
        files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]
        
        # Prendiamo un campione casuale di 20 video per speaker per velocità
        if len(files) > 20:
            np.random.shuffle(files)
            files = files[:20]
            
        scores_for_this_speaker = []
        
        for filename in files:
            path = os.path.join(input_folder, filename)
            try:
                data = np.load(path)
            except: continue
            
            video_scores = []
            for phoneme, embeddings in data.items():
                if phoneme in gold_vectors:
                    target_vec = gold_vectors[phoneme]
                    
                    # Apply adapter if present
                    if adapter:
                        tensor_in = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
                        with torch.no_grad():
                            embeddings = adapter(tensor_in).cpu().numpy()

                    for vec in embeddings:
                        sim = 1 - cosine(vec, target_vec)
                        video_scores.append(sim)
            
            if video_scores:
                scores_for_this_speaker.append(np.mean(video_scores))
        
        if scores_for_this_speaker:
            avg = np.mean(scores_for_this_speaker)
            impostor_scores[speaker_id] = scores_for_this_speaker
            # print(f"  -> {speaker_id}: {avg:.4f}")

    # --- RISULTATI ---
    print("\n" + "="*40)
    print(f"RISULTATI ONE-VS-ALL ({target_speaker})")
    print("="*40)
    
    # Separiamo Self da Impostors
    self_scores = impostor_scores.get(target_speaker, [])
    others_flat = []
    
    if self_scores:
        print(f"✅ SELF-TEST ({target_speaker}): {np.mean(self_scores):.4f}")
    else:
        print(f"⚠️ SELF-TEST ({target_speaker}): N/A (Nessun video trovato?)")

    print("-" * 40)
    print(f"{'SPEAKER':<10} | {'SCORE':<10} | {'STATUS'}")
    print("-" * 40)
    
    sorted_speakers = sorted(impostor_scores.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    
    for spk in sorted_speakers:
        if spk == target_speaker: continue
        
        scores = impostor_scores[spk]
        avg = np.mean(scores)
        others_flat.extend(scores)
        
        status = "✅ BLOCKED" if avg < 0.75 else "❌ PASSED"
        print(f"{spk:<10} | {avg:.4f}     | {status}")

    print("-" * 40)
    
    if others_flat:
        avg_impostor = np.mean(others_flat)
        print(f"MEDIA IMPOSTORI: {avg_impostor:.4f}")
        if self_scores:
            print(f"GAP SICUREZZA:   {np.mean(self_scores) - avg_impostor:.4f}")

    # --- PLOT ---
    plt.figure(figsize=(12, 6))
    
    # Plot Impostors
    for spk, scores in impostor_scores.items():
        if spk == target_speaker: continue
        sns.kdeplot(scores, label=spk, fill=True, alpha=0.1)
        
    # Plot Target (più evidente)
    if self_scores:
        sns.kdeplot(self_scores, label=f"TARGET ({target_speaker})", color="blue", linewidth=3, fill=False)
        
    plt.title(f"Distribuzione Score: {target_speaker} vs Altri")
    plt.xlabel("Cosine Similarity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("cross_check_result.png")
    print("\nGrafico salvato in cross_check_result.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="s1", help="Speaker ID da proteggere (es. s1)")
    parser.add_argument("--adapter", type=str, help="Path to trained adapter model (.pth)")
    args = parser.parse_args()
    
    cross_identity_test(args.target, args.adapter)