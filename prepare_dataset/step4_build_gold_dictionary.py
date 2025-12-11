import os
import numpy as np
import json
from tqdm import tqdm

# CONFIGURAZIONE
DATASET_OUTPUT = "dataset/output"
GOLD_STORE_DIR = os.path.join(DATASET_OUTPUT, "gold_store")

def build_gold_dictionary_for_speaker(speaker_id):
    input_folder = os.path.join(DATASET_OUTPUT, f"video_embeddings_{speaker_id}")
    output_file = os.path.join(GOLD_STORE_DIR, f"{speaker_id}_gold_dictionary.json")
    
    print(f"\n--- Building Gold Dictionary for: {speaker_id} ---")
    
    if not os.path.exists(input_folder):
        print(f"SKIP: Cartella embeddings non trovata: {input_folder}")
        return

    # Accumuliamo tutti i vettori per ogni fonema
    accumulator = {}
    
    files = [f for f in os.listdir(input_folder) if f.endswith('.npz')]
    print(f"Lettura di {len(files)} file di embedding...")
    
    if len(files) == 0:
        print("Nessun file .npz trovato.")
        return

    for f in tqdm(files, desc=f"Reading {speaker_id}"):
        try:
            path = os.path.join(input_folder, f)
            data = np.load(path)
            
            for phoneme, embeddings in data.items():
                if phoneme not in accumulator:
                    accumulator[phoneme] = []
                for vec in embeddings:
                    accumulator[phoneme].append(vec)
        except Exception as e:
            print(f"Errore file {f}: {e}")

    # Calcoliamo la media (il vettore "Gold")
    gold_dict = {}
    valid_phonemes = 0
    
    print("Calcolo medie vettoriali...")
    for phoneme, vec_list in accumulator.items():
        # Scartiamo fonemi rari (meno di 50 esempi in tutto il dataset)
        if len(vec_list) < 50:
            continue
            
        all_vecs = np.array(vec_list)
        mean_vec = np.mean(all_vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        
        gold_dict[phoneme] = {
            "vector": mean_vec.tolist(),
            "count": len(vec_list)
        }
        valid_phonemes += 1

    # Salvataggio
    os.makedirs(GOLD_STORE_DIR, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(gold_dict, f, indent=4)
        
    print(f"✅ CREATO: {output_file}")
    print(f"Contiene {valid_phonemes} fonemi unici.")

def main():
    if not os.path.exists(DATASET_OUTPUT):
        print(f"ERRORE: {DATASET_OUTPUT} non esiste.")
        return

    # Cerca cartelle che iniziano con video_embeddings_s*
    embedding_folders = [d for d in os.listdir(DATASET_OUTPUT) 
                         if os.path.isdir(os.path.join(DATASET_OUTPUT, d)) 
                         and d.startswith("video_embeddings_s")]
    
    # Estrai gli ID speaker (es. video_embeddings_s1 -> s1)
    speakers = []
    for folder in embedding_folders:
        parts = folder.split("_")
        if len(parts) >= 3:
            spk = parts[-1] # s1, s2...
            speakers.append(spk)
    
    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    
    print(f"Trovati {len(speakers)} speaker con embeddings: {speakers}")
    
    for spk in speakers:
        build_gold_dictionary_for_speaker(spk)

if __name__ == "__main__":
    main()