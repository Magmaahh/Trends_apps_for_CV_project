import os
import json
import numpy as np
from scipy.spatial.distance import cosine

# --- CONFIGURAZIONE ---
GOLD_DICT_PATH = "dataset/output/gold_store/s1_gold_dictionary.json"
EMBEDDINGS_FOLDER = "dataset/output/video_embeddings_real"

def calculate_similarity():
    # 1. Carica il Dizionario Gold
    with open(GOLD_DICT_PATH, "r") as f:
        gold_dict = json.load(f)
    
    # Convertiamo i vettori Gold in numpy array per velocità
    gold_vectors = {}
    for ph, data in gold_dict.items():
        gold_vectors[ph] = np.array(data["vector"])

    # 2. Prendi alcuni file a caso per il test
    files = [f for f in os.listdir(EMBEDDINGS_FOLDER) if f.endswith('.npz')]
    np.random.shuffle(files)
    test_files = files[:5] # Testiamo su 5 file

    print(f"--- TEST DI VALIDAZIONE SU {len(test_files)} VIDEO ---")
    
    overall_scores = []

    for filename in test_files:
        path = os.path.join(EMBEDDINGS_FOLDER, filename)
        data = np.load(path)
        
        scores = []
        details = []

        for phoneme, embeddings in data.items():
            # Saltiamo fonemi che non sono nel dizionario (es. troppo rari)
            if phoneme not in gold_vectors:
                continue
            
            target_vec = gold_vectors[phoneme]
            
            # Calcoliamo la similarità per ogni istanza di quel fonema nel video
            for vec in embeddings:
                # Cosine Similarity = 1 - Cosine Distance
                # Range: -1 (opposto) a 1 (identico)
                sim = 1 - cosine(vec, target_vec)
                scores.append(sim)
                
                # Salviamo dettagli per debug (solo i primi per non intasare)
                if len(details) < 3:
                    details.append(f"{phoneme}: {sim:.4f}")

        if not scores:
            print(f"File {filename}: Nessun fonema valido trovato.")
            continue

        avg_score = np.mean(scores)
        overall_scores.append(avg_score)
        
        print(f"\nVIDEO: {filename}")
        print(f"  -> Score Medio: {avg_score:.4f}")
        print(f"  -> Dettagli: {', '.join(details)}...")

    # 3. Verdetto Finale
    if not overall_scores: return
    
    grand_mean = np.mean(overall_scores)
    print("\n" + "="*40)
    print(f"SCORE GLOBALE MEDIO: {grand_mean:.4f}")
    print("="*40)
    
    print("\nINTERPRETAZIONE:")
    if grand_mean > 0.85:
        print("✅ ECCELLENTE. Il sistema è robustissimo.")
    elif grand_mean > 0.70:
        print("✅ BUONO. Funziona, ma c'è margine di miglioramento.")
    elif grand_mean > 0.50:
        print("⚠️ ACCETTABILE/MEDIOCRE. Distingue a malapena.")
    else:
        print("❌ FALLIMENTO. Il sistema sta tirando a indovinare (Random).")
        print("Soluzione necessaria: Fine-Tuning della ResNet con Triplet Loss.")

if __name__ == "__main__":
    calculate_similarity()