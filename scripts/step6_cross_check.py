import os
import json
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAZIONE ---
GOLD_DICT_PATH = "dataset/output/gold_store/s1_gold_dictionary.json"     # La verità (Speaker 1)
ATTACKER_FOLDER = "dataset/output/video_embeddings_s2"      # L'impostore (Speaker 2)

def cross_identity_test():
    print(f"--- TEST DI SICUREZZA: S2 vs S1 GOLD ---")
    
    # 1. Carichiamo il profilo di S1
    if not os.path.exists(GOLD_DICT_PATH):
        print("Errore: Manca il Gold Dictionary di S1!")
        return

    with open(GOLD_DICT_PATH, "r") as f:
        gold_dict = json.load(f)
    
    # Ottimizzazione lookup
    gold_vectors = {ph: np.array(data["vector"]) for ph, data in gold_dict.items()}

    # 2. Carichiamo i video di S2
    if not os.path.exists(ATTACKER_FOLDER):
        print("Errore: Manca la cartella embeddings di S2. Hai lanciato lo step 2?")
        return

    files = [f for f in os.listdir(ATTACKER_FOLDER) if f.endswith('.npz')]
    print(f"Analizzo {len(files)} video dell'impostore...")
    
    # Ne testiamo 50 a caso per avere statistica solida
    np.random.shuffle(files)
    test_files = files[:50] 
    
    all_scores = []

    for filename in test_files:
        path = os.path.join(ATTACKER_FOLDER, filename)
        try:
            data = np.load(path)
        except: continue
        
        video_scores = []
        for phoneme, embeddings in data.items():
            # Confrontiamo solo i fonemi che esistono nel dizionario di S1
            if phoneme in gold_vectors:
                target_vec = gold_vectors[phoneme]
                for vec in embeddings:
                    # Similarity
                    sim = 1 - cosine(vec, target_vec)
                    video_scores.append(sim)

        if video_scores:
            avg_video = np.mean(video_scores)
            all_scores.append(avg_video)

    if not all_scores:
        print("Nessun dato valido trovato.")
        return

    # --- RISULTATI ---
    mean_score = np.mean(all_scores)
    max_score = np.max(all_scores)
    min_score = np.min(all_scores)
    
    print("\n" + "="*40)
    print(f"RISULTATI CROSS-TEST (S1 vs S2)")
    print("="*40)
    print(f"BASELINE S1 (Target): ~0.9485")
    print(f"IMPOSTORE S2 (Score): {mean_score:.4f}")
    print(f"GAP DI SICUREZZA:     {0.9485 - mean_score:.4f}")
    print("-" * 40)
    
    # Interpretazione
    if mean_score < 0.80:
        print("✅ SUCCESSO: Identità diverse distinte correttamente.")
        print("   Il sistema rileva che S2 muove la bocca diversamente da S1.")
    elif mean_score < 0.90:
        print("⚠️ ZONA GRIGIA: C'è differenza, ma non enorme.")
        print("   Potrebbe servire addestramento per accentuare le differenze.")
    else:
        print("❌ FALLIMENTO: Il sistema non distingue S1 da S2.")
        print("   La ResNet sta guardando caratteristiche generiche (bocca aperta/chiusa)")
        print("   invece dell'identità specifica del movimento.")

    # Plot della distribuzione (Opzionale)
    plt.figure(figsize=(10, 6))
    sns.histplot(all_scores, kde=True, color="red", label="Impostore (S2)")
    # Linea fittizia per S1 (basata sul tuo test precedente)
    plt.axvline(x=0.9485, color='blue', linestyle='--', label="Target Reale (S1)")
    plt.title("Distribuzione Score: Reale vs Impostore")
    plt.legend()
    plt.xlabel("Cosine Similarity")
    plt.savefig("cross_identity_result.png")
    print("Grafico salvato in cross_identity_result.png")

if __name__ == "__main__":
    cross_identity_test()