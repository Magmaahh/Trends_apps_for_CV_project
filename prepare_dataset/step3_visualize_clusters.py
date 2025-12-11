import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# --- CONFIGURAZIONE ---
INPUT_FOLDER = "../dataset/output/video_embeddings_real"
# Scegliamo alcuni fonemi molto diversi visivamente per vedere se li distingue
# B/M/P = Labbra chiuse
# AA/AE = Bocca molto aperta
# UW = Labbra a cerchio (bacio)
TARGET_PHONEMES = ["B", "M", "AA1", "UW1", "F", "IY1"]

def visualize_embeddings():
    print("Caricamento dati...")
    
    data_points = []
    labels = []
    
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith('.npz')]
    
    # Raccogliamo un po' di dati (non troppi sennò t-SNE è lento)
    max_points_per_phoneme = 200 
    counts = {p: 0 for p in TARGET_PHONEMES}
    
    for f in files:
        try:
            data = np.load(os.path.join(INPUT_FOLDER, f))
            # Itera sui fonemi che ci interessano
            for phoneme in TARGET_PHONEMES:
                if phoneme in data and counts[phoneme] < max_points_per_phoneme:
                    # I vettori salvati sono matrici (N, 128), ne prendiamo uno a caso o tutti
                    vecs = data[phoneme]
                    if len(vecs) > 0:
                        # Prendiamo solo il primo vettore dell'istanza per semplicità
                        data_points.append(vecs[0]) 
                        labels.append(phoneme)
                        counts[phoneme] += 1
        except Exception as e:
            continue
            
        # Stop se abbiamo abbastanza dati per tutti
        if all(c >= max_points_per_phoneme for c in counts.values()):
            break

    print(f"Punti raccolti: {len(data_points)}")
    if len(data_points) < 50:
        print("Troppi pochi dati per visualizzare. Controlla i nomi dei fonemi nel print dello step precedente.")
        return

    # Convertiamo in numpy array
    X = np.array(data_points)
    y = np.array(labels)

    # --- t-SNE ---
    print("Calcolo t-SNE (riduzione dimensionale)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    # --- PLOT ---
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=y, palette="deep", s=60, alpha=0.8)
    plt.title("Spazio Latente Visivo: I fonemi sono separati?", fontsize=16)
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend(title="Fonema")
    plt.grid(True, alpha=0.3)
    
    output_img = "../dataset/output/visualization_clusters.png"
    plt.savefig(output_img)
    print(f"Grafico salvato in: {output_img}")
    plt.show()

if __name__ == "__main__":
    visualize_embeddings()