import numpy as np

def analizza_struttura_npz(percorso_file):
    """
    Carica un file .npz e ne mostra il contenuto in modo leggibile.
    """
    try:
        # Carichiamo il file .npz
        # Usiamo 'with' per assicurarci che il file venga chiuso correttamente dopo la lettura
        with np.load(percorso_file) as dati:
            
            # Estraiamo la lista dei nomi degli array presenti
            nomi_array = dati.files
            
            if not nomi_array:
                print("Il file è vuoto.")
                return

            print(f"--- Struttura del file: {percorso_file} ---")
            print(f"Trovati {len(nomi_array)} array.\n")

            # Cicliamo attraverso ogni array per vederne le proprietà
            for nome in nomi_array:
                array_corrente = dati[nome]
                
                print(f"Nome Array:  {nome}")
                print(f"Dimensioni:   {array_corrente.shape}")
                print(f"Tipo Dati:    {array_corrente.dtype}")
                print("-" * 30)

    except FileNotFoundError:
        print(f"Errore: Il file '{percorso_file}' non è stato trovato.")
    except Exception as e:
        print(f"Si è verificato un errore imprevisto: {e}")

# --- ESEMPIO DI UTILIZZO ---
# Sostituisci 'il_tuo_file.npz' con il nome reale del tuo file
analizza_struttura_npz('bbaf2n.npz')