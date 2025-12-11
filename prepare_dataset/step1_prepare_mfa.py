import os
import shutil
from pathlib import Path

# --- CONFIGURAZIONE ---
DATASET_INIT = "dataset/init"
DATASET_OUTPUT = "dataset/output"

def prepare_single_speaker(speaker_id):
    print(f"\n--- Processing Speaker: {speaker_id} ---")
    
    # Costruisci i percorsi dinamici
    align_source = os.path.join(DATASET_OUTPUT, "alignments", speaker_id)
    audio_source = os.path.join(DATASET_INIT, "audio_25k", speaker_id)
    output_dir = os.path.join(DATASET_OUTPUT, f"mfa_workspace_{speaker_id}")

    # 1. Verifica che i percorsi esistano
    if not os.path.exists(align_source):
        print(f"SKIP: Cartella allineamenti non trovata: {align_source}")
        return
    if not os.path.exists(audio_source):
        print(f"SKIP: Cartella audio non trovata: {audio_source}")
        return

    # Pulisci e ricrea la cartella di output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Input Alignments: {align_source}")
    print(f"Input Audio:      {audio_source}")

    # 2. Trova tutti i file .align
    align_files = list(Path(align_source).glob("*.align"))
    if len(align_files) == 0:
        align_files = list(Path(align_source).rglob("*.align"))

    print(f"Trovati {len(align_files)} file di allineamento.")

    if len(align_files) == 0:
        print("Nessun file .align trovato.")
        return

    count = 0
    for align_file in align_files:
        file_id = align_file.stem # es. "bbaf2n"
        
        # 3. Leggi e pulisci il testo
        words = []
        try:
            with open(align_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3: # start stop word
                        word = parts[2]
                        if word not in ["sil", "sp"]: # Ignora silenzi
                            words.append(word)
        except Exception as e:
            print(f"Errore leggendo {file_id}: {e}")
            continue
        
        full_text = " ".join(words).upper()
        
        # 4. Scrivi il file .lab
        lab_path = os.path.join(output_dir, f"{file_id}.lab")
        with open(lab_path, "w") as f:
            f.write(full_text)
            
        # 5. Copia l'audio corrispondente
        wav_src = os.path.join(audio_source, f"{file_id}.wav")
        
        if os.path.exists(wav_src):
            shutil.copy(wav_src, os.path.join(output_dir, f"{file_id}.wav"))
            count += 1
        else:
            if count == 0: 
                print(f"ATTENZIONE: Non trovo audio per {file_id}")

    print(f"Completato {speaker_id}: {count} coppie create in {output_dir}")


def main():
    audio_root = os.path.join(DATASET_INIT, "audio_25k")
    if not os.path.exists(audio_root):
        print(f"ERRORE: Cartella audio root non trovata: {audio_root}")
        return

    # Trova tutte le cartelle che iniziano con 's' (s1, s2, ..., s34)
    speakers = [d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d)) and d.startswith('s')]
    
    # Ordina numericamente (s1, s2, s10 invece di s1, s10, s2)
    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    
    print(f"Trovati {len(speakers)} speaker: {speakers}")
    
    for spk in speakers:
        prepare_single_speaker(spk)
        
    print("\n--- TUTTI GLI SPEAKER PROCESSATI ---")
    print("Ora puoi lanciare MFA per ogni speaker.")

if __name__ == "__main__":
    main()