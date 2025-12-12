import os
import shutil
from pathlib import Path

# --- CONFIGURAZIONE ---
DATASET_INIT = "dataset/init"
DATASET_OUTPUT = "dataset/output"

# Grid Corpus Grammar Mapping
GRID_GRAMMAR = {
    0: {'b': 'bin', 'l': 'lay', 'p': 'place', 's': 'set'},
    1: {'b': 'blue', 'g': 'green', 'r': 'red', 'w': 'white'},
    2: {'a': 'at', 'b': 'by', 'i': 'in', 'w': 'with'},
    3: {x: x.upper() for x in "abcdefghijklmnopqrstuvwxyz"}, # Letters are just letters
    4: {'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', 
        '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine', 'z': 'zero'},
    5: {'a': 'again', 'n': 'now', 'p': 'please', 's': 'soon'}
}

def decode_grid_filename(filename_stem):
    """
    Decodes a 6-character Grid Corpus filename into a sentence.
    Example: bbaf2n -> BIN BLUE AT F TWO NOW
    """
    if len(filename_stem) != 6:
        return None
    
    words = []
    chars = list(filename_stem.lower())
    
    try:
        for i, char in enumerate(chars):
            if i in GRID_GRAMMAR:
                mapping = GRID_GRAMMAR[i]
                if char in mapping:
                    words.append(mapping[char])
                else:
                    # Fallback for letters if not explicitly in map (though we added a-z)
                    if i == 3: 
                        words.append(char.upper())
                    else:
                        return None # Invalid char for position
            else:
                return None
    except:
        return None
        
    return " ".join(words).upper()

def prepare_single_speaker(speaker_id):
    print(f"\n--- Processing Speaker: {speaker_id} ---")
    
    # Costruisci i percorsi dinamici
    align_source = os.path.join(DATASET_OUTPUT, "alignments", speaker_id)
    audio_source = os.path.join(DATASET_INIT, "audio_25k", speaker_id)
    output_dir = os.path.join(DATASET_OUTPUT, f"mfa_workspace_{speaker_id}")

    # 1. Verifica sorgente audio
    if not os.path.exists(audio_source):
        print(f"SKIP: Cartella audio non trovata: {audio_source}")
        return

    # Pulisci e ricrea la cartella di output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    print(f"Input Audio:      {audio_source}")
    
    # 2. Trova tutti i file .wav
    wav_files = list(Path(audio_source).glob("*.wav"))
    print(f"Trovati {len(wav_files)} file audio.")
    
    if len(wav_files) == 0:
        print("Nessun file .wav trovato.")
        return

    # 3. Processa ogni file audio
    count_align = 0
    count_decode = 0
    
    for wav_file in wav_files:
        file_id = wav_file.stem
        full_text = None
        
        # A. Cerca allineamento esistente
        align_file = os.path.join(align_source, f"{file_id}.align")
        if os.path.exists(align_file):
            try:
                words = []
                with open(align_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            word = parts[2]
                            if word not in ["sil", "sp"]:
                                words.append(word)
                if words:
                    full_text = " ".join(words).upper()
                    count_align += 1
            except Exception:
                pass
        
        # B. Se non trovato o fallito, usa decoding filename
        if not full_text:
            full_text = decode_grid_filename(file_id)
            if full_text:
                count_decode += 1
        
        # C. Scrivi output se abbiamo testo
        if full_text:
            # Scrivi .lab
            lab_path = os.path.join(output_dir, f"{file_id}.lab")
            with open(lab_path, "w") as f:
                f.write(full_text)
                
            # Copia .wav
            shutil.copy(wav_file, os.path.join(output_dir, f"{file_id}.wav"))
            
    print(f"Completato {speaker_id}: {count_align + count_decode} coppie create ({count_align} da align, {count_decode} da decode)")

def main():
    audio_root = os.path.join(DATASET_INIT, "audio_25k")
    if not os.path.exists(audio_root):
        print(f"ERRORE: Cartella audio root non trovata: {audio_root}")
        return

    # Trova tutte le cartelle che iniziano con 's'
    speakers = [d for d in os.listdir(audio_root) if os.path.isdir(os.path.join(audio_root, d)) and d.startswith('s')]
    speakers.sort(key=lambda x: int(x[1:]) if x[1:].isdigit() else 999)
    
    print(f"Trovati {len(speakers)} speaker: {speakers}")
    
    for spk in speakers:
        prepare_single_speaker(spk)
        
    print("\n--- TUTTI GLI SPEAKER PROCESSATI ---")
    print("Ora puoi lanciare MFA per ogni speaker.")

if __name__ == "__main__":
    main()