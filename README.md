# Trends App - Computer Vision Module (Deepfake Detection)

Questo modulo implementa una pipeline di Computer Vision per rilevare deepfake basati su lip-sync, analizzando la coerenza tra i fonemi (audio) e i visemi (movimento delle labbra).

## Struttura del Progetto

Il progetto è organizzato per separare chiaramente dati, codice e script:

- **`dataset/`**: Contiene tutti i dati.
    - **`init/`**: I dataset originali (es. `s1`, `s2`) e i file raw.
    - **`output/`**: I file generati dalla pipeline (allineamenti MFA, embedding video, dizionari gold).
- **`src/`**: Il codice sorgente del modulo (logica core).
- **`prepare_dataset/`**: Script numerati (`step1` -> `step4`) per costruire il dataset e addestrare i riferimenti.
- **`scripts/`**: Tool per l'utilizzo finale (audit video, test di sicurezza).

---

## Setup Ambiente (Dual Environment)

Per garantire la compatibilità tra strumenti diversi (MFA richiede Python vecchio, Torch preferisce Python nuovo), utilizziamo un approccio a **due livelli**:

### 1. Livello Sistema (Conda) - `mfa` & `ffmpeg`
Usiamo Conda per installare i binari di sistema e il Montreal Forced Aligner.

```bash
# Crea l'ambiente base con le dipendenze di sistema
conda env create -f environment.yml

# Attiva l'ambiente
conda activate deepfake_cv_env

# Verifica che mfa funzioni
mfa version
```

### 2. Livello Progetto (Python Venv) - `torch`, `opencv`
Usiamo un virtual environment locale per le librerie Python del progetto.

```bash
# Crea il venv (se non esiste)
python -m venv .venv

# Attiva il venv
source .venv/bin/activate

# Installa le dipendenze Python
pip install -r requirements.txt
```

> **IMPORTANTE**: Quando esegui gli script Python, assicurati di essere nel `.venv`. Quando usi `mfa` da riga di comando, assicurati di avere l'ambiente Conda attivo.

---

## Workflow: Creazione Dataset e Analisi

Segui questi passaggi per preparare i dati e analizzare i video.

### Fase 1: Preparazione Dati
Gli script si trovano in `prepare_dataset/`.

1.  **Prepara i file per MFA**:
    ```bash
    python prepare_dataset/step1_prepare_mfa.py
    ```
    *Input*: `dataset/init/audio_25k` + `dataset/output/alignments`
    *Output*: `dataset/output/mfa_workspace`

2.  **Esegui l'allineamento (MFA)**:
    *(Da eseguire con ambiente Conda attivo)*
    ```bash
    mfa align dataset/output/mfa_workspace english_us_arpa english_us_arpa dataset/output/mfa_output_phonemes
    ```

3.  **Genera gli Embedding Video**:
    ```bash
    python prepare_dataset/step2_generate_video_embeddings.py
    ```
    *Input*: Video in `dataset/init` + TextGrid in `dataset/output/mfa_output_phonemes`
    *Output*: `dataset/output/video_embeddings`

4.  **Costruisci il Dizionario Gold (Riferimento)**:
    ```bash
    python prepare_dataset/step4_build_gold_dictionary.py
    ```
    *Output*: `dataset/output/gold_store/s1_gold_dictionary.json`

### Fase 2: Utilizzo e Analisi
Gli script si trovano in `scripts/`.

-   **Audit di un singolo video**:
    Confronta un video sospetto contro l'identità "Gold".
    ```bash
    PYTHONPATH=. python scripts/02_audit_video.py \
      dataset/init/s1/video_test.mpg \
      dataset/output/gold_store/s1_gold.json \
      --mfa-dict english_us_arpa \
      --mfa-acoustic english_us_arpa
    ```

-   **Cross-Check (Sicurezza)**:
    Verifica se il sistema distingue due identità diverse (es. S1 vs S2).
    ```bash
    python scripts/step6_cross_check.py
    ```