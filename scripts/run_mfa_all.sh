#!/bin/bash

# Script per allineare TUTTI gli speaker automaticamente con MFA.
# Da eseguire nell'ambiente Conda (dove c'è mfa installato).

DATASET_OUTPUT="dataset/output"
MFA_DICT="english_us_arpa"
MFA_ACOUSTIC="english_us_arpa"

echo "--- INIZIO ALLINEAMENTO MASSIVO ---"

# Trova tutte le cartelle mfa_workspace_s*
for workspace in "$DATASET_OUTPUT"/mfa_workspace_s*; do
    if [ -d "$workspace" ]; then
        # Estrai il nome dello speaker (es. s1, s2)
        dirname=$(basename "$workspace")
        speaker_id=${dirname#mfa_workspace_}
        
        output_dir="$DATASET_OUTPUT/mfa_output_phonemes_$speaker_id"
        
        echo ""
        echo ">>> Elaborazione Speaker: $speaker_id"
        echo "    Input:  $workspace"
        echo "    Output: $output_dir"
        
        # Controlla se ci sono file .wav nel workspace
        count=$(find "$workspace" -name "*.wav" | wc -l)
        if [ "$count" -eq 0 ]; then
            echo "    ⚠️  SKIP: Nessun file audio trovato in $workspace"
            continue
        fi

        # Esegui MFA
        # --clean forza la pulizia della cache precedente per evitare errori
        mfa align "$workspace" "$MFA_DICT" "$MFA_ACOUSTIC" "$output_dir" --clean --verbose
        
        if [ $? -eq 0 ]; then
            echo "    ✅ Successo per $speaker_id"
        else
            echo "    ❌ ERRORE per $speaker_id"
        fi
    fi
done

echo ""
echo "--- FINE ALLINEAMENTO ---"
