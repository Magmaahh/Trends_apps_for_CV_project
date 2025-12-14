#!/bin/bash

# Directory containing MFA workspaces
MFA_OUTPUT_DIR="dataset/output"

# Path to your dictionary and acoustic model
DICT_PATH="dataset/output/mfa_data/english_us_arpa.dict"
MODEL_PATH="dataset/output/mfa_data/english_us_arpa.zip"

echo "Starting MFA alignment for all speakers..."

for d in ${MFA_OUTPUT_DIR}/mfa_workspace_s*; do
    if [ -d "$d" ]; then
        speaker=$(basename "$d" | sed 's/mfa_workspace_//')
        echo "Aligning speaker: $speaker"
        
        output_dir="${MFA_OUTPUT_DIR}/mfa_output_phonemes_${speaker}"
        mkdir -p "$output_dir"
        
        mfa align "$d" "$DICT_PATH" "$MODEL_PATH" "$output_dir" --clean --verbose
        
        echo "Finished $speaker"
    fi
done

echo "All alignments complete."
