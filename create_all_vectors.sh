#!/bin/bash

# Usage examples:
# ./create_all_control_vectors.sh "0" "./aya-23-35B" "aya-23:35b-" 8192 8 orig none
# ./create_all_control_vectors.sh "1" "./Qwen1.5-14B-Chat" "qwen-1.5:14b-" 5120 4 orig none
# ./create_all_control_vectors.sh "0,1" "./c4ai-command-r-plus" "command-r-plus:104b-" 12288 2 orig none

# Check if we have the correct number of arguments
if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <cuda_devices> <model_path> <output_prefix> <num_samples> [batch_size=1] [precision=orig] [attn=none]"
    echo "Example: $0 \"0,1\" \"/path/to/model\" \"model-prefix-\" 12345 8 orig none"
    exit 1
fi

# Assuming the 'data' sub-folder is in the default location.
DATA="data"
STEMS="$DATA/prompt_stems.json"
PROMPTS="$DATA/writing_prompts.txt"

# Assign arguments to variables
CUDA_DEVICES="$1"
MODEL_PATH="$2"
OUTPUT_PREFIX="$3"
NUM_SAMPLES="$4"
BATCH_SIZE="${5:-1}"
PRECISION="${6:-orig}"
ATTN="${7:-none}"

# Define arrays for continuations and output suffixes
continuations=(
    "$DATA/writing_style/character_focus.json"
    "$DATA/writing_style/language.json"
    "$DATA/writing_style/storytelling.json"
    "$DATA/dark_tetrad/help_vs_harm.json"
    "$DATA/dark_tetrad/empathic_vs_callous.json"
    "$DATA/dark_tetrad/candid_vs_cagey.json"
    "$DATA/dark_tetrad/humility_vs_narcissism.json"
    "$DATA/other/optimism_vs_pessimism.json"
)

output_suffixes=(
    "character_focus_"
    "language_"
    "storytelling_"
    "help_vs_harm_"
    "empathic_vs_callous_"
    "candid_vs_cagey_"
    "humility_vs_narcissism_"
    "optimism_vs_pessimism_"
)

# Set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# Loop through continuations and create control vectors
for i in "${!continuations[@]}"; do
    python3 ./create_control_vectors.py \
        --model "$MODEL_PATH" \
        --outpath "${OUTPUT_PREFIX}${output_suffixes[i]}" \
        --prompts "$STEMS" \
        --writing-prompts-file "$PROMPTS" \
        --continuations "${continuations[i]}" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --precision "$PRECISION" \
        --attn "$ATTN"
done
