#!/bin/bash
# REPLACE API KEYS BEFORE RUNNING

OPENAI_API_KEY="REPLACE"
export OPENAI_API_KEY
ANTHROPIC_API_KEY="REPLACE"
export ANTHROPIC_API_KEY
#MODEL_SET=("gpt-4o-2024-08-06" "claude-3-7-sonnet-20250219")
MODEL_SET=("claude-3-7-sonnet-20250219")

TEMPERATURE_SET=(0.25)

echo "Running LLM experiments for biblical reference detection..."
prompt_dir="/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/verse_detection/"
# create list of all files in prompt_dir with .txt extension
prompt_files=$(ls $prompt_dir*.txt)

# sampling, temp = 0.25, 0.5, 0.75, 1.0
for MODEL in "${MODEL_SET[@]}"
do
    for TEMPERATURE in "${TEMPERATURE_SET[@]}"
    do
        for PROMPT in $prompt_files
        do
            echo "Running model: $MODEL with temperature: $TEMPERATURE"
            echo "Prompt path: $PROMPT"
            # NOW: ONLY RUN IF IN A LINE IN "/home/laviniad/projects/religion_in_congress/data/prompts_to_run.txt"
            if grep -q "$PROMPT" "/home/laviniad/projects/religion_in_congress/data/prompts_to_run.txt"; then
                echo "Running prompt: $PROMPT"
                python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_closed_llm_outputs_br.py --model $MODEL --sample --temperature $TEMPERATURE --prompt $PROMPT
            else
                echo "Skipping prompt: $PROMPT"
                continue
            fi
        done
    done

done
