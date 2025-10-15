#!/bin/bash
OPENAI_API_KEY="REPLACE"
export OPENAI_API_KEY
ANTHROPIC_API_KEY="REPLACE"
export ANTHROPIC_API_KEY
MODEL_SET=("claude-3-7-sonnet-20250219")
TEMPERATURE_SET=(0.25)

echo "Running LLM experiments for religiosity labeling..."

# sampling, temp = 0.25, 0.5, 0.75, 1.0
prompt_dir="/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/best_rc/"
# create list of all files in prompt_dir with .txt extension
prompt_files=$(ls $prompt_dir*.txt)

echo "Running LLM experiments for religiosity labeling..."

# sampling, temp = 0.25, 0.5, 0.75, 1.0
for MODEL in "${MODEL_SET[@]}"
do
    for TEMPERATURE in "${TEMPERATURE_SET[@]}"
    do
        for PROMPT in $prompt_files
        do
            echo "Running model: $MODEL with temperature: $TEMPERATURE and prompt: $PROMPT"
            python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_closed_llm_outputs_rc.py --debug --model $MODEL --sample --temperature $TEMPERATURE --prompt $PROMPT        
        done
    done
done