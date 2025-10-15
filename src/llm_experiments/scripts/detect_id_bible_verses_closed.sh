#!/bin/bash
OPENAI_API_KEY="REPLACE"
export OPENAI_API_KEY
ANTHROPIC_API_KEY="REPLACE"
export ANTHROPIC_API_KEY

MODEL_SET=("gpt-4o-2024-08-06" "claude-3-7-sonnet-20250219")
TEMPERATURE_SET=(0.25)

echo "Running LLM experiments for biblical reference detection..."
# no sampling, temp = 0
#for MODEL in "${MODEL_SET[@]}"
#do
#    echo "Running model: $MODEL"
#    python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_closed_llm_outputs_br.py --model $MODEL --prompt_path /home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/PROMPT_DETECT_AND_ID.txt
#done

# sampling, temp = 0.25, 0.5, 0.75, 1.0
for MODEL in "${MODEL_SET[@]}"
do
    for TEMPERATURE in "${TEMPERATURE_SET[@]}"
    do
        echo "Running model: $MODEL with temperature: $TEMPERATURE"
        python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_closed_llm_outputs_br.py --model $MODEL --sample --temperature $TEMPERATURE --prompt_path /home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/PROMPT_DETECT_AND_ID.txt
    done
done
