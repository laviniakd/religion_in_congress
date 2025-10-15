#!/bin/bash
MODEL_SET=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.2-11B-Instruct" "mistralai/Mistral-7B-Instruct-v0.3" "mistralai/Ministral-8B-Instruct-2410")

TEMPERATURE_SET=(0.25)

echo "Running LLM experiments for religiosity labeling..."
# no sampling, temp = 0
#for MODEL in "${MODEL_SET[@]}"
#do
#    echo "Running model: $MODEL"
#    python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_llm_outputs_rc.py --debug --hf_model $MODEL
#done

# sampling, temp = 0.25, 0.5, 0.75, 1.0
for MODEL in "${MODEL_SET[@]}"
do
    for TEMPERATURE in "${TEMPERATURE_SET[@]}"
    do
        echo "Running model: $MODEL with temperature: $TEMPERATURE"
        python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_llm_outputs_rc.py --debug --hf_model $MODEL --sample --temperature $TEMPERATURE
    done
done