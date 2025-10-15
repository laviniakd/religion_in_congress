#!/bin/bash

HF_TOKEN=REPLACE
HF_HOME=/data/laviniad/transformers_cache
export HF_HOME
export HF_TOKEN

LLAMA_SET=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.1-70B-Instruct")
MISTRAL_SET=("mistralai/Mistral-Small-3.1-24B-Instruct-2503" "mistralai/Mistral-Small-24B-Instruct-2501" "mistralai/Ministral-8B-Instruct-2410")
OLMO_SET=("allenai/OLMo-2-0325-32B-Instruct" "allenai/OLMo-2-1124-13B-Instruct" "allenai/OLMo-2-1124-7B-Instruct")
MODEL_SET=("${LLAMA_SET[@]}" "${MISTRAL_SET[@]}" "${OLMO_SET[@]}")
TEMPERATURE_SET=(0.25)

#prompt_dir="/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/religious_classification/"
#prompt_files=$(ls $prompt_dir*.txt)

prompt_files=("/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/religious_classification/religious_classification_prompt.txt")

echo "Running LLM experiments for religiosity labeling..."

for MODEL in "${MODEL_SET[@]}"
do
    for TEMPERATURE in "${TEMPERATURE_SET[@]}"
    do
        for PROMPT in $prompt_files
        do
            echo "Running model: $MODEL with temperature: $TEMPERATURE"
            echo "Prompt path: $PROMPT"
            python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_llm_outputs_rc.py --debug --hf_model $MODEL --sample --temperature $TEMPERATURE
        done
    done

    # write model name to file in llm_logs
    echo $MODEL >> /home/laviniad/projects/religion_in_congress/data/llm_logs/model_fin/model_name_rc.txt
done