#!/bin/bash
OPENAI_API_KEY="REPLACE"
export OPENAI_API_KEY
ANTHROPIC_API_KEY="REPLACE"
export ANTHROPIC_API_KEY
#MODEL_SET=("gpt-4o-2024-08-06" "claude-3-7-sonnet-20250219")
MODEL_SET=("claude-3-7-sonnet-20250219")

#DATA_PATH="/data/laviniad/congress_errata/TEST_reference_df.csv"
DATA_PATH="/home/laviniad/results_combined.csv"

TEMPERATURE_SET=(0.25)

echo "Running LLM experiments for biblical reference detection..."
prompt_dir="/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/best_br/"
# create list of all files in prompt_dir with .txt extension
prompt_files=("$prompt_dir"*.txt)

# sampling, temp = 0.25, 0.5, 0.75, 1.0
for MODEL in "${MODEL_SET[@]}"
do
    for TEMPERATURE in "${TEMPERATURE_SET[@]}"
    do
        for PROMPT in "${prompt_files[@]}"
        do
            MODEL_NAME_SHORT=$(echo $MODEL | cut -d'/' -f2)
            MODEL_NAME_SHORT=$MODEL_NAME_SHORT".txt"
            PROMPT_SHORT=$(basename $PROMPT)
            if [[ "$MODEL_NAME_SHORT" == "$PROMPT_SHORT" ]]; then
                echo "Running model: $MODEL with temperature: $TEMPERATURE"
                echo "Running prompt: $PROMPT"
                python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_closed_llm_outputs_br.py --model $MODEL --sample --temperature $TEMPERATURE --prompt_path $PROMPT --data_path $DATA_PATH
            else
                echo "Skipping prompt: $PROMPT"
                continue
            fi
        done
    done
done

HF_TOKEN=REPLACE
HF_HOME=/data/laviniad/transformers_cache
export HF_HOME
export HF_TOKEN

#LLAMA_SET=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
#MISTRAL_SET=("mistralai/Ministral-8B-Instruct-2410")
#OLMO_SET=("allenai/OLMo-2-0325-32B-Instruct")
#OPEN_MODEL_SET=("${LLAMA_SET[@]}" "${MISTRAL_SET[@]}" "${OLMO_SET[@]}")
#OPEN_MODEL_SET=("meta-llama/Llama-3.1-8B-Instruct")
OPEN_MODEL_SET=()

for MODEL in "${OPEN_MODEL_SET[@]}"
do
    for TEMPERATURE in "${TEMPERATURE_SET[@]}"
    do
        for PROMPT in "${prompt_files[@]}"
        do
            MODEL_NAME_SHORT=$(echo $MODEL | cut -d'/' -f2)
            MODEL_NAME_SHORT=$MODEL_NAME_SHORT".txt"
            PROMPT_SHORT=$(basename $PROMPT)
            if [[ "$MODEL_NAME_SHORT" == "$PROMPT_SHORT" ]]; then
                echo "Running model: $MODEL with temperature: $TEMPERATURE"
                echo "Running prompt: $PROMPT"
                python /home/laviniad/projects/religion_in_congress/src/llm_experiments/add_llm_outputs_br.py --hf_model $MODEL --sample --temperature $TEMPERATURE --prompt_path $PROMPT --data_path $DATA_PATH --device 'auto'
            else
                echo "Skipping prompt: $PROMPT"
                continue
            fi
        done
    done
done
