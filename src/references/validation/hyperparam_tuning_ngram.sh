#!/bin/bash

#method="fuzzy_string_matching_token"
#method="embedding_similarity"
method="ngram_shingling_min_count_overlap"
#method="fuzzy_string_matching_token"

# embedding_sim
if [[ "$method" == "embedding_similarity" ]]; then
    input_values=(0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
fi

# ngram_shingling
if [[ "$method" == "ngram_shingling_min_count_overlap" ]]; then
    input_values=(1 2 3 4 5) # integers, n of ngram
fi

# fuzzy_string_matching_token
if [[ "$method" == "fuzzy_string_matching_token" ]]; then
    input_values=(0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75)
fi

# Initialize variables to store the best F1 score and corresponding input value
best_f1=0
best_input=""

# Create a file to store the F1 scores
output_file="/home/laviniad/projects/religion_in_congress/src/references/validation/f1_scores.txt"
echo "Hyperparameter F1 Score" > "$output_file"

touch /home/laviniad/projects/religion_in_congress/src/references/validation/f1_log.txt
counter=0

# Loop through each input value
for value in "${input_values[@]}"
do
    # Run eval.py with the current input value
    python /home/laviniad/projects/religion_in_congress/src/references/validation/eval.py --method "$method" --hyperparam "$value" --id "$counter"
    f1=$(cat /home/laviniad/projects/religion_in_congress/src/references/validation/f1_log.txt)

    echo "F1 score for $method with hyperparameter $value: $f1"

    # Record the F1 score in the output file
    echo "$value $f1" >> "$output_file"

    # Check if the current F1 score is better than the previous best
    if (( $(echo "$f1 > $best_f1" | bc -l) ))
    then
        best_f1=$f1
        best_input=$value
    fi
    counter=$((counter+1))
done

echo "Best F1 score: $best_f1"
echo "Best hyperparameter for $method: $best_input"
