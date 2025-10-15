# save best prompts to /home/laviniad/projects/religion_in_congress/src/llm_experiments/best_br_prompts with model name in filename
import json
import pprint
import pandas as pd
from generate_prompt_variations import PromptVariationGenerator

best_prompt_f1_df = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/llm_logs/best_prompt_f1_df.csv')

base_prompt_path = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/PROMPT_DETECT_AND_ID.txt"
output_folder = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/best_br/"
output_meta = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_metadata/best_br/"
pvg = PromptVariationGenerator(base_prompt_path=base_prompt_path, output_folder=output_folder)

def generate_prompt(best_br_metadata):
    best_br_prompt = pvg._apply_variations(
        best_br_metadata,
    )
    return best_br_prompt


for i, row in best_prompt_f1_df.iterrows():
    model = row['model']
    timestamp = row['timestamp']
    prompt_metadata = row['prompt_metadata']
    prompt_metadata = json.loads(prompt_metadata)
    print(f"Model: {model}")
    print("Metadata:")
    pprint.pprint(prompt_metadata)

    model_name = model.split("/")[-1]

    best_rc_prompt = generate_prompt(prompt_metadata)
    print(best_rc_prompt)
    with open(f"{output_folder}{model_name}.txt", "w") as f:
        f.write(best_rc_prompt)
    with open(f"{output_meta}{model_name}.json", "w") as f:
        json.dump(prompt_metadata, f)
    print("..........")

    