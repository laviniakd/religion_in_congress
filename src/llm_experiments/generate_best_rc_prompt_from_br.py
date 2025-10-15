import json
import pprint
from generate_prompt_variations import PromptVariationGenerator

path_to_best_br_metadata = "/home/laviniad/projects/religion_in_congress/data/llm_logs/errata/best_prompt_metadata_br_claude.json"


# create a new file with the best rc prompt based on br metadata
def generate_best_rc_prompt_from_br(best_br_metadata):
    best_rc_prompt = pvg._apply_variations(
        best_br_metadata,
        task_type="religious_classification",
    )
    return best_rc_prompt

if __name__ == "__main__":
    base_prompt_path = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompts/RELIGIOUS_THEME_KEYWORDS_PROMPT.txt"
    output_folder = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations/best_rc/"
    output_meta = "/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_metadata/best_rc/"
    pvg = PromptVariationGenerator(base_prompt_path=base_prompt_path, output_folder=output_folder)
    # create a new file with the best rc prompt based on br metadata
    with open(path_to_best_br_metadata, "r") as f:
        best_br_metadata = json.load(f)

    best_br_metadata['example_type'] = None
    
    print("Best BR metadata:")
    pprint.pprint(best_br_metadata)

    best_rc_prompt = generate_best_rc_prompt_from_br(best_br_metadata)
    print(best_rc_prompt)
    with open(f"{output_folder}best_rc_prompt_based_on_br.txt", "w") as f:
        json.dump(best_rc_prompt, f, indent=4)
    with open(f"{output_meta}best_rc_prompt_metadata.json", "w") as f:
        json.dump(best_br_metadata, f, indent=4)
              