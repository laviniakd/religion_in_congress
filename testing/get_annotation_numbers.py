# want to run load_all_annotations in /home/laviniad/projects/religion_in_congress/src/llm_experiments/postprocess_llm_outputs.py and see df length + label dist

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.llm_experiments.postprocess_llm_outputs import load_all_annotations

validation_df, test_df = load_all_annotations()

print(f"Total annotations (validation): {len(validation_df)}")

print(f"Total annotations (test): {len(test_df)}")
