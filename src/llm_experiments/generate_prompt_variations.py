import os
import itertools
import json
from typing import List, Dict, Any
import argparse
import shutil

import re
from tqdm import tqdm


##### BIBLE REFERENCE TASK #####
## NOT TRYING THIS WITH RELIGIOUS CLASSIFICATION

# example 1 is an abraham lincoln quote
# example 2 is something I made up
ONE_SHOT_EXAMPLE = "It may seem strange that any men should dare to ask a just God’s assistance in wringing their bread from the sweat of other men’s faces but let us judge not that we be not judged."
TWO_SHOT_EXAMPLE = "The church is a powerful force in our world, and its influence can be felt in many spheres of society."

ONE_SHOT_EXAMPLE_JUST_DETECT = "EXAMPLE INPUT:\n" + ONE_SHOT_EXAMPLE + "\n\nEXAMPLE OUTPUT:\n{output_format_yes}"
ONE_SHOT_EXAMPLE_VERSE_ID = "EXAMPLE INPUT:\n" + ONE_SHOT_EXAMPLE + "\n\nEXAMPLE OUTPUT:\n{output_format_yes}\nGenesis 3:19"
TWO_SHOT_EXAMPLE_JUST_DETECT = "EXAMPLE INPUT 1:\n" + ONE_SHOT_EXAMPLE + "\n\nEXAMPLE OUTPUT:\n{output_format_yes}\n\n" + "EXAMPLE INPUT 2:\n" + TWO_SHOT_EXAMPLE + "\n\nEXAMPLE OUTPUT 2:\n{output_format_no}"
TWO_SHOT_EXAMPLE_VERSE_ID = "EXAMPLE INPUT 1:\n" + ONE_SHOT_EXAMPLE + "\n\nEXAMPLE OUTPUT:\n{output_format_yes}\nGenesis 3:19\n\n" + "EXAMPLE INPUT 2:\n" + TWO_SHOT_EXAMPLE + "\n\nEXAMPLE OUTPUT 2:\n{output_format_no}\n{output_format_null_verse}"

class PromptVariationGenerator:
    """Generate different variations of a base prompt according to specified parameters."""
    
    def __init__(self, base_prompt_path: str, output_folder: str):
        """
        Initialize the prompt variation generator.
        
        Args:
            base_prompt_path: Path to the file containing the base prompt
            output_folder: Path to the folder where variations will be saved
        """
        self.base_prompt_path = base_prompt_path
        self.output_folder = output_folder
        self.base_prompt = self._read_base_prompt()
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
    def _read_base_prompt(self) -> str:
        """Read the base prompt from file."""
        with open(self.base_prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    def generate_variations(self, task_type: str, multiple_tasks: bool) -> None:
        """Generate all prompt variations based on the specified parameters."""
        variation_params = self._get_variation_parameters(task=task_type)
        
        # Generate all possible combinations of variations
        keys = variation_params.keys()
        values = variation_params.values()
        
        # Get all combinations
        combinations = list(itertools.product(*values))
        
        
        # Generate and save each variation
        prompts = []
        num_prompts = 0
        for i, combo in tqdm(enumerate(combinations)):

            variation_dict = dict(zip(keys, combo))
            if (task_type == "verse_detection") and (variation_dict['json_format'] and variation_dict['output_format']): # can't have both
                continue
        
            variation_dict['base_prompt'] = self.base_prompt
            prompt_text = self._apply_variations(variation_dict, task_type=task_type, multiple_tasks=multiple_tasks)

            if prompt_text not in prompts:
                num_prompts += 1
                # Create a descriptive filename
                n_tasks = 2 if multiple_tasks else 1
                FILE_ID = f".{task_type}.{n_tasks}_tasks"
                filename = f"variation_{i+1:03d}" + FILE_ID
                filename = filename.replace('/', '_').replace('\\', '_')[:150] + ".txt"
                
                # Save the prompt variation
                with open(os.path.join(self.output_folder, filename), 'w', encoding='utf-8') as f:
                    f.write(prompt_text)
                with open(os.path.join(self.output_folder.replace("variations", "metadata"), filename.replace('.txt', '.json')), 'w', encoding='utf-8') as f:
                    json.dump(variation_dict, f)

                prompts.append(prompt_text)
                
        print(f"Generated {num_prompts} prompt variations in {self.output_folder}")
    
    def _get_variation_parameters(self, task) -> Dict[str, List[Any]]:
        """Define the variation parameters."""
        if task == "verse_detection":
            return {
                # Change output format
                "output_format": [
                    None,
                    ["YES", "NO"],
                    ["Verse", "Not Verse"],
                    ["True", "False"],
                ],
                
                # Add OUTPUT: at end
                #"add_output_marker": [False, True],
                
                # Request JSON format output
                "json_format": [
                    None,
                    'is_verse: boolean indicating if this is a biblical verse, verse_citation: citation if it is a verse, otherwise null'
                ],
                
                # Add example(s)
                "example_type": [
                    None,  # No examples
                    "one_shot",  # One positive example
                    "two_shot"   # One positive and one negative example
                ],
                
                # Add chain-of-thought prompt
                "use_cot": [False, True],
                
                # Add role prompting
                "role": [
                    None,  # No role
                    "scholar of religious rhetoric",
                    "seasoned political scientist",
                ]
            }
        else:
            return {
                # Add chain-of-thought prompt
                "use_cot": [False, True],
                
                # Add role prompting
                "role": [
                    None,  # No role
                    "scholar of religious rhetoric",
                    "seasoned political scientist",
                ]
            }
    
        
    def _apply_variations(self, variation_dict: Dict[str, Any], task_type="verse_detection", multiple_tasks=True) -> str:
        """
        Apply the specified variations to the base prompt.
        
        Args:
            variation_dict: Dictionary of variation parameters
            
        Returns:
            Modified prompt with all variations applied
        """
        prompt = self.base_prompt
        # remove "INPUT STRING:" from the end of the prompt and add later
        # remove "SPEECH:" from the end of the prompt and add later
        prompt = prompt.replace("INPUT STRING:", "").replace("SPEECH:", "").strip() + "\n"

        # Apply role prompting (if specified)
        if variation_dict["role"]:
            role_text = f"Answer the following as a {variation_dict['role']}.\n\n"
            prompt = role_text + prompt
            
        # Generate examples based on example type
        examples_text = ""
        if (task_type == "verse_detection"):
            if variation_dict["output_format"]:
                if variation_dict["example_type"] == "one_shot":
                    examples_text = self._generate_one_shot_example(task_type, multiple_tasks, variation_dict["output_format"])
                    examples_text = "The following example represents a correctly formatted input string-output pair.\n" + examples_text
                elif variation_dict["example_type"] == "two_shot":
                    examples_text = self._generate_two_shot_examples(task_type, multiple_tasks, variation_dict["output_format"])
                    examples_text = "The following examples represent correctly formatted input string-output pairs.\n" + examples_text
                prompt = prompt + "\n\n" + examples_text

        # Apply chain-of-thought prompting
        if variation_dict["use_cot"]:
            cot_text = "\n\nPlease think step-by-step through your reasoning before providing your final answer."
            prompt += cot_text
            
        # Apply output format specifications
        if (task_type == "verse_detection"):
            if variation_dict["output_format"]:
                instr = f"Output \"{variation_dict['output_format'][0]}\" if it is a Bible verse reference, and \"{variation_dict['output_format'][1]}\" if it is not. Do not explain your answer."
                prompt += f"\n\n{instr}"
            
        # Request JSON output if specified + not covered by output_format
        if (task_type == "verse_detection"):
            if (variation_dict["json_format"]) and (not variation_dict["output_format"]):
                json_structure = json.dumps(variation_dict["json_format"], indent=2)
                prompt += f"\n\nFormat your response as JSON with the following structure:\n{json_structure}"
            
        # add "INPUT STRING:" or "SPEECH:"
        if task_type == "verse_detection":
            prompt += "\n\nINPUT STRING:\n"
        else:
            prompt += "\n\nSPEECH:\n"

        # make any series of \ns longer than \n\n into \n\n
        # regex for more than 2 newlines
        too_many_newlines = r"\n\n\n+"
        prompt = re.sub(too_many_newlines, "\n\n", prompt)
            
        return prompt
    
    def _generate_one_shot_example(self, task, multi, output_format) -> str:
        """Generate a one-shot example (positive example)."""
        if task == "verse_detection":
            if multi:
                return ONE_SHOT_EXAMPLE_VERSE_ID.format(output_format_yes=output_format[0])
            else:
                return ONE_SHOT_EXAMPLE_JUST_DETECT.format(output_format_yes=output_format[0])
        else:
            print("Task not supported")
        
    
    def _generate_two_shot_examples(self, task, multi, output_format) -> str:
        """Generate two-shot examples (positive and negative)."""
        if task == "verse_detection":
            if multi:
                return TWO_SHOT_EXAMPLE_VERSE_ID.format(output_format_yes=output_format[0], output_format_no=output_format[1], output_format_null_verse="NOT VERSE")
            else:
                return TWO_SHOT_EXAMPLE_JUST_DETECT.format(output_format_yes=output_format[0], output_format_no=output_format[1])
        else:
            print("Task not supported")


def main():
    parser = argparse.ArgumentParser(description='Generate prompt variations.')
    parser.add_argument('base_prompt', help='Path to the base prompt file')
    parser.add_argument('--task_type', default='verse_detection') # other possible task is religious_classification
    parser.add_argument('--multiple_tasks', action='store_true')
    parser.add_argument('--output', '-o', default='/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_variations', 
                        help='Output folder for prompt variations (default: prompt_variations)')
    parser.add_argument('--output_metadata', default='/home/laviniad/projects/religion_in_congress/src/llm_experiments/prompt_metadata', 
                        help='Output folder for prompt variations (default: prompt_variations)')
    parser.add_argument('--clear_directories', action='store_true', help='Clear output directories before generating variations')
    
    args = parser.parse_args()

    # couple of checks
    if not os.path.exists(args.base_prompt):
        raise FileNotFoundError(f"Base prompt file {args.base_prompt} does not exist.")

    assert args.task_type in ['verse_detection', 'religious_classification'], "task_type must be either verse_detection or religious_classification"
    MULTIPLE_TASKS = True if args.multiple_tasks else False
    if args.task_type == 'verse_detection':
        args.output = os.path.join(args.output, 'verse_detection')
    else:
        args.output = os.path.join(args.output, 'religious_classification')

    if args.clear_directories:
        if os.path.exists(args.output):
            for filename in os.listdir(args.output):
                file_path = os.path.join(args.output, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
            for filename in os.listdir(args.output.replace("variations", "metadata")):
                file_path = os.path.join(args.output.replace("variations", "metadata"), filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            print(f"Output directory {args.output} does not exist. Creating it.")
            os.makedirs(args.output, exist_ok=True)
    
    generator = PromptVariationGenerator(args.base_prompt, args.output)
    generator.generate_variations(task_type=args.task_type, multiple_tasks=MULTIPLE_TASKS)


if __name__ == "__main__":
    main()