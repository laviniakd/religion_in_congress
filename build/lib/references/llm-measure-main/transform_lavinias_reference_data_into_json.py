# want dict along lines of {"context": {id1: text, id2: text2, ...}, "labels": {id1: text, ...}, "prompts": {}, "IDs": {}}

import pandas as pd
import json

# load eval data
#ses = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/special_example_set_with_labels.csv')
annotations_from_dallas_and_lavinia = pd.read_csv('/home/laviniad/projects/religion_in_congress/data/dallas_lavinia_combined_annotations.csv')

#prompt = "\n\nIf a scholar of religion and politics considered the above sentence, would she say it is a reference to a specific Bible verse?\nA: Reference\nB: Not Reference\nC: None\n\nConstraint: Answer with only the option above that is most accurate and nothing else."
prompt = "\n\nThis is a sentence spoken by an American politician in a legislative environment. Please do the following:\n1. Output either VERSE (if it is a verse) or NOT_VERSE (if it is not a verse), followed by a comma and space (\", \").\n2. If it's a verse, output the Bible verse itself in the standard citation format, e.g., Genesis 1:1; if it's not a verse, output NOT_VERSE again.\nFor example, for the sentence \"We know that there is no greater honor than to die for one's brothers.\" you should output VERSE, John 15:13. For the sentence \"The markets will crash without a bank bailout.\" you should output NOT_VERSE, NOT_VERSE\n\n"
result_dict = {"context": {}, "labels": {}, "prompts": {}, "IDs": {}}

# iterate through
for idx, row in annotations_from_dallas_and_lavinia.iterrows():
    context = row['text']
    label = row['Match_dallas']
    prompt = prompt
    ID = row['congress_idx_dallas']
    
    result_dict["context"][ID] = context
    result_dict["labels"][ID] = label
    result_dict["prompts"][ID] = prompt
    result_dict["IDs"][ID] = ID

with open('/home/laviniad/projects/religion_in_congress/src/references/llm-measure-main/data/biblical_references/150_annotated.json', 'w') as f:
    json.dump(result_dict, f)
