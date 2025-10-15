import argparse
import json
import torch 
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import numpy as np
from random import sample
import os
from openai import OpenAI
import tiktoken
import regex as re


def read_json(input_filename):
    with open(input_filename) as input_file:
        data = json.load(input_file)
    return data

def preprocess_prompt(prompt, tokenizer,max_tokens):
    prompt = tokenizer.clean_up_tokenization(
    tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            tokenizer(prompt, max_length=max_tokens, truncation=True)[
                "input_ids"
            ]
            )
        )
    )
    return prompt

def initiate_flan5(model_string, hftoken, decode=True):
    tokenizer = AutoTokenizer.from_pretrained(model_string,truncation_side="left", token=hftoken)
    if "llama" in model_string:
        model = AutoModelForCausalLM.from_pretrained(model_string,device_map="auto", token=hftoken)
        print("loaded auto causal")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_string,device_map="auto")
    
    
    return model, tokenizer

def get_gpt_response(model_string, prompt, temp, max_tokens,label_types=None):
    if label_types:
        encoding = tiktoken.get_encoding("cl100k_base")
        answer_tokens = [encoding.encode(i)[0] for i in label_types]
        weight = 20
        bias = {str(i): weight for i in answer_tokens}
    else:
        bias = {}

    client = OpenAI()
    api_query = client.chat.completions.create(
        model=model_string,
        messages=prompt,
        temperature=temp,
        max_tokens=max_tokens,
        logprobs = True,
        logit_bias=bias,
        seed= 5

    )
    response = api_query.choices[0].message.content
    logprob_out = [token.logprob for token in api_query.choices[0].logprobs.content]
    #system
    return response, logprob_out


def tokenized_labelset(tokenizer, label_set, add_comma=False):
    ls = set()
    for x in tokenizer(label_set, add_special_tokens=False)["input_ids"]:
        for y in x:
            ls.add(y)

    if add_comma:
        ls.add(tokenizer(", ", add_special_tokens=False)["input_ids"][0])

    return sorted(ls)

def flan_generate(prompt, model, model_string, tokenizer, temp):

    if "chat" in model_string or "Instruct" in model_string:
        inputs = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        input_ids = inputs.to("cuda")
        attention_mask = torch.ones_like(input_ids).to("cuda")
        
    else:
        tokenized = tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids.to("cuda")
        attention_mask = tokenized.attention_mask.to("cuda")

    if temp > 0:
        outputs = model.generate(input_ids, do_sample = True, temperature = temp, return_dict_in_generate=True, output_scores=True)
    
    elif temp == 0:    
        outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, do_sample = False, return_dict_in_generate=True, output_scores=True, attention_mask=attention_mask)#,max_new_tokens=2)
       # outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, do_sample=  False, return_dict_in_generate=True, output_scores=True)

        
    transition_scores = model.compute_transition_scores(sequences=outputs.sequences,scores=outputs.scores, normalize_logits=True)
    probs = torch.exp(transition_scores[0, :]).tolist()
    probs.insert(0,0) #pad token has no probability 
    outputs = tokenizer.decode(outputs.sequences[0,:])
    if "chat" in model_string or "Instruct" in model_string:
        # print("***")
        # print(outputs)
        # print("***")
        if re.search("end_header_id",outputs):
            outputs = outputs.split("<|end_header_id|>")[-1]
            outputs = outputs.strip()
        elif re.search("[/INST]", outputs):
            outputs = outputs.split("[/INST]")[1]
        
    return outputs, probs


def flan_generate_force_decode(prompt, model,tokenizer, temp, labels=None, nprobs = None):
    if nprobs == None:
        if labels:
            nprobs = len(labels)
        else:
            nprobs = 1
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    
    label_set = [label.lower() if len(label) > 2 else label for label in labels]
    
    LS = tokenized_labelset(tokenizer, label_set)
    decoder_input_ids = tokenizer("",return_tensors="pt").input_ids.to("cuda")
    decoder_input_ids = model._shift_right(decoder_input_ids)
    logits = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids).logits.flatten()
    all_probs = (torch.nn.functional.softmax(
                    torch.tensor([logits[i] for i in range(logits.shape[0])]),
                    dim=0,
                ).detach().cpu().numpy()
    )

    probs = (
        torch.nn.functional.softmax(
            torch.tensor([logits[i] for i in LS]),
            dim=0,
        ).detach().cpu().numpy()
    )

    LS_str_map = tokenizer.decode(LS).split(" ")

    response = {i: LS_str_map[i] for i in range(len(LS))}[np.argmax(probs)]

    
    ind = np.argpartition(all_probs, -nprobs)[-nprobs:]
    ind = ind[np.argsort(all_probs[ind])]
    top_gen = tokenizer.decode(ind) #top nprobs most likely generations 
    top_gen_probs = all_probs[ind] #probabilities of most likely generations 

    return response, top_gen, top_gen_probs
        

def parse_arguments():
    parser = argparse.ArgumentParser(description="llm-measure")
    parser.add_argument("--path",
                        type = str,
                        help = "path to json file")
    parser.add_argument("--model",
                       type = str,
                       choices = [
                           "google/flan-t5-small",
                           "google/flan-t5-base",
                           "google/flan-t5-large",
                           "google/flan-t5-xl",
                           "google/flan-t5-xxl",
                           "google/flan-ul2",
                           "meta-llama/Llama-2-13b-chat-hf",
                           "meta-llama/Llama-2-13b",
                           "meta-llama/Llama-2-13b-chat",
                           "meta-llama/Meta-Llama-3-8B",
                           "meta-llama/Meta-Llama-3-8B-Instruct",
                           "gpt-4",
                           "gpt-4o",
                           "gpt-3.5-turbo",
                           "gpt-3.5-turbo-instruct"
                       ])
    parser.add_argument("--label_type", 
                        type = list,
                        default = None)
    parser.add_argument("--parallel",
                        type = bool,
                        help = "whether or not to parallel",
                        default = False)
    parser.add_argument("--prompt_end",
                        type = str,
                        help = "file with prompt end")
    parser.add_argument("--NGPUs",
                        type = int,
                        default = 1)
    parser.add_argument("--clean_prompt",
                        type = bool,
                        default = False,
                        help = "whether or not to clean the prompt using ziems preprocessing method")
    parser.add_argument("--max_tokens",
                        type = int,
                        default = 4094)
    parser.add_argument("--temperature",
                        type = float)
    parser.add_argument("--decode",
                        type = bool,
                        default = False
                        )
    parser.add_argument("--samples_key",
                       type = str,
                       help = "key in json file for samples (something like a tweet that goes before prompt end)")
    parser.add_argument("--labels_key",
                       type = str,
                       help = "key in json file for labels",
                       default=None)
    parser.add_argument("--other_key",
                        type = str,
                        default = None,
                        help = "key in json file for other aspects (e.g. target for stance), goes in prompt_end")
    parser.add_argument("--IDs_key",
                        type = str,
                        help = "key in json file for IDs")
    parser.add_argument("--ntimes",
                         type = int,
                         default =1)
    parser.add_argument("--log_dir",
                        type = str)
    parser.add_argument("--system_prompt",
                        type = str,
                        default=None)
    parser.add_argument("--hftoken",
                        type = str,
                        help="HuggingFace token")

    args = parser.parse_args()
    return args

def main(): 
  #  device = torch.device("cuda")
    args = parse_arguments()
    path = args.path
    data = read_json(path)
    IDs = list(data[args.IDs_key].keys())
    if args.other_key:
        data_aspect = data[args.other_key]
    if args.labels_key:
        labels = data[args.labels_key]
    if args.samples_key:
        samples = data[args.samples_key]


    parallel = args.parallel
    NGPUS = args.NGPUs
    clean_prompt = args.clean_prompt
    max_tokens = args.max_tokens
    prompt_file = args.prompt_end 
    prompt_end_template = open(prompt_file).read()
    prompt_end_template = prompt_end_template.replace("\\n", "\n")

    if args.system_prompt:
        system_prompt_file = args.system_prompt
        system_prompt = open(system_prompt_file).read()
        system_prompt = system_prompt.replace("\\n", "\n")


    temp = args.temperature
    model_string = args.model
    decode = args.decode
    if args.label_type:
        label_types = args.label_type
    else:
        label_types = None
    n_times = args.ntimes
    log_dir = args.log_dir

    # set up log directory 
    now = datetime.now()
    d = now.strftime("%Y%m%d-%H-%M")
    if "/" in model_string:
        log_file_list = [d, model_string.split("/")[1], str(temp)]
    else:
        log_file_list = [d, model_string, str(temp)]
    if decode:
        log_file_list.append("decode")
    log_file = "_".join(log_file_list)
    
    print(log_file)
    
    if "gpt" not in model_string:
        model, tokenizer = initiate_flan5(model_string, args.hftoken, decode)
        if parallel: 
            heads_per_gpu = len(model.encoder.block) // NGPUS
            device_map = {
                gpu: list(
                    range(
                        0 + (gpu * heads_per_gpu),
                        (0 + (gpu * heads_per_gpu)) + heads_per_gpu,
                    )
                )
                for gpu in range(NGPUS)
            }
            model.parallelize(device_map)
            
        model.eval()

    output = {'Index': [], "labels":[]}
    if args.other_key:
        output['aspect']= []
    
    print(n_times)
    seen = 0 
    for j in range(n_times):
        output['response_'+str(j)] = []
        if decode:
            output['run_'+str(j)+"_probs"] = []
        else:
            output['run_'+str(j)+"_topgen"] = []
            output['run_'+str(j)+"_topgen_probs"] = []
        


        for k in tqdm(range(len(IDs))): #tqdm(IDs):
            # get data from json:
            i = IDs[k]
            if args.samples_key:
                sample = samples[str(i)]
               # sample = re.sub(r'\s([?.!"](?:\s|$))', r'\1', sample)
            if args.labels_key:
                label = labels[str(i)]
            else:
                label = "na"
            if args.other_key:
                target = data_aspect[str(i)]
                # if target == "Donald Trump" or target == "Hillary Clinton":
                #     continue
        
            # formatting prompt:
                prompt_end = prompt_end_template.format(target)
            # if k > 3:
            #     continue
            else: 
                prompt_end = None
               
            if args.samples_key and prompt_end:
                prompt = sample + prompt_end
            elif args.samples_key:
                prompt = sample + prompt_end_template
            elif prompt_end:
                prompt = prompt_end
            else:
                prompt = prompt_end_template

            prompt = prompt.encode().decode('utf-8')
            if "chat" in model_string or "Instruct" in model_string or "gpt" in model_string:
                if args.system_prompt:
                    prompt = [{"role":"system", "content":system_prompt},{"role":"user","content":prompt}]
                else:
                    prompt = [{"role":"user","content":prompt}]
            if clean_prompt:
                #prompt = re.sub('\n', '', prompt) 
                prompt = prompt.replace("\\n",' ')
                print(prompt)
                prompt = preprocess_prompt(prompt, tokenizer, max_tokens)
            
            # generating: 
            if "gpt" in model_string:
                response, probs = get_gpt_response(model_string, prompt, temp, max_tokens, label_types)
                output['run_'+str(j)+"_probs"].append(probs)

            elif decode:
                response, probs = flan_generate(prompt, model, model_string,tokenizer,temp)
                output['run_'+str(j)+"_probs"].append(probs)
            else:
                response, topgen, topgenprobs = flan_generate_force_decode(prompt, model, tokenizer, temp, labels=label_types)
                output['run_'+str(j)+"_topgen"].append(topgen)
                output['run_'+str(j)+"_topgen_probs"].append(topgenprobs)
            if seen < 5:
                print(prompt)
                print(response)
            
            # documenting 
            if j == 0:
                output["Index"].append(i)
                output['labels'].append(label)
                if args.other_key:
                    output['aspect'].append(target)

            output['response_'+str(j)].append(response)
            seen += 1
    out_path = os.path.join(log_dir+log_file)
    os.makedirs(out_path)

    # write arguments to log file 
    with open(out_path+"/args.txt", "w") as f:
        for i in vars(args):
            f.write("{}\t{}\n".format(i, vars(args)[i]))
        if type(prompt) == str:
            f.write(prompt)
        elif type(prompt) == list:
            f.write(" ".join(str(x) for x in prompt))
    out_df = pd.DataFrame(output)
    
    out_df.to_csv(out_path+"/responses.csv")

if __name__ == "__main__":
    main()