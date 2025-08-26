import torch
import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
import re

from tools import score, utils
import paths

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from copy import deepcopy

def load_pretrained(model_name):
    if model_name == "gemma2":
        model_name = paths.gemma_2b
    elif model_name == "gemma2-it":
        model_name = paths.gemma2_2b_it
    elif model_name == "tinyllama":
        model_name = paths.tinyllama
    elif model_name == "llama32":
        model_name = paths.llama32
    elif model_name == "deepseek":
        model_name = paths.deepseek_d_qwen
    elif model_name == "codegemma2b":
        model_name = paths.codegemma2b
    elif model_name == "codegemma7b":
        model_name = paths.codegemma7b
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16  
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, 
                        quantization_config = bnb_config, device_map = "auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, device_map = "auto")
    if tokenizer.pad_token_id < model.config.vocab_size:
        print(tokenizer.pad_token_id) 
    return model, tokenizer

def batch2tensor(tokenizer, input_ids_batch):
    input_ids_batch = [torch.tensor(input_ids) for input_ids in input_ids_batch]
    max_length = max([seq.size(0) for seq in input_ids_batch])
    input_tensor_batch = torch.stack([
    torch.cat([torch.full((max_length - seq.size(0),), tokenizer.pad_token_id), seq]) 
        for seq in input_ids_batch])
    attention_mask_batch = torch.where(input_tensor_batch == tokenizer.pad_token_id, torch.tensor(0), torch.tensor(1))
    return input_tensor_batch, attention_mask_batch

def get_new_output(outputs):
    new_outputs = []
    for i in range(len(outputs)):
        new_output = re.findall(r"Output: ([\s\S]*)", outputs[i])
        new_outputs.append(new_output[0])
    return new_outputs

def generate(model, tokenizer, tasks, batch_size = 4):
    prompt_ids = tokenizer.encode(tasks["prompt"])
    inputs, _, _ = utils.template(tokenizer, prompt_ids, tasks["inputs"], tasks["targets"])
    batch_num = (len(inputs) + batch_size - 1) // batch_size
    outputs = []
    for batch_idx in tqdm(range(batch_num)):
        #clean the cache
        torch.cuda.empty_cache()
        batch_inputs = inputs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        input_tensor_batch, attention_mask_batch = batch2tensor(tokenizer, batch_inputs)
        batch_outputs = model.generate(input_ids=input_tensor_batch, attention_mask=attention_mask_batch, 
                    max_new_tokens=200, do_sample=True, top_p=0.7, temperature=0.95)
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        outputs.extend(batch_outputs)
    outputs = get_new_output(outputs)
    tasks["outputs"] = outputs
    return tasks

def NWQ(model, tokenizer, tasks, batch_size = 4):
    subtasks = tasks.keys()
    for subtask in subtasks:
        print(subtask)
        prompt_ids = tokenizer.encode(tasks[subtask]["prompt"])
        inputs, _, _, _ = utils.template(tokenizer, prompt_ids, tasks[subtask]["inputs"], tasks[subtask]["targets"])
        batch_num = (len(inputs) + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in tqdm(range(batch_num)):
            #clean the cache
            torch.cuda.empty_cache()
            batch_inputs = inputs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            input_tensor_batch, attention_mask_batch = batch2tensor(tokenizer, batch_inputs)
            input_tensor_batch = input_tensor_batch.to("cuda")
            attention_mask_batch = attention_mask_batch.to("cuda")
            batch_outputs = model.generate(input_ids=input_tensor_batch, attention_mask=attention_mask_batch, 
                        max_new_tokens=200, do_sample=True, top_p=0.7, temperature=0.95)
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            outputs.extend(batch_outputs)
        outputs = get_new_output(outputs)
        tasks[subtask]["NWQ"] = outputs
    return tasks
    
def NWV(model, tokenizer, tasks):
    subtasks = tasks.keys()
    for subtask in subtasks:
        print(subtask)
        prompt_ids = tokenizer.encode(tasks[subtask]["prompt"])
        if "vf_tokens" not in tasks[subtask].keys():
            print("subtask: ", subtask, "has no vf_tokens")
            continue
        inputs, _, _, _ = utils.template(tokenizer, prompt_ids, [tasks[subtask]["vf_tokens"]], [tasks[subtask]["signal"]])
        input_tensor, attention_mask = batch2tensor(tokenizer, inputs)
        input_tensor = input_tensor.to("cuda")
        attention_mask = attention_mask.to("cuda")
        output_ids = model.generate(input_ids=input_tensor, attention_mask=attention_mask, 
                    max_new_tokens=200, do_sample=True, top_p=0.7, temperature=0.95)
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output = get_new_output(output)[0]
        tasks[subtask]["NWV"] = output
    return tasks

def WQ(model, tokenizer, tasks, batch_size = 4):
    subtasks = tasks.keys()
    for subtask in subtasks:
        print(subtask)
        if "w_prompt" not in tasks[subtask].keys():
            print("subtask: ", subtask, "has no w_prompt")
            continue
        prompt_ids = tokenizer.encode(tasks[subtask]["w_prompt"])
        inputs, _, _, _ = utils.template(tokenizer, prompt_ids, tasks[subtask]["inputs"], tasks[subtask]["targets"])
        batch_num = (len(inputs) + batch_size - 1) // batch_size
        outputs = []
        for batch_idx in tqdm(range(batch_num)):
            #clean the cache
            torch.cuda.empty_cache()
            batch_inputs = inputs[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            input_tensor_batch, attention_mask_batch = batch2tensor(tokenizer, batch_inputs)
            input_tensor_batch = input_tensor_batch.to("cuda")
            attention_mask_batch = attention_mask_batch.to("cuda")
            batch_outputs = model.generate(input_ids=input_tensor_batch, attention_mask=attention_mask_batch, 
                        max_new_tokens=200, do_sample=True, top_p=0.7, temperature=0.95)
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            outputs.extend(batch_outputs)
        outputs = get_new_output(outputs)
        tasks[subtask]["WQ"] = outputs
    return tasks

def WV(model, tokenizer, tasks):
    subtasks = tasks.keys()
    for subtask in subtasks:
        print(subtask)
        if "w_prompt" not in tasks[subtask].keys():
            print("subtask: ", subtask, "has no w_prompt")
            continue
        prompt_ids = tokenizer.encode(tasks[subtask]["w_prompt"])
        inputs, _, _, _ = utils.template(tokenizer, prompt_ids, [tasks[subtask]["vf_tokens"]], [tasks[subtask]["signal"]])
        input_tensor, attention_mask = batch2tensor(tokenizer, inputs)
        input_tensor = input_tensor.to("cuda")
        attention_mask = attention_mask.to("cuda")
        output_ids = model.generate(input_ids=input_tensor, attention_mask=attention_mask, 
                    max_new_tokens=200, do_sample=True, top_p=0.7, temperature=0.95)
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output = get_new_output(output)[0]
        tasks[subtask]["WV"] = output
    return tasks

'''get model_name and task_name from command，model_name contains: "tinyllama", "llama-chat", "gemma", "gemma2-it"
task_name contains: "NWQ", "NWV", "WQ", "WV"

then load the pretrained model through model_name,

then construct four task_functions respectively for "NWQ", "NWV", "WQ", "WV"

each function has similar structure: NW means no watermark, W means watermark, Q means query, V means vf_tokens
the function will generate the output 
'''

'''tasks is a dictionary which has many subtasks, each subtask has a dictionary which contains "prompt", "inputs", "targets"; 
they should first get NWQ, and using main.py to get "w_prompt", "vf_tokens" and "signal", then get NWV, WQ, WV to calculate the scores

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_name", type=str, default="gemma2-it")
    argparser.add_argument("--input_path", type=str, default="prompts/")
    argparser.add_argument("--batch_size", type=int, default=4)
    argparser.add_argument("--task_type", type=str, default="outputs/")
    args = argparser.parse_args()

    model, tokenizer = load_pretrained(args.model_name)

    if 

    tasks = pickle.load(open(args.input_path + "tasks.pkl", "rb"))
    tasks = NWQ(args, model, tokenizer, tasks, args.batch_size)
    tasks = NWV(args, model, tokenizer, tasks, args.batch_size)
    tasks = WQ(args, model, tokenizer, tasks, args.batch_size)
    tasks = WV(args, model, tokenizer, tasks, args.batch_size)

    scores = {}
    for subtask in tasks.keys():
        score_harmfulness = score.bleu([tasks[subtask]["NWQ"]], tasks[subtask]["WQ"])
        score_accuracy = score.avg_window_bleu([tasks[subtask]["signal"]], tasks[subtask]["WV"])
        score_diff = score.avg_window_bleu([tasks[subtask]["signal"]], tasks[subtask]["NWV"])
        scores[subtask] = {"harmfulness":score_harmfulness, "accuracy":score_accuracy, "diff":score_diff}
    
    out_path = args.input_path
'''