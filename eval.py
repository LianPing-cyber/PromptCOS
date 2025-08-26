import torch

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import numpy as np
import pandas as pd
import argparse
import pickle
from tqdm import tqdm
import re
import generate
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

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--M", type=str, default="gemma2-it")
    argparser.add_argument("--input_file", type=str, default="prompts/")
    #argparser.add_argument("--output_path", type=str, default="outputs/")
    argparser.add_argument("--batch_size", type=int, default=4)
    args = argparser.parse_args()

    if args.M not in args.input_file:
        print("Model Selection Error!!!!!")

    model, tokenizer = load_pretrained(args.M)

    tasks = pickle.load(open(args.input_file, "rb"))
    tasks = generate.NWQ(model, tokenizer, tasks, batch_size = args.batch_size)
    tasks = generate.NWV(model, tokenizer, tasks)
    tasks = generate.WV(model, tokenizer, tasks)
    tasks = generate.WQ(model, tokenizer, tasks, batch_size = args.batch_size)

    result_outputs = args.input_file.replace("w_prompts", "results") + "outputs.pkl"

    pickle.dump(tasks, open(result_outputs, "wb"))


