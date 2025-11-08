import torch

import os
import json
import numpy as np
import pandas as pd
import argparse
import pickle
import random
import heapq

from embedding_tools.watermark_embedding import watermarking


import time

from modelscope import AutoModelForCausalLM, AutoTokenizer


def components_init(vocab_size, num_at, num_vf, num_sm):
    tokens_at = random.sample(range(vocab_size), num_at)
    tokens_vf = random.sample(range(vocab_size), num_vf)
    tokens_sm = random.sample(range(vocab_size), num_sm)
    return tokens_at, tokens_vf, tokens_sm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model name or model path')
    parser.add_argument('--filter_path', type=str, required=False, default = "No", help='filter path if you want to use a filter')
    parser.add_argument("--original_prompts", type=str, required=True, help='prompts to be watermarked, a .json file')
    parser.add_argument("--prompt_train_data", type=str, required=True, help='data to optimize/evaluate the task performance of prompts, including input-output pairs, a .parquet file with columns: input, output')
    parser.add_argument('--num_fb', type=int, required=True, help='fidelity batch size, how many input-output pairs selected in each iteration to calculate the fidelity penalty')
    parser.add_argument('--num_at', type=int, required=True, help='number of auxiliary tokens, i.e., the number of tokens to be inserted into the original prompt')
    parser.add_argument('--num_cv', type=int, required=True, help='number of cover tokens, i.e., the number of tokens in the original prompt that can be optimized to cover the auxiliary tokens')
    parser.add_argument('--num_vf', type=int, required=True, help='number of verification tokens, i.e., how many tokens are used as verification tokens')
    parser.add_argument('--num_sm', type=int, required=True, help='number of signal mark tokens, i.e., how many tokens are used to form the signal mark')
    parser.add_argument('--ss', type=int, required=True, help='semantic strength of the signal mark, default is 2', default=2)
    parser.add_argument('--top_k', type=int, required=True, help='top k tokens to search for approximate optimal tokens during gradient search', default=100)
    parser.add_argument('--max_epoch', type=int, required=True, help='maximum number of optimization epochs, default is 10', default=50)
    parser.add_argument('--p_f', type=float, required=True, help='penalty factor for fidelity, default is 0.5', default=0.5)
    parser.add_argument('--p_d', type=float, required=True, help='penalty factor for deletion of auxiliary tokens, default is 0.5', default=0.5)
    parser.add_argument('--save_file', type=str, required=True, help='path to save the optimized prompts')
    args = parser.parse_args()

    time_start = time.time()

    model_path = args.model_path
    filter_path = args.filter_path
    original_prompts = args.original_prompts
    prompt_train_data = args.prompt_train_data
    num_fb = args.num_fb
    num_at = args.num_at
    num_cv = args.num_cv
    num_vf = args.num_vf
    num_sm = args.num_sm
    ss = args.ss
    top_k = args.top_k
    max_epoch = args.max_epoch
    p_f = args.p_f
    p_d = args.p_d
    save_file = args.save_file

    #load necessary data
    with open(original_prompts, 'r') as f:
        original_prompts = json.load(f)
    
    original_prompts = original_prompts[:2]
    #data should be regularized as json, a list, each element is a dict, with keys 'input' and 'output'
    prompt_data = json.load(open(prompt_train_data, 'r'))

    #load corresponding model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path,
        attn_implementation="eager", 
        dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    #load the token filter, which can pass through the strange words during the prompt building process
    filter = pickle.load(open(filter_path, "rb"))

    #initialize components: tokens_at, tokens_cv, tokens_vf, tokens_sm
    tokens_at, tokens_vf, tokens_sm = components_init(tokenizer.vocab_size, num_at, num_vf, num_sm)

    #start watermarking
    w_prompts = []

    for i in range(len(original_prompts)):
        w_prompt = watermarking(original_prompts[i], prompt_data, 
            tokens_at, num_cv, tokens_vf, tokens_sm, ss, top_k, 
            max_epoch, p_f, p_d, 
            model, tokenizer, filter)
        """
        about the w_prompt:
            w_prompt = {
                'prefix': general prefix, str, before the prompt,
                'infix': general infix, str, between prompt and query,
                'suffix': general suffix, str, after the query,
                'prompt': original prompt to be watermarked, str,
                'w_prompt': watermarked prompt, str,
                'query_vf': verification query, str,
                'signal_mark': signal_mark, str,
                "w_prompt_tokens": token ids of watermarked prompt, list, 
                "query_vf_tokens": token ids of verification query, list,
                "signal_mark_tokens": token ids of signal mark, list,
            }
        """
        w_prompt["w_prompt"] = tokenizer.decode(w_prompt["w_prompt_tokens"])
        w_prompt["query_vf"] = tokenizer.decode(w_prompt["query_vf_tokens"])
        w_prompt["signal_mark"] = tokenizer.decode(w_prompt["signal_mark_tokens"])
        w_prompts.append(w_prompt)

    #save the copyright components in .json format
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))

    with open(save_file, 'w') as f:
        json.dump(w_prompts, f, indent=4)

    time_end = time.time()
    print('Total time cost: {:.1f}min'.format((time_end - time_start) / 60))
