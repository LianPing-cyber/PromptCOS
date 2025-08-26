import torch

import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import numpy as np
import pandas as pd
import argparse
import pickle
import random
import heapq
import tools.gradient_op as gradient_op
import tools.utils as utils
import paths
import generate
import time

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


#def get_candidates(averaged_grad, embeddings, pos_to_flip, top_k, filter):

def get_loss(model, tokenizer, inputs, outputs, w_prompt_ids,
            ij_token_ids, vf_token_ids, args):
    #loss1 for keeping the most similar output with original output
    input_ids, output_ids,_,_ = utils.template(tokenizer, w_prompt_ids, inputs, outputs)
    #get_max_length
    wm_input= [tokenizer.decode(vf_token_ids)]
    wm_output = [tokenizer.decode(ij_token_ids*args.ss)]
    wm_input_ids, wm_output_ids,_,_ = utils.template(tokenizer, w_prompt_ids, wm_input, wm_output)
    max_length = max([len(input_ids[i]) + len(output_ids[i]) for i in range(len(input_ids))])
    max_length = max([max_length, len(wm_input_ids[0]) + len(wm_output_ids[0])])
    input_tensor, label_tensor = utils.t2tmasked_batch(tokenizer, input_ids, output_ids, max_length)
    input_tensor, label_tensor = input_tensor.to("cuda"), label_tensor.to("cuda")
    loss1 = args.hpr*model(input_tensor, labels = label_tensor).loss / len(output_ids)
    #loss2 for watermark effectiveness
    wm_input_tensor, wm_output_tensor = utils.t2tmasked_batch(tokenizer, wm_input_ids, wm_output_ids, max_length)
    wm_input_tensor, wm_output_tensor = wm_input_tensor.to("cuda"), wm_output_tensor.to("cuda")
    loss2 = model(wm_input_tensor, labels = wm_output_tensor).loss
    loss = loss1 + loss2
    return loss

def get_vf_loss(model, tokenizer, w_prompt_ids,
            ij_token_ids, vf_token_ids, args):
    #loss2 for watermark effectiveness
    wm_input= [tokenizer.decode(vf_token_ids)]
    wm_output = [tokenizer.decode(ij_token_ids*args.ss)]
    wm_input_ids, wm_output_ids,_,_ = utils.template(tokenizer, w_prompt_ids, wm_input, wm_output)
    max_length = max([len(wm_input_ids[i]) + len(wm_output_ids[i]) for i in range(len(wm_input_ids))])
    wm_input_tensor, wm_output_tensor = utils.t2tmasked_batch(tokenizer, wm_input_ids, wm_output_ids, max_length)
    wm_input_tensor, wm_output_tensor = wm_input_tensor.to("cuda"), wm_output_tensor.to("cuda")
    loss2 = model(wm_input_tensor, labels = wm_output_tensor).loss
    return loss2


def choose_query_output(inputs, outputs, op_num):
    curr_num = np.random.randint(op_num)
    curr_query = inputs[curr_num]
    curr_output = outputs[curr_num]
    return curr_num, curr_query, curr_output

def get_best_candidate(candidates, model, tokenizer, another_tokens,
                        curr_inputs, curr_outputs, args, type="ij"):
    best_loss = 999999
    best_candidate = None
    for candidate in candidates:
        torch.cuda.empty_cache()
        if type == "ij":
            #get_loss(model, tokenizer, inputs, outputs, w_prompt_ids, ij_tokens, vf_tokens, args):
            loss = get_loss(model, tokenizer, curr_inputs, curr_outputs, candidate,
                        candidate[:args.ij_length], another_tokens,args)
        elif type == "vf":
            loss = get_vf_loss(model, tokenizer, another_tokens,
                        another_tokens[:args.ij_length], candidate, args)
        loss = loss.detach()
        if loss < best_loss:
            best_loss = loss
            best_candidate = candidate
    torch.cuda.empty_cache()
    return best_loss, best_candidate

def get_ct(model, tokenizer, input_ids, dpn, output = None, layers = [-1], device = "cuda"):
    input_ids = input_ids
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_ids, output_hidden_states=True, output_attentions=True)
    attentions = output.attentions
    scores = input_ids.size(1)*[0]
    for layer in layers:
        for head_idx in range(attentions[layer].size(1)):
            head_attention = attentions[layer][0, head_idx].cpu().numpy()
            for token_idx in range(head_attention.shape[0]):
                scores[token_idx] += sum(head_attention[token_idx])
    largest_n = heapq.nlargest(dpn, scores)
    indices = [i for i, x in enumerate(scores) if x in largest_n]
    return indices

def watermark(args, model, tokenizer, task):
    #find the important tokens' positons in prompt
    #initialize the injected tokens and vf tokens
    #loss1: WQ loss: input, output, prompt_position, _ = template(w_prompt, queries, outputs)
    #loss2: WQ penalty loss: input, output, _, vf_position = template(c_w_prompt, vf_tokens, output)
    #positions of flipped tokens = prompt_position: prompt_position + ij_length
    #loss3: WV loss: input, output, _, vf_position = template(w_prompt, vf_tokens, ij_tokens*sr)
    #positions of flipped tokens = prompt_position: prompt_position + ij_length + vf_position: vf_position + vf_length

    np.random.seed(args.ran_seed)
    top_k = args.top_k
    gn = args.gn
    
    if args.M == "deepseek":
        filter = []#pickle.load(open("filters/f_deepseek.pkl", "rb"))
    elif args.M == "tinyllama" or "llama32":
        filter = []#pickle.load(open("filters/f_llama.pkl", "rb"))
    
    ij_token_ids = np.random.randint(0, model.config.vocab_size, args.ij_length).tolist()
    vf_token_ids = np.random.randint(0, model.config.vocab_size, args.vf_length).tolist()

    embeddings = gradient_op.get_embeddings(model)
    embedding_gradient = gradient_op.GradientStorage(embeddings)    

    end_iter = False
    counter = 0

    prompt_ids = tokenizer.encode(task["prompt"], add_special_tokens=False)
    ct_pos_list = get_ct(model, tokenizer, prompt_ids, args.dpn)
    w_prompt_ids = ij_token_ids + prompt_ids

    for i in range(10):
        if end_iter:
            break
        print(i)
        curr_inputs = random.sample(task["inputs"], gn)
        curr_outputs = random.sample(task["targets"], gn)

        a, b, p_pos, v_pos = utils.template(tokenizer, w_prompt_ids, curr_inputs, curr_outputs)
        if i == 0:
            ct_pos_list = [x + p_pos + args.ij_length for x in ct_pos_list]
            ij_pos_flip = list(range(p_pos, p_pos + args.ij_length)) + ct_pos_list
            vf_pos_flip = list(range(v_pos, v_pos + args.vf_length))

        print(ij_pos_flip, vf_pos_flip)
        print(a, b)

        best_loss = get_loss(model, tokenizer, curr_inputs, curr_outputs, w_prompt_ids, 
            ij_token_ids, vf_token_ids, args)
        best_loss = best_loss.detach()
        
        print("Initial Loss: " + str(best_loss.data.item()))
        print("start ij pos optimization")
        for pos_to_flip in ij_pos_flip:
            if end_iter:
                break
            torch.cuda.empty_cache()
            model.zero_grad()

            #loss1
            loss = get_loss(model, tokenizer, curr_inputs, curr_outputs, w_prompt_ids, 
                ij_token_ids, vf_token_ids, args)
            loss.backward()

            grad = embedding_gradient.get()
            averaged_grad = torch.sum(grad, dim=0)
            averaged_grad = averaged_grad[pos_to_flip].unsqueeze(0)
            
            candidate_tokens = gradient_op.hotflip_attack(averaged_grad, embeddings.weight,
                                    increase_loss=False, num_candidates=top_k, filter = filter)

            if len(candidate_tokens) == 0:
                print("No candidate tokens, please give higher top_k")
                continue

            candidates = []
            for token in candidate_tokens:
                if token != w_prompt_ids[pos_to_flip - p_pos]:
                    candidate = deepcopy(w_prompt_ids)
                    candidate[pos_to_flip - p_pos] = token
                    candidates.append(candidate)

            curr_best_loss, best_candidate = get_best_candidate(candidates, model, tokenizer, vf_token_ids, 
                        curr_inputs, curr_outputs, args)
            
            if curr_best_loss < best_loss:
                counter = 0
                best_loss = curr_best_loss
                w_prompt_ids = deepcopy(best_candidate)
                ij_token_ids = w_prompt_ids[:args.ij_length]
                print("optimized")
            elif counter == len(ij_pos_flip) + len(vf_pos_flip):
                print(i,"\nNo improvement, ending iteration")
                end_iter = True
            else:
                counter = counter + 1
            print(pos_to_flip,"Loss: " + str(best_loss.data.item()))
            print(tokenizer.decode(w_prompt_ids) + '\n')
           
        if end_iter:
            break
        best_vf_loss = get_vf_loss(model, tokenizer, w_prompt_ids,
            ij_token_ids, vf_token_ids, args)
        best_vf_loss = best_vf_loss.detach()
        print("Initial VF Loss: " + str(best_vf_loss.data.item()))
        print("start vf pos optimization")
        for pos_to_flip in vf_pos_flip:
            torch.cuda.empty_cache()
            if end_iter:
                break
            model.zero_grad()
            loss = get_vf_loss(model, tokenizer, w_prompt_ids, 
                ij_token_ids, vf_token_ids, args)
            loss.backward()
            
            grad = embedding_gradient.get()
            averaged_grad = torch.sum(grad, dim=0)
            averaged_grad = averaged_grad[pos_to_flip].unsqueeze(0)

            candidate_tokens = gradient_op.hotflip_attack(averaged_grad, embeddings.weight,
                                    increase_loss=False, num_candidates=top_k, filter = filter)
            candidates = []
            for token in candidate_tokens:
                if token != vf_token_ids[pos_to_flip - v_pos]:
                    candidate = deepcopy(vf_token_ids)
                    candidate[pos_to_flip - v_pos] = token
                    candidates.append(candidate)

            curr_best_loss, best_candidate = get_best_candidate(candidates, model, tokenizer, w_prompt_ids,
                        curr_inputs, curr_outputs, args, type="vf")
            
            if curr_best_loss < best_vf_loss:
                counter = 0
                best_vf_loss = curr_best_loss
                vf_token_ids = deepcopy(best_candidate)
                print("optimized")
            elif counter == len(ij_pos_flip) + len(vf_pos_flip):
                print("\nNo improvement, ending iteration")
                end_iter = True
            else:
                counter = counter + 1
            print(pos_to_flip,"Loss: " + str(best_vf_loss.data.item()))
            print(tokenizer.decode(vf_token_ids) + '\n')
    task["w_prompt"] = tokenizer.decode(w_prompt_ids)
    task["vf_tokens"] = tokenizer.decode(vf_token_ids)
    task["signal"] = tokenizer.decode(ij_token_ids)
    return task


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--M', type=str, required=True, help='Test Model: gemma, llama, chatglm')
    parser.add_argument("--input_path", type=str, required=True, help='load NWQ{model name}.pkl from input_file_path')
    parser.add_argument("--task_name", type=str, required=True, help='task name')
    parser.add_argument("--dpn", type=int, required=True, help='deletion penalty number')
    parser.add_argument('--dpr', type=float, required=True, help='deletion penalty rate')
    parser.add_argument('--hpr', type=float, required=True, help='harmfulness penalty rate')
    parser.add_argument('--gn', type=int, required=True, help='generlization samples number')
    parser.add_argument('--ss', type=int, required=True, help='semantic strength')
    parser.add_argument('--vf_length', type=int, required=True, help='vf watermark length')
    parser.add_argument('--ij_length', type=int, required=True, help='ij watermark length')
    parser.add_argument('--output_path', type=str, required=True, help='output_path of w_tasks')
    parser.add_argument('--top_k', type=int, required=True, help='top k for appromixly searching')
    parser.add_argument('--ran_seed', type=int, required=True, help='output_path')

    args = parser.parse_args()
    model, tokenizer = load_pretrained(args.M)
    
    tasks = pickle.load(open(args.input_path + "tasks.pkl", "rb"))
    #tasks = generate.NWQ(model, tokenizer, tasks, 32)
    #pickle.dump(tasks, open(args.input_path + args.M + "NWQ.pkl", "wb"))

    #tasks = pickle.load(open(args.input_path + args.M + "NWQ.pkl", "rb"))

    start = time.perf_counter()

    if args.task_name == "all":
        for name in tasks.keys():
            print("*"*10 + name + "*"*10)
            w_task = watermark(args, model, tokenizer, tasks[name])
            tasks[name] = w_task
            pickle.dump(tasks, open(f"{args.output_path}{args.M}_{args.task_name}_{args.ss}_{args.dpn}_{args.dpr}_{args.hpr}_{args.ij_length}_{args.vf_length}_{args.top_k}_{args.gn}_{args.ran_seed}", 'wb'))
    else:
        w_task = watermark(args, model, tokenizer, tasks[args.task_name])
        tasks[args.task_name] = w_task
        print(tasks[args.task_name])

    end = time.perf_counter()
    print("Time: ", end - start)