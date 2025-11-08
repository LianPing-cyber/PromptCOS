import torch
import numpy as np

from embedding_tools import gradient_op
from bisect import bisect_right
from embedding_tools.optimization import optimize_prompt, optimize_query_vf, optimize_signal_mark

from copy import deepcopy

def merge_with_at(main_lis, score_lis, at_lis, n):
    m = len(at_lis)
    L = len(score_lis)
    n_eff = min(n, L)

    idx_small = sorted(range(L), key=lambda i: (score_lis[i], i))[:m]
    idx_large = sorted(range(L), key=lambda i: (-score_lis[i], i))[:n_eff]

    insert_points_sorted = sorted(idx_small)
    merge_list = list(main_lis)
    at_pos = [None] * m

    for t, ins_idx in enumerate(insert_points_sorted):
        pos = ins_idx + t           
        merge_list.insert(pos, at_lis[t])
        at_pos[t] = pos

    cv_pos = []
    for j in idx_large:
        shift = bisect_right(insert_points_sorted, j)
        cv_pos.append(j + shift)
        
    return merge_list, at_pos, cv_pos, idx_small, idx_large

def get_ct(input_ids, model, layers=(-1,), device=None):
    if device is None:
        device = getattr(model, "device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    else:
        input_ids = input_ids.to(dtype=torch.long)
    input_ids = input_ids.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,  
        )

    attentions = outputs.attentions
    seq_len = input_ids.size(1)
    scores = torch.zeros(seq_len, dtype=torch.float32)

    for layer in layers:
        attn = attentions[layer][0]                   
        attn = attn.to(dtype=torch.float32).cpu()        

        scores += attn.sum(dim=-1).sum(dim=0)

    return scores.tolist()

#w_prompts, query_vf, signal_mark = watermarking(o_prompt, tokens_at, tokens_cv, tokens_vf, tokens_sm, ss, top_k, max_epoch, p_f, p_d, model, tokenizer)
def prompt_token_merge(w_prompt, tokens_at, num_cv, model, tokenizer):
    # w_prompt will have elements: [prefix, prompt, infix, suffix, w_prompt_tokens, vf_query_tokens, signal_mark_tokens], 
    # the complete input for the llm is {prefix + prompt + infix + query + suffix}
    # However, prefix, infix, suffix are fixed form for instruction-tuning, so we only need to modify the prompt, that is:
    # Despite we consider the template content (including prefix, infix, suffix), we only recoginze the prompt part as the "real" system prompt
    pre_tokens = tokenizer.encode(w_prompt["prefix"], add_special_tokens=False)
    prompt_tokens = tokenizer.encode(w_prompt["prompt"], add_special_tokens=False)
    in_tokens = tokenizer.encode(w_prompt["infix"], add_special_tokens=False)

    importance_score_orders = get_ct(prompt_tokens, model)
    assert len(importance_score_orders) == len(prompt_tokens)
    w_prompt_tokens, at_pos, cv_pos, _, _ = merge_with_at(prompt_tokens, importance_score_orders, tokens_at, num_cv)
    
    # merge with pre_tokens and in_tokens; and then, adjust the at_pos and cv_pos
    w_prompt_tokens = pre_tokens + w_prompt_tokens + in_tokens
    at_pos = [i + len(pre_tokens) for i in at_pos]
    cv_pos = [i + len(pre_tokens) for i in cv_pos]

    w_prompt["w_prompt_tokens"] = w_prompt_tokens

    return w_prompt, at_pos, cv_pos


def watermarking(original_prompt, prompt_data, 
    tokens_at, num_cv, tokens_vf, tokens_sm, ss, top_k,
    max_epoch, p_f, p_d, 
    model, tokenizer, filter):
    
    np.random.seed(42)

    curr_epoch = 0

    w_prompt = original_prompt
    w_prompt["query_vf_tokens"] = tokens_vf
    w_prompt["signal_mark_tokens"] = tokens_sm

    # connect the tokens_at, tokens_cv with the original prompt, and build the initialized watermarked prompt
    w_prompt, at_pos_lis, cv_pos_lis = prompt_token_merge(w_prompt, tokens_at, num_cv, model, tokenizer)

    while curr_epoch < max_epoch:
        flags = [1,1,1]
        # 1. Optimize the tokens_at, tokens_cv, i.e., the tokens at at_pos_lis and cv_pos_lis in w_prompt

        print(">>>>>>>>>>>"*10)
        print(f"Epoch {curr_epoch}: No optimizing.")
        print(f"    w_prompt tokens: {w_prompt['w_prompt_tokens']}")
        print(f"    vf_query tokens: {tokenizer.decode(w_prompt['query_vf_tokens'])}")
        print(f"    signal_mark tokens: {tokenizer.decode(w_prompt['signal_mark_tokens'])}")

        o_w_prompt_tokens = deepcopy(w_prompt["w_prompt_tokens"])
        w_prompt = optimize_prompt(w_prompt, prompt_data, 
            at_pos_lis, cv_pos_lis,
            ss, top_k, p_f, p_d, model, tokenizer, filter)

        print("^^^^^"*10)
        print(f"Epoch {curr_epoch}: Optimized w_prompt.")
        print(f"    w_prompt tokens: {w_prompt['w_prompt_tokens']}")

        # If there is no change in the tokens_at, tokens_cv, then turn off the flag
        if w_prompt["w_prompt_tokens"] == o_w_prompt_tokens:
            print(f"    !!!!No change in the tokens_at, tokens_cv, turn off the flag!!!")
            flags[0] = 0
        
        # 2. Optimize the tokens_vf
        o_query_vf_tokens = deepcopy(w_prompt["query_vf_tokens"])
        w_prompt = optimize_query_vf(w_prompt,
            ss, top_k, model, tokenizer, filter)
        
        print("^^^^^"*10)
        print(f"Epoch {curr_epoch}: Optimized verification query.")
        print(f"    vf_query tokens: {tokenizer.decode(w_prompt['query_vf_tokens'])}")

        # If there is no change in the query_vf_tokens, then turn off the flag
        if w_prompt["query_vf_tokens"] == o_query_vf_tokens:
            print(f"    !!!!No change in the tokens_vf, turn off the flag!!!")
            flags[1] = 0
                
        # 3. Optimize the tokens_sm
        o_signal_mark_tokens = deepcopy(w_prompt["signal_mark_tokens"])
        w_prompt = optimize_signal_mark(w_prompt,
            ss, model, tokenizer)

        print("^^^^^"*10)
        print(f"Epoch {curr_epoch}: Optimized signal mark.")
        print(f"    signal_mark tokens: {tokenizer.decode(w_prompt['signal_mark_tokens'])}")

        # If there is no change in the signal_mark_tokens, then turn off the flag
        if w_prompt["signal_mark_tokens"] == o_signal_mark_tokens:
            print(f"    !!!!No change in the tokens_sm, turn off the flag!!!")
            flags[2] = 0

        # If all the flags are turned off, then break the loop
        if sum(flags) == 0:
            print("_______No change in the tokens_at, tokens_cv, tokens_vf, tokens_sm, break the loop_____!")
            break

        curr_epoch += 1 
    
    print(f"Watermark embedding finished, using epoch: {curr_epoch}")
    return w_prompt