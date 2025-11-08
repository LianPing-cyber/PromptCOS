import random
import torch

from embedding_tools import gradient_op
from embedding_tools.loss_function import get_loss_prompt, get_loss_vf

from copy import deepcopy

def select_prompt_data(prompt_data, batch = 4):
    chosen = random.sample(prompt_data, batch)
    inputs = [elem["input"] for elem in chosen]
    outputs = [elem["output"] for elem in chosen]
    return inputs, outputs

def optimize_prompt(w_prompt, prompt_data, 
            at_pos_lis, cv_pos_lis,
            ss, top_k, p_f, p_d, model, tokenizer, filter):
    # start a epoch with pos_for_optimization
    # Now w_prompt has elements: [prefix, prompt, infix, suffix, w_prompt_tokens, vf_query_tokens, signal_mark_tokens], the complete input for the llm is
    # The w_prompt_tokens has included the prefix and infix
    # The right input to llm is [w_prompt_tokens  + query(tokens)+ suffix(tokens)]
    embeddings = gradient_op.get_embeddings(model)
    embedding_gradient = gradient_op.GradientStorage(embeddings)
    
    # 1. Random select the data from prompt_data to evaluate the fidelity
    inputs_fidelity, outputs_fidelity = select_prompt_data(prompt_data)
    # 2. Get the best loss now, during at optimization period, mask the p_d
    best_loss = get_loss_prompt(w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity,
                ss, p_f, 0, model, tokenizer)
    best_loss = best_loss.detach().cpu().numpy()

    print("-----"*5)
    print(f"start an optimization for auxiliary tokens (at) in w_prompt.")
    print("Initial Loss After Random Sample: " + str(best_loss))
    print("-----"*5)

    for pos in at_pos_lis:
        model.zero_grad(set_to_none=True)     
        embedding_gradient.clear()  
        loss = get_loss_prompt(w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity,
                ss, p_f, 0, model, tokenizer)
        loss.backward()
        grad = embedding_gradient.get()
        averaged_grad = torch.sum(grad, dim=0)
        averaged_grad = averaged_grad[pos].unsqueeze(0)

        candidate_tokens = gradient_op.hotflip_attack(averaged_grad, embeddings.weight,
                                    increase_loss=False, num_candidates=top_k, filter = filter)

        if len(candidate_tokens) == 0:
            print(f"No candidate tokens at position {pos}, please give higher top_k")
            continue
        
        o_token = w_prompt["w_prompt_tokens"][pos]
        curr_best_loss, curr_best_token = choose_best_w_prompt(
            w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity, ss, p_f, 0, model, tokenizer,
            candidate_tokens, pos
        )

        if curr_best_loss < best_loss:
            best_loss = curr_best_loss
            w_prompt["w_prompt_tokens"][pos] = curr_best_token
            print(f"Update the best loss to {best_loss} at position {pos}. Replace token {o_token} with token {curr_best_token}")
        else:
            print(f"Keep the best loss at position {pos}.")
    #Start to optimize the cover tokens
    best_loss = get_loss_prompt(w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity,
                ss, p_f, p_d, model, tokenizer)
    best_loss = best_loss.detach().cpu().numpy()

    print("+++++"*5)
    print(f"start an optimization for cover tokens (cv) in w_prompt.")
    print("Initial Loss After Random Sample: " + str(best_loss))
    print("+++++"*5)
    for pos in cv_pos_lis:
        model.zero_grad(set_to_none=True)     
        embedding_gradient.clear()  
        loss = get_loss_prompt(w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity,
                ss, p_f, p_d, model, tokenizer)
        loss.backward()
        grad = embedding_gradient.get()
        averaged_grad = torch.sum(grad, dim=0)
        averaged_grad = averaged_grad[pos].unsqueeze(0)
        
        candidate_tokens = gradient_op.hotflip_attack(averaged_grad, embeddings.weight,
                                    increase_loss=True, num_candidates=top_k, filter = filter)
        if len(candidate_tokens) == 0:
            print(f"No candidate tokens at position {pos}, please give higher top_k")
            continue
        
        o_token = w_prompt["w_prompt_tokens"][pos]
        curr_best_loss, curr_best_token = choose_best_w_prompt(
            w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity, ss, p_f, p_d, model, tokenizer,
            candidate_tokens, pos
        )
        if curr_best_loss < best_loss:
            best_loss = curr_best_loss
            w_prompt["w_prompt_tokens"][pos] = curr_best_token
            print(f"Update the best loss to {best_loss} at position {pos}. Replace token {o_token} with token {curr_best_token}")
        else:
            print(f"Keep the best loss at position {pos}.")
    return w_prompt

def optimize_query_vf(w_prompt, ss, top_k, model, tokenizer, filter):
    embeddings = gradient_op.get_embeddings(model)
    embedding_gradient = gradient_op.GradientStorage(embeddings)
    
    # 1. query_vf doesn't need to consider fidelity, deletion_penalty
    # 2. Get the best loss now, during at optimization period, mask the p_d
    best_loss = get_loss_vf(w_prompt, ss, model, tokenizer)
    best_loss = best_loss.detach().cpu().numpy()

    print("-----"*5)
    print(f"start an optimization for tokens in verification query (query_vf).")
    print("Initial Loss For Effectiveness: " + str(best_loss))
    print("-----"*5)

    start_pos = len(w_prompt["w_prompt_tokens"])

    for pos in range(len(w_prompt["query_vf_tokens"])):
        model.zero_grad(set_to_none=True)
        embedding_gradient.clear()  
        loss = get_loss_vf(w_prompt, ss, model, tokenizer)
        loss.backward()
        grad = embedding_gradient.get()
        averaged_grad = torch.sum(grad, dim=0)
        averaged_grad = averaged_grad[pos + start_pos].unsqueeze(0)

        candidate_tokens = gradient_op.hotflip_attack(averaged_grad, embeddings.weight,
                                    increase_loss=False, num_candidates=top_k, filter = filter)

        if len(candidate_tokens) == 0:
            print(f"No candidate tokens at position {pos}, please give higher top_k")
            continue

        curr_best_loss, curr_best_token = choose_best_query_vf(
            w_prompt, ss, model, tokenizer, candidate_tokens, pos)

        o_token = w_prompt["query_vf_tokens"][pos]
        if curr_best_loss < best_loss:
            best_loss = curr_best_loss
            w_prompt["query_vf_tokens"][pos] = curr_best_token
            print(f"Update the best loss to {best_loss} at position {pos}.  Replace token {o_token} with token {curr_best_token}")
        else:
            print(f"Keep the best loss at position {pos}.")

    return w_prompt

def optimize_signal_mark(w_prompt, ss, model, tokenizer):
    o_signal_mark_list = []
    candidate_tokens = get_signal_mark_candidates(w_prompt, tokenizer)
    flag = True
    best_loss = get_loss_vf(w_prompt, ss, model, tokenizer)
    best_loss = best_loss.detach().cpu().numpy()
    print("-----"*5)
    print(f"start an optimization for tokens in signal mark (sm).")
    print("Initial Loss For Effectiveness: " + str(best_loss))
    print("-----"*5)
    while flag:
        for pos in range(len(w_prompt["signal_mark_tokens"])):
            o_token = w_prompt["signal_mark_tokens"][pos]
            best_loss, best_token = choose_best_signal_mark(w_prompt, ss, model, tokenizer, candidate_tokens, pos)
            print(f"Update the best loss to {best_loss} at position {pos}.  Replace token {o_token} with token {best_token}")
            w_prompt["signal_mark_tokens"][pos] = best_token
            if w_prompt["signal_mark_tokens"] in o_signal_mark_list:
                flag = False
                break
            o_signal_mark_list.append(w_prompt["signal_mark_tokens"])
    return w_prompt

def choose_best_signal_mark(w_prompt, ss, model, tokenizer, candidate_tokens, pos):
    c_w_prompt = deepcopy(w_prompt)
    best_loss = get_loss_vf(w_prompt, ss, model, tokenizer)
    best_loss = best_loss.detach().cpu().numpy()
    best_token = w_prompt["signal_mark_tokens"][pos]
    for token in candidate_tokens:
        c_w_prompt["signal_mark_tokens"][pos] = token
        curr_loss = get_loss_vf(c_w_prompt, ss, model, tokenizer)
        curr_loss = curr_loss.detach().cpu().numpy()
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_token = token
    return best_loss, best_token

def get_signal_mark_candidates(w_prompt, tokenizer):
    candidates = []
    pre_length = len(tokenizer.encode(w_prompt["prefix"], add_special_tokens=False))
    in_length = len(tokenizer.encode(w_prompt["infix"], add_special_tokens=False))
    real_w_prompt_tokens = w_prompt["w_prompt_tokens"][pre_length:][:in_length]
    for token in real_w_prompt_tokens:
        if token not in candidates:
            candidates.append(token)
    for token in candidates:
        if token in w_prompt["query_vf_tokens"]:
            candidates.remove(token)
        #move the short tokens
        if len(tokenizer.decode(token)) < 3:
            candidates.remove(token)
    return candidates

def choose_best_query_vf(w_prompt, ss, model, tokenizer, candidate_tokens, pos):
    c_w_prompt = deepcopy(w_prompt)
    best_loss = 10000000
    best_token = None
    for token in candidate_tokens:
        c_w_prompt["query_vf_tokens"][pos] = token
        curr_loss = get_loss_vf(c_w_prompt, ss, model, tokenizer)
        curr_loss = curr_loss.detach().cpu().numpy()
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_token = token
    if best_token is None:
        print(f"error, error, error, error in choose_best_query_vf")
    return best_loss, best_token
        
def choose_best_w_prompt(w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity,
                    ss, p_f, p_d, model, tokenizer, candidate_tokens, pos):
    c_w_prompt = deepcopy(w_prompt)
    best_loss = 10000000
    best_token = None
    for token in candidate_tokens:
        c_w_prompt["w_prompt_tokens"][pos] = token
        curr_loss = get_loss_prompt(c_w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity,
                    ss, p_f, p_d, model, tokenizer)
        curr_loss = curr_loss.detach().cpu().numpy()
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_token = token
    if best_token is None:
        print(f"error, error, error, error in choose_best_w_prompt")
    return best_loss, best_token


