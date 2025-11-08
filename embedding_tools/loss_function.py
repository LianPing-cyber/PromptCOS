import torch
from copy import deepcopy

def get_loss(model, tokenizer, inputs, outputs, max_input_length, max_output_length):
    inputs = [sublist + [tokenizer.pad_token_id] * (max_input_length - len(sublist)) for sublist in inputs]
    outputs = [sublist + [tokenizer.pad_token_id] * (max_output_length - len(sublist)) for sublist in outputs]
    batch_input_ids = [subinput + suboutput for subinput, suboutput in zip(inputs, outputs)]
    batch_label_ids = [[-100]*max_input_length + suboutput for suboutput in outputs]
    batch_input_tensor = torch.tensor(batch_input_ids).to(model.device)
    batch_label_tensor = torch.tensor(batch_label_ids).to(model.device)
    loss = model(batch_input_tensor, labels=batch_label_tensor).loss
    return loss

def prompt_template(w_prompt, inputs, tokenizer):
    
    templated_inputs = []
    pre_tokens = w_prompt["w_prompt_tokens"]
    post_tokens = tokenizer.encode(w_prompt["suffix"], add_special_tokens=False)

    for input in inputs:
        in_tokens = input
        templated_input = pre_tokens + in_tokens + post_tokens
        templated_inputs.append(templated_input)

    return templated_inputs
        

def get_loss_prompt(w_prompt, at_pos_lis, inputs_fidelity, outputs_fidelity,
                    ss, p_f, p_d, model, tokenizer):
    # merge the tokens with 
    _inputs_eff_tokens = prompt_template(w_prompt, [w_prompt["query_vf_tokens"]], tokenizer)
    outputs_eff_tokens = [([tokenizer.pad_token_id] + w_prompt["signal_mark_tokens"]) * ss]

    # input/output in prompt_data are list of strings, we should encode them first
    inputs_fidelity_tokens = [tokenizer.encode(i, add_special_tokens=False) for i in inputs_fidelity]
    _inputs_fidelity_tokens = prompt_template(w_prompt, inputs_fidelity_tokens, tokenizer)
    outputs_fidelity_tokens = [tokenizer.encode(i, add_special_tokens=False) for i in outputs_fidelity]
    
    # penalty for deleting/disturbing at_tokens
    if p_d == 0:
        _inputs_robust_tokens = []
        outputs_robust_tokens = []
    else:
        noised_w_prompt = deepcopy(w_prompt)
        for i in range(len(at_pos_lis)):
            noised_w_prompt["w_prompt_tokens"][at_pos_lis[i]] = tokenizer.pad_token_id
        inputs_robust_tokens = [tokenizer.encode(i, add_special_tokens=False) for i in inputs_fidelity]
        _inputs_robust_tokens = prompt_template(noised_w_prompt, inputs_robust_tokens, tokenizer)
        outputs_robust_tokens = [tokenizer.encode(i, add_special_tokens=False) for i in outputs_fidelity]
    
    max_input_length = max(len(item) for item in (_inputs_eff_tokens + _inputs_fidelity_tokens + _inputs_robust_tokens))
    max_output_length = max(len(item) for item in (outputs_eff_tokens + outputs_fidelity_tokens + outputs_robust_tokens))

    loss_eff = get_loss(model, tokenizer, _inputs_eff_tokens, outputs_eff_tokens, max_input_length, max_output_length)
    loss_fidelity = get_loss(model, tokenizer, _inputs_fidelity_tokens, outputs_fidelity_tokens, max_input_length, max_output_length)
    if p_d == 0:
        return loss_eff + p_f * loss_fidelity
    else:
        loss_robust = get_loss(model, tokenizer, _inputs_robust_tokens, outputs_robust_tokens, max_input_length, max_output_length)
        return loss_eff + p_f * loss_fidelity - p_d * loss_robust

def get_loss_vf(w_prompt, ss, model, tokenizer):
    _inputs_eff_tokens = prompt_template(w_prompt, [w_prompt["query_vf_tokens"]], tokenizer)
    outputs_eff_tokens = [([tokenizer.pad_token_id] + w_prompt["signal_mark_tokens"]) * ss]

    max_input_length = max(len(item) for item in _inputs_eff_tokens)
    max_output_length = max(len(item) for item in outputs_eff_tokens)

    loss_eff = get_loss(model, tokenizer, _inputs_eff_tokens, outputs_eff_tokens, max_input_length, max_output_length)
    return loss_eff
