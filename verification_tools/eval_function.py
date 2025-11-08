from verification_tools.scores import score_ws, score_acc, score_bert_distance
from copy import deepcopy

def eval_effectiveness(w_prompts, model, content):
    # calculate this score need two contents:
    # 1. generate outputs for verification queries using watermarked prompts (verify_watermark)
    # 2. generate outputs for verification queries using normal prompts (verify_normal)
    # If these elements are not in content, we need to generate them first. 
    if content.verify_watermark is None:
        # Use the w_prompt as the final prompt to generate outputs for verification queries
        inputs = []
        for w_prompt in w_prompts:
            prompt = w_prompt['w_prompt']
            query = w_prompt['query_vf']
            suffix = w_prompt['suffix']
            full_input = prompt + query + suffix
            inputs.append(full_input)
        outputs = model.generate(inputs)
        outputs = delete_input(inputs, outputs)
        content.add_content("verify_watermark", outputs)

    if content.verify_normal is None:
        inputs = []
        for w_prompt in w_prompts:
            prompt = f"{w_prompt['prefix']}{w_prompt['prompt']}{w_prompt['infix']}"
            query = w_prompt['query_vf']
            suffix = w_prompt['suffix']
            full_input = prompt + query + suffix
            inputs.append(full_input)
        outputs = model.generate(inputs)
        outputs = delete_input(inputs, outputs)
        content.add_content("verify_normal", outputs)

    # Now we have both contents, we can calculate the scores
    marks = [w_prompt["signal_mark"] for w_prompt in w_prompts]
    true_ws, true_ws_max, true_ws_min = score_ws(content.verify_watermark, marks)

    print(marks)
    print(content.verify_watermark)
    print(content.verify_normal)

    false_ws, false_ws_max, false_ws_min = score_ws(content.verify_normal, marks)
    mdws = true_ws - false_ws_max
    return true_ws, false_ws, mdws


def eval_fidelity(w_prompts, prompt_train_data, model,
                    eval_data_num, content, classer):
    # first calculate the acc of original prompt.
    if content.query_normal is None:
        all_full_inputs = []
        eval_data = prompt_train_data[:eval_data_num]
        for w_prompt in w_prompts:
            prompt = f"{w_prompt['prefix']}{w_prompt['prompt']}{w_prompt['infix']}"
            suffix = w_prompt['suffix']
            full_inputs = [prompt + item['input'] + suffix for item in eval_data]
            all_full_inputs.extend(full_inputs)
        outputs = model.generate(all_full_inputs)
        content.add_content("query_normal", outputs)
    
    answers = content.query_normal
    references = [item['output'] for item in prompt_train_data[:eval_data_num]] * len(w_prompts)
    o_acc_score = score_acc(classer, answers, references)

    if content.query_watermark is None:
        all_full_inputs = []
        eval_data = prompt_train_data[:eval_data_num]
        for w_prompt in w_prompts:
            prompt = w_prompt['w_prompt']
            suffix = w_prompt['suffix']
            full_inputs = [prompt + item['input'] + suffix for item in eval_data]
            all_full_inputs.extend(full_inputs)
        outputs = model.generate(all_full_inputs)
        content.add_content("query_watermark", outputs)
    answers = content.query_watermark
    w_acc_score = score_acc(classer, answers, references)

    acc_degradation = w_acc_score - o_acc_score

    bert_score = score_bert_distance(content.query_normal, content.query_watermark)
    
    return acc_degradation, bert_score

def eval_robustness(w_prompt, prompt_train_data, model,
                    eval_data_num, content, attack_type):
    # first calculate the fi
    pass

def delete_input(inputs, outputs):
    result = []
    for input_str, output_str in zip(inputs, outputs):
        remaining_output = output_str.replace(input_str, "")
        result.append(remaining_output)
    return result