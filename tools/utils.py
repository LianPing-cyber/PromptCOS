import numpy as np
import torch

def random_tokens_init(tokenizer, num_tokens, vocab_size):
    token_ids = np.random.randint(0, vocab_size, num_tokens)
    tokens = tokenizer.decode(token_ids)
    return token_ids, tokens

#template to construct the query
def query_template(query_text, prompt_text):
    input_text = f"Input:{prompt_text} {query_text}. Output: "
    return input_text

#template

def template(tokenizer, prompt_ids, queries, outputs):
    prefix = "System: "
    infix = " Input: "
    suffix = " Output: "
    prefix_token_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    infix_token_ids = tokenizer.encode(infix, add_special_tokens=False)
    suffix_token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    prompt_token_ids = prompt_ids
    prompt_position = len(prefix_token_ids)
    vf_position = len(prefix_token_ids + prompt_token_ids + infix_token_ids)
    inputs = []
    targets = []
    for query, output in zip(queries, outputs):
        input_token_ids = prefix_token_ids + prompt_token_ids + infix_token_ids + tokenizer.encode(query, add_special_tokens=False) + suffix_token_ids
        target_token_ids = tokenizer.encode(output, add_special_tokens=False)
        inputs.append(input_token_ids)
        targets.append(target_token_ids)
    return inputs, targets, prompt_position, vf_position
        
    
#make masked tensor from token i
def t2tmasked_batch(tokenizer, input_ids_batch, output_ids_batch, max_length):
    #pad tokenizer.pad_token_id between the input_ids and output_ids
    input_tensor_batch = torch.stack([
        torch.cat([torch.tensor(input_ids), torch.full((max_length - len(input_ids) - len(output_ids),), tokenizer.pad_token_id), torch.tensor(output_ids)]) 
        for input_ids, output_ids in zip(input_ids_batch, output_ids_batch)])
    # use [-100] to mask the tokens before the output_ids
    label_tensor_batch = torch.stack([
        torch.cat([torch.full((max_length - len(output_ids),), -100), torch.tensor(output_ids)]) 
        for input_ids, output_ids in zip(input_ids_batch, output_ids_batch)])
    return input_tensor_batch, label_tensor_batch


'''
def make_tensor(tokenizer, ij_tokens, vf_tokens, 
                prompt, query, output, type="normal"):
    input, target = template(ij_tokens, vf_tokens, 
                        prompt, query, output, type)
    whole_input = input + target
    input_token_ids = tokenizer.encode(whole_input)
    target_token_ids = tokenizer.encode(target, add_special_tokens=False)
    label_token_ids = [-100]*(len(input_token_ids) - len(target_token_ids)) + target_token_ids
    input_tensors = torch.tensor(input_token_ids, dtype=torch.long).to("cuda:0").unsqueeze(0)
    label_tensors = torch.tensor(label_token_ids, dtype=torch.long).to("cuda:0").unsqueeze(0)
    return input_tensors, label_tensors
'''
    
def get_results(model, tokenizer, ij_tokens, vf_tokens, 
                prompt, query, output):
    normal_inputs, normal_labels = make_tensor(tokenizer, ij_tokens, vf_tokens, 
                prompt, query, output, "normal")
    vf_inputs, vf_labels = make_tensor(tokenizer, ij_tokens, vf_tokens,
                prompt, query, output, "vf")
    
    normal_loss = model.forward(normal_inputs, labels = normal_labels).loss
    vf_loss = model.forward(vf_inputs, labels = vf_labels).loss
    normal_loss = normal_loss.detach().cpu().numpy()
    vf_loss = vf_loss.detach().cpu().numpy()
    #get the loss of the model

    normal_query = tokenizer(f"Input: {ij_tokens} {prompt} {query}. Output:", return_tensors="pt", padding=True).to("cuda:0")
    normal_query_1 = tokenizer(f"Input: {prompt} {query}. Output:", return_tensors="pt", padding=True).to("cuda:0")
    vf_query = tokenizer(f"Input: {ij_tokens} {prompt} {vf_tokens}. Output:", return_tensors="pt", padding=True).to("cuda:0")
    vf_query_1 = tokenizer(f"Input: {prompt} {vf_tokens}. Output:", return_tensors="pt", padding=True).to("cuda:0")
    normal_query_length = len(tokenizer.encode(f"{ij_tokens} {prompt} {query}"))
    vf_query_length = len(tokenizer.encode(f"{ij_tokens} {prompt} {vf_tokens}"))

    normal_output_ids = model.generate(**normal_query, max_new_tokens = 100)[0]
    normal_output_ids_1 = model.generate(**normal_query_1, max_new_tokens = 100)[0]
    vf_output_ids = model.generate(**vf_query, max_new_tokens = 100)[0]
    vf_output_ids_1 = model.generate(**vf_query_1, max_new_tokens = 100)[0]

    normal_output = tokenizer.decode(normal_output_ids)
    normal_output_1 = tokenizer.decode(normal_output_ids_1)
    vf_output = tokenizer.decode(vf_output_ids)
    vf_output_1 = tokenizer.decode(vf_output_ids_1)

    return [normal_loss, normal_output, normal_output_1, vf_loss, vf_output, vf_output_1]

def generate(model, tokenizer, input_text, type = "all", max_new_tokens = 100):
    input_ids = tokenizer(input_text, return_tensors="pt", padding=True).to("cuda:0")
    with torch.no_grad():
        outputs = model.generate(**input_ids, max_new_tokens = max_new_tokens)
    output_text = tokenizer.decode(outputs[0])
    if type == "all":
        return output_text
    elif type == "new":
        new_loc = output_text.find(input_text)
        output_text = output_text[len(input_text)+ new_loc:]
        return output_text