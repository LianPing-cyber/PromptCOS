import torch

from modelscope import AutoModelForCausalLM, AutoTokenizer

class Model:
    def __init__(self, model_path):
        # Load pre-trained model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
        self.model.eval()

    def generate(self, inputs, temperature=1.0, top_p=0.9, max_new_tokens=256, batch_size=4):
        # Process inputs into batches
        batched_inputs = self.inputs_process(inputs, batch_size=batch_size)
        all_outputs = []
        
        # Iterate over batched inputs
        for batch in batched_inputs:
            with torch.no_grad():
                # Generate output tokens for the batch
                output_ids = self.model.generate(
                    **batch,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode outputs to text
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            all_outputs.extend(outputs)
        return all_outputs

    def inputs_process(self, inputs, batch_size=4):
        # Tokenize the inputs into input IDs
        tokenized_inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        
        # Split the tokenized inputs into batches
        batched_inputs = []
        num_batches = (len(inputs) + batch_size - 1) // batch_size  # This calculates the number of batches
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(inputs))
            
            # Slice the tokenized inputs and move them to the correct device (auto-device for model)
            batch = {key: value[start:end].to(self.model.device) for key, value in tokenized_inputs.items()}
            batched_inputs.append(batch)
        
        return batched_inputs
        
        
        