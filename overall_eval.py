import argparse
import json

from verification_tools.eval_function import eval_effectiveness, eval_fidelity, eval_robustness
from verification_tools.test_model import Model
from verification_tools.judge_model import JudgeModel

class TestContentSaver:
    def __init__(self):
        self.verify_watermark = None
        self.verify_normal = None
        self.query_watermark = None
        self.query_normal = None

    #each content is a list of strings
    def add_content(self, key = "verify_watermark", content = None):
        if key == "verify_watermark":
            self.verify_watermark = content
        elif key == "verify_normal":
            self.verify_normal = content
        elif key == "query_watermark":
            self.query_watermark = content
        elif key == "query_normal":
            self.query_normal = content



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='model name or model path')
    parser.add_argument("--prompts_file", type=str, required=True, help='prompts to be watermarked, a .json file')
    parser.add_argument("--prompt_train_data", type=str, required=True, help='data to optimize/evaluate the task performance of prompts, including input-output pairs, a .parquet file with columns: input, output')
    parser.add_argument("--eval_data_num", type=int, required=True, help='number of evaluation data samples to use')
    parser.add_argument("--output_file", type=str, required=True, help='file to save evaluation results')
    parser.add_argument('--temperature', type=float, required=False, help='temperature for generation', default=1.0)
    parser.add_argument('--top_p', type=float, required=False, help='top_p for generation', default=0.9)
    parser.add_argument('--max_new_tokens', type=int, required=False, help='max_new_tokens for generation', default=256)
    parser.add_argument('--batch_size', type=int, required=False, help='batch size for generation', default=4)
    #parser.add_argument('--generate_batch', type=int, required=True, help='number of generations to evaluate the optimized prompts', default=2)
    parser.add_argument('--if_effectiveness', type=bool, required=False, help='whether to evaluate effectiveness', default=False)
    parser.add_argument('--if_fidelity', type=bool, required=False, help='whether to evaluate fidelity', default=False)
    parser.add_argument('--if_robustness', type=bool, required=False, help='whether to evaluate robustness', default=False)
    parser.add_argument('--attack_type', type=str, required=False, help='attack type for robustness evaluation, e.g., noise', default="noise")
    parser.add_argument("--api_key", type=str, required=False, help='API key for judge model if needed', default=None)
    parser.add_argument("--base_url", type=str, required=False, help='Base URL for judge model if needed', default=None)
    args = parser.parse_args()

    model_path = args.model_path
    prompt_file = args.prompts_file
    prompt_train_data = args.prompt_train_data
    eval_data_num = args.eval_data_num

    if_effectiveness = args.if_effectiveness
    if_fidelity = args.if_fidelity
    if_robustness = args.if_robustness

    attack_type = args.attack_type

    model = Model(model_path)

    prompt_train_data = json.load(open(prompt_train_data, 'r'))

    w_prompts = json.load(open(prompt_file, 'r'))
    w_prompts = w_prompts[:2]
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

    content = TestContentSaver()

    results = {}

    if if_robustness or if_fidelity:
        classer = JudgeModel(api_key=args.api_key, base_url=args.base_url)

    if if_effectiveness:
        print("*"*10, "Evaluating Effectiveness", "*"*10)
        true_ws, false_ws, mdws = eval_effectiveness(w_prompts, model, content) 
        results["true_ws"] = true_ws
        results["false_ws"] = false_ws
        results["mdws"] = mdws

    if if_fidelity:
        print("*"*10, "Evaluating Fidelity", "*"*10)
        origin_score, curr_score, bert_score = eval_fidelity(w_prompts, prompt_train_data, model,
                                                    eval_data_num, content, classer)
        results["origin_score"] = origin_score
        results["curr_score"] = curr_score
        results["bert_score"] = bert_score

    if if_robustness:
        print("*"*10, "Evaluating Robustness", "*"*10)
        adv_true_ws, adv_score, adv_bert_score = eval_robustness(w_prompts, prompt_train_data, model,
                                                    eval_data_num, content, attack_type)
        results["adv_true_ws"] = adv_true_ws
        results["adv_score"] = adv_score
        results["adv_bert_score"] = adv_bert_score

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {args.output_file}")
