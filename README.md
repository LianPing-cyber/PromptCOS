**Abstract**

The rapid progress of large language models (LLMs) has greatly enhanced reasoning tasks and facilitated the development of LLM-based applications. A critical factor in improving LLM-based applications is the design of effective system prompts, which significantly impact the behavior and output quality of LLMs. However, system prompts are susceptible to theft and misuse, which could undermine the interests of prompt owners. Existing methods protect prompt copyrights through watermark injection and verification but face challenges due to their reliance on intermediate LLM outputs (e.g., logits), which limits their practical feasibility.

In this paper, we propose PromptCOS, a method for auditing prompt copyright based on content-level output similarity. It embeds watermarks by optimizing the prompt while simultaneously co-optimizing a special verification query and content-level signal marks. This is achieved by leveraging cyclic output signals and injecting auxiliary tokens to ensure reliable auditing in content-only scenarios. Additionally, it incorporates cover tokens to protect the watermark from malicious deletion. For copyright verification, PromptCOS identifies unauthorized usage by comparing the similarity between the suspicious output and the signal mark. Experimental results demonstrate that our method achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% greater than the best baseline), high fidelity (accuracy degradation of no more than 0.58%), robustness (resilience against three types of potential attacks), and computational efficiency (up to 98.1% reduction in computational cost). 

**Main Files**

Use the main.py to embed the watermark into the prompts, optimizing the signal mark and verification query. The results will be saved in w_prompts folder.


Our method mainly employ the gradient greedy search to optimize the prompt. The core code realization of this method is included in tools/gradient_op.py.


**Order examples**

python -u main.py --M tinyllama --input_path prompts/gsm8k/ape/ --task_name all --dpn 5 --dpr 0.5 --hpr 0.5 --gn 4 --ss 1 --vf_length 5 --ij_length 3 --output_path w_prompts/gsm8k/ape/ --top_k 100 --ran_seed 100 


Before you start watermark embedding, you should edit the **paths.py** to give the right paths of models.

**Requirements**

transformers>=4.30.0; numpy>=1.20.0; pandas>=1.3.0; tqdm>=4.62.0; bitsandbytes>=0.41.0; scikit-learn>=1.0.0; matplotlib>=3.5.0; tensorboard>=2.10.0; accelerate>=0.20.0; datasets>=2.10.0; safetensors>=0.3.0; protobuf>=3.20.0; entencepiece>=0.1.97; ipykernel>=6.0.0; jupyter>=1.0.0; notebook>=6.4.0; fastapi>=0.104.1; uvicorn[standard]>=0.24.0; jinja2>=3.1.2; python-multipart>=0.0.6; aiofiles>=23.2.1; pydantic>=2.4.2; python-jose[cryptography]>=3.3.0; passlib[bcrypt]>=1.7.4

