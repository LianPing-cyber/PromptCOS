**Order examples**

python -u main.py --M tinyllama --input_path prompts/gsm8k/ape/ --task_name all --dpn 5 --dpr 0.5 --hpr 0.5 --gn 4 --ss 1 --vf_length 5 --ij_length 3 --output_path w_prompts/gsm8k/ape/ --top_k 100 --ran_seed 100 


Before you start watermark embedding, you should edit the **paths.py** to give the right paths of models.


**Requirements**

transformers>=4.30.0; numpy>=1.20.0; pandas>=1.3.0; tqdm>=4.62.0; bitsandbytes>=0.41.0; scikit-learn>=1.0.0; matplotlib>=3.5.0; tensorboard>=2.10.0; accelerate>=0.20.0; datasets>=2.10.0; safetensors>=0.3.0; protobuf>=3.20.0; entencepiece>=0.1.97; ipykernel>=6.0.0; jupyter>=1.0.0; notebook>=6.4.0; fastapi>=0.104.1; uvicorn[standard]>=0.24.0; jinja2>=3.1.2; python-multipart>=0.0.6; aiofiles>=23.2.1; pydantic>=2.4.2; python-jose[cryptography]>=3.3.0; passlib[bcrypt]>=1.7.4

