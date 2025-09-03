# 📜 PromptCOS: Copyright Auditing for System Prompts via Content-Level Output Similarity

## 🧠 Abstract

The rapid advancement of large language models (LLMs) has significantly improved reasoning capabilities and enabled a wide range of LLM-based applications. A key component in enhancing these applications is the design of **effective system prompts**, which directly influence model behavior and output quality.

However, system prompts are vulnerable to theft and misuse, posing a risk to prompt creators. While existing methods protect prompt copyright via watermarking and verification, they often rely on intermediate outputs (e.g., logits), which limits their practicality.

We introduce **PromptCOS**, a novel method for **prompt copyright auditing** based solely on **content-level output similarity**. PromptCOS embeds watermarks by jointly optimizing the prompt, a verification query, and content-level signal marks. It uses **cyclic output signals** and **auxiliary token injection** for robust auditing in content-only scenarios, and **cover tokens** to defend against malicious watermark removal.

### 🔬 Key Features

- **Effectiveness**: Achieves **99.3% average watermark similarity**
- **Distinctiveness**: **60.8% improvement** over the best baseline
- **Fidelity**: Accuracy drop ≤ **0.58%**
- **Robustness**: Resilient to **three types of attacks**
- **Efficiency**: Reduces computation by up to **98.1%**

---

## 📁 Main Files

### 🔧 `main.py`

Use `main.py` to embed watermarks into prompts while optimizing signal marks and verification queries. Output results are saved in the `w_prompts/` directory.

Before using, configure model paths in **`paths.py`**.

### 🧪 Arguments for `main.py`

| Argument        | Type    | Description |
|----------------|---------|-------------|
| `--M`           | `str`   | Model name: `gemma`, `llama`, `chatglm` |
| `--input_path`  | `str`   | Path to input `.pkl` file (e.g., `NWQ{model}.pkl`) |
| `--task_name`   | `str`   | Task name |
| `--dpn`         | `int`   | Deletion penalty number |
| `--dpr`         | `float` | Deletion penalty rate |
| `--hpr`         | `float` | Harmfulness penalty rate |
| `--gn`          | `int`   | Number of generalization samples |
| `--ss`          | `int`   | Semantic strength |
| `--vf_length`   | `int`   | Verification watermark length |
| `--ij_length`   | `int`   | Injection watermark length |
| `--output_path` | `str`   | Path to output watermarked tasks |
| `--top_k`       | `int`   | Top-k for approximate search |
| `--ran_seed`    | `int`   | Random seed |

### 🧠 Core Method

Our watermarking process uses a **gradient greedy search** for prompt optimization. The core implementation is in `tools/gradient_op.py`.


---

## ▶️ Usage Example

```bash
python -u main.py \
  --M tinyllama \
  --input_path prompts/gsm8k/ape/ \
  --task_name all \
  --dpn 5 \
  --dpr 0.5 \
  --hpr 0.5 \
  --gn 4 \
  --ss 1 \
  --vf_length 5 \
  --ij_length 3 \
  --output_path w_prompts/gsm8k/ape/ \
  --top_k 100 \
  --ran_seed 100
```

## 📦 Requirements

Make sure to install the following dependencies:

```text
transformers>=4.30.0
numpy>=1.20.0
pandas>=1.3.0
tqdm>=4.62.0
bitsandbytes>=0.41.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
tensorboard>=2.10.0
accelerate>=0.20.0
datasets>=2.10.0
safetensors>=0.3.0
sentencepiece>=0.1.97
ipykernel>=6.0.0
jupyter>=1.0.0
notebook>=6.4.0
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
jinja2>=3.1.2
python-multipart>=0.0.6
aiofiles>=23.2.1
pydantic>=2.4.2
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
```
