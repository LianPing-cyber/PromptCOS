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

Our watermarking process uses a **gradient greedy search** for prompt optimization. The core implementation is in:

