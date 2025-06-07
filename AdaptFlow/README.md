
# AdaptFlow: Adaptive Workflow Optimization via Meta-Learning

[![Project](https://img.shields.io/badge/Proj-AdaptFlow-blue)](https://zrc007.github.io/AdaptFlow_page/)
[![Paper](https://img.shields.io/badge/arXiv-2505.xxxxx-blue)](https://arxiv.org/abs/2505.xxxxx)

<div align="center">

📄 [Paper (Anonymous ACL Submission)](https://anonymous.4open.science/r/AdaptFlow-FD17/)

</div>

---

## 📌 Overview

**AdaptFlow** is a general-purpose **meta-optimization framework** that automates the construction of agentic workflows using large language models (LLMs). Unlike previous static or heuristic approaches, AdaptFlow introduces a **bi-level learning scheme** inspired by **Model-Agnostic Meta-Learning (MAML)**:

- 🧠 **Inner Loop:** Performs subtask-specific workflow refinement using LLM-generated textual feedback.
- 🌐 **Outer Loop:** Aggregates symbolic updates across subtasks into a generalizable initialization.

AdaptFlow supports **test-time adaptation** and operates in **non-differentiable code spaces** using only natural language feedback—enabling scalable and model-agnostic optimization.

---

## 💡 Key Features

- ✅ Bi-level optimization with symbolic gradients.
- 🔁 Task clustering and subtask-specific adaptation.
- 🧩 Modular design with interpretable workflows.
- ⚙️ Test-time adaptation for unseen tasks.
- 📈 State-of-the-art results across QA, code, and math reasoning benchmarks.

---

## 📊 Benchmarks

AdaptFlow is evaluated across **8 public benchmarks** in **3 domains**:

| Domain       | Datasets                                                  |
|--------------|-----------------------------------------------------------|
| QA           | HotpotQA, DROP                                            |
| Code         | HumanEval, MBPP                                           |
| Mathematics  | GSM8K, MATH, AIME, OlympiadBench                          |

See [Table 6 in the paper](https://anonymous.4open.science/r/AdaptFlow-FD17/) for detailed dataset statistics.

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://anonymous.4open.science/r/AdaptFlow-FD17/
cd AdaptFlow
pip install -r requirements.txt
```

### 2. Run Train and Evaluation

```bash

python {path_to_dataset}/search.py

```

### 3. Command-Line Arguments

| Argument            | Type   | Description                                                     |
| ------------------- | ------ | --------------------------------------------------------------- |
| `--save_dir`        | `str`  | Directory to store all output results.                          |
| `--expr_name`       | `str`  | Name for the current experiment run.                            |
| `--n_inner_loop`    | `int`  | Number of inner loop iterations (subtask-level updates).        |
| `--n_outer_loop`    | `int`  | Number of outer loop iterations (meta-level updates).           |
| `--version`         | `str`  | Version tag to distinguish different runs.                      |
| `--train_dir_path`  | `str`  | Path to the training (validation) dataset.                      |
| `--test_dir_path`   | `str`  | Path to the test dataset.                                       |
| `--model`           | `str`  | Backbone LLM used for executing workflows.                      |
| `--update_model`    | `str`  | LLM used for generating workflow updates.                       |


---
