

# A Tale of Two Graphs: Separating Knowledge Exploration from Outline Structure for Open-Ended Deep Research

**[中文版 README](README_ZH.md)**

[![Project](http://img.shields.io/badge/Project-DualGraph-E3E4C8.svg)](https://microsoft.github.io/DKI_LLM/dualgraph/dualgraph_index.html)
[![Paper](http://img.shields.io/badge/Paper-arxiv.2602.13830-99D4C8.svg)](https://arxiv.org/abs/2602.13830)

We introduce **DualGraph Memory**, an architecture that separates what the agent knows from how it writes via two co-evolving graph structures: an **Outline Graph (OG)** that governs report structure and a **Knowledge Graph (KG)** that stores fine-grained knowledge units. By analyzing KG topology alongside OG structural signals, DualGraph generates targeted search queries for more efficient and comprehensive iterative knowledge-driven exploration. DualGraph consistently outperforms state-of-the-art baselines in report depth, breadth, and factual grounding across three benchmarks, and reaches a **53.08 RACE score** on DeepResearch Bench with GPT-5.

<div align="center">
  <img width="70%" src="docs/overview.png">
</div>

## Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt

# Or using uv (recommended, faster)
uv pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

```bash
cd deepresearch/baselines
cp .env.template .env
# Edit .env with your credentials
```

**LLM Configuration (required):**

| Variable | Description | Example |
|---|---|---|
| `LLM_PROVIDER` | `azure_openai` or `openai` | `azure_openai` |
| `LLM_MODEL_NAME` | Model deployment name | `gpt-4.1-20250414` |
| `LLM_API_KEY` | API key (not needed for Azure AAD) | |
| `LLM_BASE_URL` | API base URL | `https://api.openai.com/v1` |

**Search Configuration (at least one required):**

| Variable | Description |
|---|---|
| `BING_APP_ID` | Bing Search API key (for `--search-provider bing`) |
| `BING_ENDPOINT` | Bing Search endpoint URL |
| `SERPER_KEY_ID` | Serper API key (for `--search-provider serper`) |

**Page Reading (at least one required):**

| Variable | Description |
|---|---|
| `READPAGE_METHOD` | `crawl4ai`, `jina`, or `firecrawl` |
| `FIRECRAWL_API_URL` | Firecrawl API URL |
| `JINA_API_KEYS` | Jina API keys |
| `JINA_FALLBACK` | `true` to use Jina as fallback reader |

### Step 3: Run (CLI)

```bash
cd deepresearch/baselines

# Run with the example dataset (default)
python main.py

# Specify model and dataset
python main.py \
    --models gpt-4.1-20250414 \
    --datasets example \
    --id-range 1 1

# Run on a custom dataset with more queries
python main.py \
    --models gpt-4.1-20250414 \
    --datasets my_dataset \
    --id-range 1 20 \
    --search-provider serper
```

### Step 4: Run (Web UI via Chainlit)

DualGraph provides an interactive web interface powered by [Chainlit](https://github.com/Chainlit/chainlit) for a more visual experience.

```bash
cd deepresearch/baselines
chainlit run app.py -w
```

Then open <http://localhost:8000> in your browser.

**Web UI features:**

- Type any research question to start a live research session
- Type **`demo`** to replay a pre-generated report (no LLM calls needed)
- Real-time progress updates as the pipeline iterates
- Adjust parameters (search provider, query counts, clustering, etc.) in the settings panel

### Step 5: Evaluation

An `example/` dataset is included for quick testing. For the benchmark datasets used in our paper, see [eval_dataset/README.md](eval_dataset/README.md) for download instructions.

To add your own dataset, place a `query.jsonl` file in `eval_dataset/<your_name>/`:

```json
{"id": 1, "prompt": "Your research question here"}
```

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--models` | `gpt-4.1-20250414-2` | Model deployment name(s) |
| `--version` | `v1` | Version stamp for output directories |
| `--datasets` | `example` | Eval dataset name(s) (subfolder under `eval_dataset/`) |
| `--search-provider` | `bing` | Search backend: `bing` or `serper` |
| `--kg-query-num` | `10` | KG-based search queries per iteration |
| `--og-query-num` | `10` | Outline-based queries per iteration |
| `--id-range` | `1 1` | Query ID range [START, END] (inclusive) |
| `--max-iter` | `5` | Max research iterations per query |
| `--max-concurrency` | `5` | Max concurrent threads for batch processing |
| `--language` | `English` | Report language: `English` or `Chinese` |
| `--disable-early-stopping` | `False` | Disable multi-criteria early stopping |

## Citation
If you find this repository useful, please consider giving a star or citing:
```
@article{shi2026dualgraph,
  title={A Tale of Two Graphs: Separating Knowledge Exploration from Outline Structure for Open-Ended Deep Research},
  author={Shi, Zhuofan and Ma, Ming and Yao, Zekun and Yang, Fangkai and Zhang, Jue and Han, Dongge and R{\"u}hle, Victor and Lin, Qingwei and Rajmohan, Saravan and Zhang, Dongmei},
  journal={arXiv preprint arXiv:2602.13830},
  year={2026}
}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

## Question

If you have any question or find any bug, please go ahead and [open an issue](https://github.com/microsoft/DKI_LLM/issues). Issues are an acceptable discussion forum as well.

If you want to contact the authors, please email: `fangkaiyang AT microsoft.com`.
