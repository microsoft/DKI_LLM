# Evaluation Datasets

Due to licensing and distribution restrictions, the benchmark evaluation datasets used in our paper are **not included** in this repository.

## Available Benchmarks

| Folder | Description | How to Obtain |
|--------|-------------|---------------|
| `deepresearchbench/` | DeepResearchBench queries | See [DeepResearchBench](https://github.com/deep-research-bench) |
| `deepresearchbench2/` | DeepResearchBench2 tasks & rubrics | See [DeepResearchBench2](https://github.com/deep-research-bench) |
| `deepresearch_gym/` | Researchy Queries (ClueWeb sample) | See [DeepResearch-Gym](https://github.com/nickjiang2378/DeepResearch-Gym) |
| `deepconsult/` | DeepConsult comparison data | See original paper / authors |

## Getting Started with the Example Dataset

An `example/` folder is provided with a sample query so you can test the pipeline immediately:

```bash
cd deepresearch/baselines
python main.py --datasets example --id-range 1 1
```

## Adding Your Own Dataset

Place a `query.jsonl` file in a new subfolder under `eval_dataset/`. Each line should be a JSON object with `id` and `prompt` fields:

```json
{"id": 1, "prompt": "Your research question here"}
{"id": 2, "prompt": "Another research question"}
```

Then run:

```bash
python main.py --datasets your_folder_name --id-range 1 2
```
