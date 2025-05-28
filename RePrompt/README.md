# RePrompt

[//]: # ([![Project]&#40;http://img.shields.io/badge/Project-SER-E3E4C8.svg&#41;]&#40;https://microsoft.github.io/DKI_LLM/ser/ser_index.html&#41;)

[//]: # ([![Paper]&#40;http://img.shields.io/badge/Paper-arxiv.2411.00418-99D4C8.svg&#41;]&#40;https://arxiv.org/abs/2411.00418&#41;)

In this paper, we propose RePrompt, a novel reprompting framework that introduces explicit reasoning into the prompt enhancement process via reinforcement learning. Instead of relying on handcrafted rules or stylistic rewrites, our method trains a language model to generate structured, self-reflective prompts by optimizing for image-level outcomes. The tailored reward models assesse the generated images in terms of human preference, semantic alignment, and visual composition, providing indirect supervision to refine prompt generation. Our approach enables end-to-end training without human-annotated data. Experiments on GenEval and T2I-Compbench show that RePrompt significantly boosts spatial layout fidelity and compositional generalization across diverse T2I backbones, establishing new state-of-the-art results. 
<div align="center">

  <img width="70%" src="docs/overview.png">

</div>

## Quick Start 🚀

### Step 1: Build Environment
```bash
conda create -n reprompt
conda activate reprompt
sh prepare.sh
```

Setup geneval with [geneval](https://github.com/djghosh13/geneval).

Setup t2i-compbench with [t2i-compbench](https://github.com/Karine-Huang/T2I-CompBench). And put it in directory ```evaluation```.

### Step 2: Init RePrompt with SFT


We use the following script to init our model with sft:

```shell
FORCE_TORCHRUN=1 llamafactory-cli train configs/qwen25-3b-full-sft.yaml 
```

### Step 3: Train RePrompt with RL


We use the following script to train our model with rl:

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file=configs/rl_ds2.yaml train_reprompt.py \
  --data data/generation_prompts_1k_filterd.txt \
  --gpt_path output/Qwen2.5-3B-sft \
  --sdmodel_name black-forest-labs/FLUX.1-dev \
  --outdir /output/reprompt_with_flux
```

### Step 4: Eval RePrompt


We use the following script to run our model on GenEval Benchmark:

```shell
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 evaluation/reprompt_infer_geneval.py
sh evaluation/geneval/eval.sh
```

We use the following script to run our model on T2I-CompBench:

```shell
torchrun --nnodes=1 --nproc_per_node=8 --node_rank=0 evaluation/reprompt_infer_t2icomp.py
```

And get the metrics with [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench) instruction.







## Citation
If you find this repository useful, please considering giving ⭐ or citing:
```
@misc{wu2025repromptreasoningaugmentedrepromptingtexttoimage,
      title={RePrompt: Reasoning-Augmented Reprompting for Text-to-Image Generation via Reinforcement Learning}, 
      author={Mingrui Wu and Lu Wang and Pu Zhao and Fangkai Yang and Jianjin Zhang and Jianfeng Liu and Yuefeng Zhan and Weihao Han and Hao Sun and Jiayi Ji and Xiaoshuai Sun and Qingwei Lin and Weiwei Deng and Dongmei Zhang and Feng Sun and Qi Zhang and Rongrong Ji},
      year={2025},
      eprint={2505.17540},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.17540}, 
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

If you want to concat the author, please email: `mingrui0001@gmail.com` and `wlu@microsoft.com`.