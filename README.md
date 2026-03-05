# Parameter-Efficient Finetuning (OFT) on SST-2 with Qwen-2.5-1.5B

This repo implements **Orthogonal Fine-Tuning (OFT)** (PEFT) for a pretrained LLM (**Qwen-2.5-1.5B**) on the **SST-2** sentiment classification task, formulated as a **prompt-based causal language modeling** problem.

## Results (SST-2 validation, 800 samples)
| Setting | Accuracy |
|---|---:|
| Before finetuning | 0.9113 |
| After OFT finetuning | **0.9525** |

Trainable parameters (OFT only): **7,888,384 / 1,551,602,688 (0.51%)**

## Repository Structure
- `scripts/`: training / evaluation / plotting scripts
- `configs/`: experiment configs
- `figures/`: loss curve and accuracy comparison figures

## Environment
- Single GPU: RTX 4070s
- OS: Windows + WSL2 (Ubuntu 24.04)
- Frameworks: PyTorch, Transformers, PEFT

## Setup
```bash
pip install -r requirements.txt

If you are in mainland China, you may need a HuggingFace mirror:
```
export HF_ENDPOINT=https://hf-mirror.com
```

## Run
### Train (OFT)
```bash
python scripts/train_oft_sst2.py --config configs/oft_sst2.yaml
```

### Evaluate (OFT)
```bash
python scripts/eval_oft_sst2.py --config configs/oft_sst2.yaml --ckpt outputs/checkpoints/oft_sst2
```

### Plot figures
```bash
python scripts/plot_loss_from_state.py --state outputs/checkpoints/oft_sst2/checkpoint-313/trainer_state.json --out figures/loss_curve.png
python scripts/plot_accuracy.py --before <the origin acc> --after <the finetuned acc> --out figures/acc_comparison.png
```

Notes
Loss is computed only on the answer tokens (prompt tokens are masked with -100).
Model checkpoints are NOT committed to this repo (see .gitignore).