#!/usr/bin/env bash
# ============================================================
# DualGraph - Example Run Commands
# ============================================================

cd deepresearch/baselines

# ---------- 1. Web UI (Chainlit) ----------
chainlit run app.py -w
# Open http://localhost:8000

# ---------- 2. Run with the example dataset (default) ----------
python main.py

# ---------- 3. Specify model and version ----------
python main.py \
    --models gpt-4.1-20250414 \
    --version v1 \
    --datasets example \
    --id-range 1 1

# ---------- 4. Run on a custom dataset ----------
python main.py \
    --models gpt-4.1-20250414 \
    --version v1 \
    --datasets my_dataset \
    --id-range 1 20 \
    --search-provider serper

# ---------- 5. Disable early stopping, use more iterations ----------
python main.py \
    --models gpt-4.1-20250414 \
    --version v1 \
    --datasets example \
    --id-range 1 1 \
    --max-iter 8 \
    --disable-early-stopping

# ---------- 6. Chinese report ----------
python main.py \
    --models gpt-4.1-20250414 \
    --version v1 \
    --datasets example \
    --id-range 1 1 \
    --language Chinese
