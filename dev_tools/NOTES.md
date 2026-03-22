# Dev Notes

Quick reference for development and cloud training.

## Cloud Training (8x H100)

```bash
# Setup
bash dev_tools/scripts/cloud_setup.sh

# Quick test
bash dev_tools/scripts/smoke_test_cloud.sh

# Full training (3 seeds)
bash dev_tools/scripts/train_all_seeds.sh

# Verify results
python3 dev_tools/utils/verify_submission.py
```

## Current Submission

Location: `records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/`

Key techniques:
- **Int4/5/6 quantization** - Aggressive compression (MLP fc=int4, proj=int5, attn=int6)
- **11 layers** - Extra layer from quantization savings
- **BigramHash(12288)** - Hash table for token pairs
- **SmearGate** - Learned bigram context blending
- **SWA** - Stochastic Weight Averaging (start_frac=0.4)
- **EMA** - Exponential Moving Average (decay=0.999)
- **Sliding window eval** - Stride=64

Target: **~1.135 bpb** (beat SOTA 1.1428)

## Manual Training

```bash
cd ~/parameter-golf
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/train_gpt.py
```

## Architecture

```
11 layers, 512 dim, 8 heads, 4 KV heads
MLP: 3x expansion (1536 hidden)
Params: ~28M → ~15MB compressed
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| matrix_lr | 0.02 |
| muon_momentum | 0.99 (warmup from 0.92) |
| muon_wd | 0.04 |
| adam_wd | 0.01 |
| swa_start_frac | 0.4 |
| ema_decay | 0.999 |
| eval_stride | 64 |

## Local Testing (Mac MLX)

```bash
pip install mlx numpy sentencepiece
python3 train_gpt_mlx.py
```
