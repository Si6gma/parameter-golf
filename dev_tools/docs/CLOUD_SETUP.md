# Cloud Training Setup Guide

This guide walks you through setting up and training on cloud GPUs (8xH100).

## Prerequisites

- Access to 8xH100 GPUs (RunPod, Lambda, etc.)
- Git repository cloned
- Python 3.10+ with pip

## Quick Start

### 1. Launch Cloud Instance

**Recommended: RunPod**
- Template: [Parameter Golf Template](https://console.runpod.io/deploy?template=y5cejece4j)
- GPU: 8x H100 SXM
- Storage: At least 100GB

**Alternative: Lambda Labs or other providers**
- Any provider with 8x H100s works
- Make sure you have SSH access

### 2. SSH into Instance

```bash
ssh root@your-instance-ip
```

### 3. Clone/Upload Code

```bash
git clone https://github.com/yourusername/parameter-golf.git
cd parameter-golf
```

### 4. Run Setup

```bash
bash cloud_setup.sh
```

This will:
- ✅ Check GPUs are available
- ✅ Install PyTorch and dependencies
- ✅ Download FineWeb dataset
- ✅ Verify submission files

### 5. Smoke Test

Before burning 30+ minutes on full training, run a quick test:

```bash
bash smoke_test_cloud.sh
```

This runs a tiny model for 30 seconds to verify everything works.

### 6. Full Training (3 Seeds)

```bash
bash train_all_seeds.sh
```

This will:
- Train 3 seeds (1337, 42, 7) sequentially
- Each run takes ~10 minutes
- Total time: ~35 minutes
- Logs saved to `logs/train_seed*.txt`

### 7. Verify Results

```bash
python3 verify_submission.py
```

This will:
- Extract val_bpb from all 3 logs
- Calculate mean and std dev
- Check if you beat SOTA (1.1428)
- Update `submission.json` with results

## What Gets Generated

After training completes, you'll have:

```
records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/
├── README.md                    # Documentation
├── submission.json              # Metadata with actual results
├── requirements.txt             # Dependencies
├── train_gpt.py                 # Training script
├── train_seed1337.txt          # Training log seed 1337
├── train_seed42.txt            # Training log seed 42
└── train_seed7.txt             # Training log seed 7
```

## Expected Results

If everything works:

| Seed | Expected val_bpb |
|------|-----------------|
| 1337 | ~1.134-1.136 |
| 42   | ~1.134-1.136 |
| 7    | ~1.134-1.136 |
| **Mean** | **~1.135** |

Target: **Beat 1.1428 by at least 0.005 nats** (~0.003 bpb)

## Troubleshooting

### Out of Memory
- Reduce `TRAIN_BATCH_TOKENS` (default: 786432)
- Reduce `TRAIN_SEQ_LEN` (default: 2048)

### Training Too Slow
- Check GPUs are being used: `nvidia-smi`
- Verify NCCL is working (should see 8 GPUs at 90%+ utilization)

### Import Errors
```bash
pip install torch numpy sentencepiece zstandard
```

### Dataset Not Found
```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## Manual Training (Single Seed)

If you want to train a single seed manually:

```bash
export RUN_ID=my_run
export SEED=1337
export NUM_LAYERS=11
export BIGRAM_VOCAB_SIZE=12288
export MLP_FC_QUANT_BITS=4
export MLP_PROJ_QUANT_BITS=5
export ATTN_QUANT_BITS=6
export SWA_ENABLED=1
export SWA_START_FRAC=0.4

torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/train_gpt.py
```

## Cost Estimate

| Provider | 8x H100 Cost | Per Run (10 min) | 3 Seeds |
|----------|-------------|------------------|---------|
| RunPod | ~$18/hr | ~$3 | ~$9 |
| Lambda | ~$20/hr | ~$3.33 | ~$10 |
| OpenAI Grant | FREE | FREE | FREE |

## Next Steps After Training

1. ✅ Verify results with `verify_submission.py`
2. ✅ Check mean val_bpb beats 1.1428
3. ✅ Create PR to main repo
4. ✅ Submit to leaderboard!

## Monitoring Training

Watch training in real-time:

```bash
tail -f logs/train_seed1337.txt | grep "step:"
```

Or watch GPU utilization:

```bash
watch -n 1 nvidia-smi
```

Good luck! 🚀
