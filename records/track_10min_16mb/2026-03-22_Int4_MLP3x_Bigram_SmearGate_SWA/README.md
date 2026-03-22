# Int4/5/6 Mixed Quantization + BigramHash + SmearGate + SWA

**Target val_bpb: ~1.135** (to be verified with 3-seed training)

> **Note**: Development tools (smoke tests, cloud scripts, etc.) are in `dev_tools/` folder. This submission folder contains only competition-required files.

## Run Command

```bash
# Train + evaluate (default settings already optimized)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Or with explicit env vars:
RUN_ID=my_run \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=12288 \
MLP_FC_QUANT_BITS=4 \
MLP_PROJ_QUANT_BITS=5 \
ATTN_QUANT_BITS=6 \
SWA_ENABLED=1 \
SWA_START_FRAC=0.4 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Architecture

- **11 layers**, 512 dim, 8 heads, 4 KV heads (GQA)
- **MLP 3x expansion** (hidden=1536), relu^2 activation
- **SmearGate**: Learned blending of current + previous token embeddings
- **BigramHash(12288)**: Hash table with 12288 buckets (dim=128, projected to 512)
- **U-Net skip connections** with learned skip_weights
- **Tied embeddings** (FP16, not quantized)

## Key Techniques

### 1. Aggressive Mixed Quantization

| Component | Quantization | Range | Compression |
|-----------|-------------|-------|-------------|
| MLP up-projection (fc) | **Int4** | [-7, 7] | 2.0x vs int8 |
| MLP down-projection (proj) | **Int5** | [-15, 15] | 1.6x vs int8 |
| Attention weights | **Int6** | [-31, 31] | 1.33x vs int8 |
| Embeddings | **FP16** | - | No quantization |

This aggressive quantization enables **11 layers** instead of 10, gaining capacity while staying under 16MB.

### 2. BigramHash(12288)

- Hash consecutive token pairs into 12288-bucket embedding table
- Dim=128, projected to model_dim=512 via learned linear
- Uses splitmix64-derived hash constants for lower collision rate
- Adds ~1.5M parameters for rich bigram signal

### 3. SmearGate

- Learned per-dimension gate blending current and previous token embeddings
- Gate init at sigmoid(3.0) ≈ 0.95 (near-identity)
- Model learns per-dimension how much previous-token context to add

### 4. EMA (Exponential Moving Average)

- Maintains moving average of model weights during training
- `ema_weight = decay * ema_weight + (1 - decay) * current_weight`
- Decay: 0.999, Start step: 100
- Applied to model before final evaluation for smoother weights
- Often combined with SWA for better generalization

### 5. Stochastic Weight Averaging (SWA)

### 4. Stochastic Weight Averaging (SWA)

- Collect checkpoints every 50 steps during warmdown
- Start at `start_frac=0.4` (last 40% of training)
- Average ~20-30 converged checkpoints
- Produces smoother weights that quantize better

### 5. Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| matrix_lr | 0.02 |
| tied_embed_lr | 0.03 |
| scalar_lr | 0.02 |
| muon_momentum | 0.99 (warmup from 0.92) |
| muon_weight_decay | 0.04 |
| adam_weight_decay | 0.01 |
| grad_clip_norm | 0.3 |
| warmdown_iters | 3000 |

### 6. Evaluation

- Sliding window with stride=64
- Full 2048 context window
- Evaluated on FineWeb validation set (tokenizer-agnostic BPB)

## Expected Gains

| Change vs SOTA (1.1428) | Expected Gain |
|------------------------|---------------|
| 11 layers (vs 10) | -0.004 bpb |
| Int4/5/6 quantization | -0.002 bpb |
| Enhanced BigramHash (12288) | -0.001 bpb |
| SmearGate optimization | -0.001 bpb |
| **Target Total** | **~1.135 bpb** |

## Files

- `train_gpt.py`: Main training script with all improvements
- `submission.json`: Submission metadata
- `train_seed*.log`: Training logs (to be generated)

## Requirements

```
torch>=2.0
numpy
sentencepiece
zstandard  # for zstd-22 compression
```

## Hardware

- 8x NVIDIA H100 80GB HBM3 SXM (or equivalent)
- Training time: ~10 minutes
- Peak memory: ~60GB per GPU

## Notes

This submission combines multiple SOTA techniques:
- Mixed precision quantization (int4/5/6) from winning entries
- BigramHash and SmearGate from 1.1428 submission
- SWA and orthogonal init from 1.1458 submission
- Aggressive hyperparameters tuned for 10-minute budget

The key innovation is using **int4 for MLP up-projection** to fund an 11th layer while maintaining precision on attention (int6) and embeddings (FP16).


---

## Development Notes

Helper scripts for training and testing are available in the `dev_tools/` folder:

```
dev_tools/
├── scripts/
│   ├── cloud_setup.sh      # Setup on cloud GPU
│   ├── train_all_seeds.sh  # Automated 3-seed training
│   └── smoke_test_cloud.sh # Quick verification test
├── utils/
│   ├── smoke_test.py       # Local Mac test
│   └── verify_submission.py # Result extraction
└── docs/
    ├── CLOUD_SETUP.md      # Full cloud guide
    └── SETUP_SUMMARY.md    # Quick reference
```

These are not part of the official submission.
