# Parameter Golf Winning Strategy

## Current SOTA Analysis (1.1428 bpb)

The current leaderboard is dominated by entries combining:

1. **Mixed Quantization**: Int5 for MLP, Int6 for attention, FP16 for embeddings
2. **SmearGate**: Learned blending of current + previous token embeddings
3. **BigramHash**: Hash table for consecutive token pairs (4096-10240 buckets)
4. **SWA**: Stochastic Weight Averaging during warmdown (start_frac=0.4)
5. **Sliding Window Eval**: Stride=64 for better context utilization
6. **Orthogonal Init**: For faster convergence with Muon
7. **3x MLP Expansion**: 1536 hidden for 512-dim model

## My Improvements

### 1. Aggressive Quantization (Int4/5/6)
- **Int4** for MLP up-projection (fc): 2x compression vs int8
- **Int5** for MLP down-projection (proj): 1.6x compression
- **Int6** for attention: Better precision for attention mechanisms
- **FP16** for embeddings: Avoid quantization noise on most sensitive params

This enables **11 layers** instead of 10, gaining ~0.005 bpb from extra depth.

### 2. Enhanced BigramHash
- Increased buckets: 12288 (vs 10240 SOTA)
- Better hash function: SplitMix64-derived constants for lower collision rate
- Dim=128 projected to model_dim=512

Expected gain: ~0.001-0.002 bpb

### 3. Test-Time Training (TTT) with LoRA
During evaluation:
- Add LoRA layers (rank=4, alpha=8) to last 2 MLP projections
- Fine-tune on recent validation context (3 steps, lr=5e-4)
- Only use tokens already evaluated (no cheating)

Expected gain: ~0.005-0.01 bpb

### 4. Training Optimizations
- Longer warmdown: 3000 steps with SWA starting at 40%
- Weight decay: 0.04 for Muon, 0.01 for AdamW
- Momentum warmup: 0.92 → 0.99 over 1500 steps
- Gradient clipping: 0.3

## Target Architecture

```
11 layers, 512 dim, 8 heads, 4 KV heads
MLP: 3x expansion (1536 hidden) with int4 fc + int5 proj
Attention: int6 quantization
Total params: ~24M (compressed to ~15MB with int4/5/6 + zstd-22)
```

## Expected Performance

| Component | Expected Gain |
|-----------|--------------|
| 11 layers (vs 10) | -0.004 bpb |
| Int4/5/6 quantization | -0.002 bpb (vs int5/6) |
| Enhanced BigramHash | -0.001 bpb |
| Test-time training | -0.008 bpb |
| **Total Target** | **~1.128 bpb** |

## Running the Improved Model

### Basic (Improved)
```bash
RUN_ID=improved \
torchrun --standalone --nproc_per_node=8 train_gpt_improved.py
```

### Advanced (with TTT)
```bash
RUN_ID=advanced \
NUM_LAYERS=11 \
BIGRAM_VOCAB_SIZE=12288 \
MLP_FC_QUANT_BITS=4 \
MLP_PROJ_QUANT_BITS=5 \
TTT_ENABLED=1 \
TTT_LORA_RANK=4 \
TTT_STEPS=3 \
torchrun --standalone --nproc_per_node=8 train_gpt_advanced.py
```

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| num_layers | 11 |
| model_dim | 512 |
| mlp_mult | 3.0 |
| train_seq_len | 2048 |
| train_batch_tokens | 786,432 |
| matrix_lr | 0.02 |
| muon_momentum | 0.99 (warmup from 0.92) |
| muon_weight_decay | 0.04 |
| swa_start_frac | 0.4 |
| eval_stride | 64 |

## Future Explorations

1. **Mamba/State Space Models**: Could replace attention for better compression
2. **Product Keys**: Memory-efficient attention alternative
3. **Hypernetworks**: Generate weights from small seeds
4. **Dynamic depth**: Skip layers based on input difficulty
5. **Better compression**: Custom entropy coding beyond zstd
6. **QAT with learned scales**: Per-channel learned quantization

## The Math of Winning

To beat current SOTA (1.1428) by 0.005 nats (required for record):
- Need val_bpb ≤ 1.1393
- With my improvements targeting ~1.128, this provides margin for statistical significance

The key insight: **aggressive quantization buys you parameters, parameters buy you capacity, and test-time training extracts maximum value from that capacity.**
