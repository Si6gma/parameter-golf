# Implementation Status

Last updated: March 22, 2026

## Current Leaderboard Status

| Rank | PR | Score | Technique | Valid? |
|------|-----|-------|-----------|--------|
| 1 | #442 | **1.1027** | 11L EMA + AdamW TTT 10ep | Maybe invalid (TTT) |
| 2 | #445 | **1.1232** | 11L TTT Burst + EMA + GPTQ-lite | Maybe invalid (TTT) |
| 3 | #452 | **1.1365** | 10L XSA + EMA + Partial RoPE + LN Scale + TTT | Maybe invalid (TTT) |
| - | Ours | **~1.135 target** | 11L Int4/5/6 + BigramHash + SmearGate + SWA + EMA | Non-TTT, should be valid |

## Our Implementation

### ✅ Implemented

| Technique | Location | Notes |
|-----------|----------|-------|
| **Int4/5/6 Quantization** | `train_gpt.py` | MLP fc=int4, proj=int5, attn=int6 |
| **11 Layers** | `train_gpt.py` | Enabled by aggressive quantization |
| **BigramHash(12288)** | `train_gpt.py` | Hash table for token pairs |
| **SmearGate** | `train_gpt.py` | Learned bigram context |
| **SWA** | `train_gpt.py` | Stochastic Weight Averaging |
| **EMA** | `train_gpt.py` | Exponential Moving Average (NEW) |
| **Sliding Window Eval** | `train_gpt.py` | Stride=64 |

### 🔄 To Research/Implement

| Technique | Priority | Source | Complexity |
|-----------|----------|--------|------------|
| **Late QAT** | HIGH | PR #450 | Medium |
| **Catalytic Residuals** | HIGH | PR #450 | Unknown |
| **GPTQ-lite** | MEDIUM | PR #445 | Medium |
| **LN Scale** | MEDIUM | PR #452 | Low |
| **Partial RoPE** | LOW | PR #452 | Medium |
| **XSA** | LOW | PR #452 | Unknown |

## EMA Implementation Details

Added March 22, 2026:

```python
class EMA:
    def __init__(self, model, decay=0.999, start_step=100):
        self.decay = decay
        self.start_step = start_step
        self.ema_params = {name: param.clone() 
                          for name, param in model.named_parameters()}
    
    def update(self, model, step):
        if step < self.start_step:
            self.ema_params[name].copy_(param)  # Warmup
        else:
            self.ema_params[name].mul_(decay).add_(param, alpha=1-decay)
    
    def apply_to_model(self, model):
        # Apply EMA weights for evaluation
        for name, param in model.named_parameters():
            param.data.copy_(self.ema_params[name])
```

**Env vars:**
- `EMA_ENABLED=1` (default)
- `EMA_DECAY=0.999` (default)
- `EMA_START_STEP=100` (default)

## Expected Performance

With EMA added, our expected performance:

| Component | Expected Gain |
|-----------|--------------|
| 11 layers (vs 10) | -0.004 bpb |
| Int4/5/6 quantization | -0.002 bpb |
| BigramHash (12288) | -0.001 bpb |
| SmearGate | -0.001 bpb |
| SWA | -0.0006 bpb |
| **EMA** | **-0.003 to -0.005 bpb** |
| **New Target** | **~1.130 to 1.132** |

If we hit 1.130, we beat the current non-TTT best (~1.1365) even if TTT entries are invalidated.

## Next Steps

1. ✅ **Deploy to RunPod and test current implementation**
   - Train 3 seeds with current code
   - Verify EMA helps

2. **Research Late QAT** (if current run doesn't beat 1.1365)
   - Apply QAT only in last N steps
   - Could add another 0.002-0.003 gain

3. **Research Catalytic Residuals**
   - Need to find paper/implementation
   - Could be significant gain

## Testing Plan

```bash
# On RunPod
cd records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With explicit EMA settings:
EMA_ENABLED=1 EMA_DECAY=0.999 EMA_START_STEP=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## If We Don't Beat 1.1365

Add these in order:
1. **Late QAT** - Start fake quantization at step 5000
2. **LN Scale** - Add learnable scale after RMSNorm
3. **Catalytic Residuals** - Once we understand what it is

## Notes

- TTT techniques may all be invalidated per Discord
- Non-TTT SOTA is currently ~1.1365 (PR #452 without TTT?)
- We should focus on non-TTT improvements to be safe
- EMA + Late QAT could get us to ~1.128 without any TTT
