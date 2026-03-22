# New Techniques to Research (March 22, 2026)

Based on latest GitHub PRs showing scores 1.1027 - 1.1365

## 1. EMA (Exponential Moving Average) 🔥

**Appears in:** PR #442 (1.1027), #445 (1.1232), #452 (1.1365)

### What is it?
EMA maintains a moving average of model weights during training:
```
ema_weight = decay * ema_weight + (1 - decay) * current_weight
```

### Why it helps?
- Smoothes weight updates
- Better generalization
- Often used with SWA (stochastic weight averaging)

### Implementation idea:
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_params = {name: param.clone().detach() 
                          for name, param in model.named_parameters()}
    
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
```

---

## 2. Catalytic Residuals 🔥

**Appears in:** PR #450 (1.1466)

### What we know:
- 12 layers with "Catalytic Residuals"
- Combined with BigramHash(10240) + SWA + Late QAT

### Hypothesis:
Could be:
- Learned residual scaling
- Gated residuals
- Some form of attention-gated skip connections
- Similar to "SmearGate" but for residuals

### Research needed:
Look up "catalytic connections" in deep learning literature

---

## 3. Late QAT (Quantization Aware Training)

**Appears in:** PR #450 (1.1466)

### What is it?
QAT applied only in late stages of training, not from beginning.

### Why it helps?
- Early training: full precision for stable convergence
- Late training: QAT to optimize for quantization

### Implementation:
```python
# Start QAT only after N steps
if step > late_qat_start_step:
    apply_fake_quantization()
```

---

## 4. GPTQ-lite

**Appears in:** PR #445 (1.1232)

### What is it?
Lightweight version of GPTQ (post-training quantization).

### Standard GPTQ:
- Quantizes weights layer by layer
- Uses Hessian information
- More aggressive than simple per-row scaling

### "Lite" version?
- Approximated Hessian
- Faster, less memory
- Good enough for small models

---

## 5. XSA (Cross-Layer/Scale Attention?)

**Appears in:** PR #452 (1.1365)

### Hypotheses:
- Cross-Layer Attention
- Extended Self-Attention
- Some form of multi-scale attention

### Combined with:
- EMA
- Partial RoPE
- LN Scale
- TTT (but TTT may be invalid)

---

## 6. Partial RoPE

**Appears in:** PR #452 (1.1365)

### What is it?
RoPE applied to only part of the attention dimensions.

### Standard RoPE:
- Applied to all head_dim

### Partial RoPE:
- Applied to head_dim // 2 or some fraction
- Rest uses absolute or no positional encoding

### Why?
- Saves compute
- May improve certain types of generalization

---

## 7. LN Scale

**Appears in:** PR #452 (1.1365)

### What is it?
Learned or fixed scaling after LayerNorm.

### Implementation:
```python
# Standard RMSNorm
x = x * rsqrt(mean(x^2))

# With LN Scale
x = x * rsqrt(mean(x^2)) * scale
```

Where `scale` is a learnable parameter.

---

## Research Priorities

### Immediate (do first):
1. **EMA** - Easy to implement, big wins in multiple PRs
2. **Late QAT** - Simple change, proven in #450
3. **LN Scale** - Very simple addition

### Secondary:
4. **Catalytic Residuals** - Need more research
5. **GPTQ-lite** - May require external library

### Tertiary (may be complex):
6. **XSA** - Unknown complexity
7. **Partial RoPE** - May hurt if not done right

---

## Next Steps

1. Implement EMA + our existing improvements
2. Add Late QAT
3. Test on RunPod
4. If promising, add LN Scale and Catalytic Residuals
