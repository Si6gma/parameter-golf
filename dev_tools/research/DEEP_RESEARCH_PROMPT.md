# Deep Research Prompt for Claude

Use this prompt to delegate deep research while training runs:

---

```
You are a research assistant for the Parameter Golf competition (https://github.com/openai/parameter-golf).

CURRENT SITUATION:
- We're training an 11-layer transformer with int4/5/6 quantization
- Current code includes: BigramHash, SmearGate, SWA, EMA
- Target: Beat 1.1428 bpb (current SOTA is actually 1.1027 with TTT, but TTT may be invalid)
- We need NON-TTT improvements

RESEARCH TASK: [INSERT TASK HERE]

Deliverables:
1. What is it? (concept explanation)
2. Why would it help? (intuition)
3. How to implement? (pseudocode or actual PyTorch code)
4. Expected gain? (estimate based on similar techniques)
5. Any papers/repos to reference?

Context from recent PRs:
- EMA + TTT getting 1.1027 (PR #442)
- Late QAT + Catalytic Residuals getting 1.1466 (PR #450)
- XSA + Partial RoPE + LN Scale getting 1.1365 (PR #452)
- GPTQ-lite + TTT getting 1.1232 (PR #445)

Current architecture:
- 11 layers, 512 dim, 8 heads, 4 KV heads
- MLP 3x expansion (1536 hidden)
- Int4 for MLP up-proj, Int5 for down-proj, Int6 for attention
- BigramHash(12288), SmearGate
- SWA + EMA

Be thorough. Return working code if possible.
```

---

## Quick Copy-Paste Tasks

### Task 1: Late QAT Research
```
RESEARCH TASK: Late QAT (Quantization Aware Training)

Research how to implement Late QAT - applying fake quantization only in the final N steps of training instead of from the beginning.

Questions:
1. When should we start fake quantization? (step 5000? 8000?)
2. What bit-widths to use during late QAT?
3. Should we use STE (Straight Through Estimator) or something else?
4. How does this interact with EMA and SWA?
5. Expected improvement over post-training quantization?

Deliver working PyTorch code that can be added to train_gpt.py
```

### Task 2: Catalytic Residuals Research
```
RESEARCH TASK: Catalytic Residuals

Research "Catalytic Residuals" or "Catalytic Connections" in deep learning.

Appears in PR #450 with 1.1466 score, combined with BigramHash + SWA + Late QAT.

Search for:
- Papers mentioning "catalytic" in context of neural networks
- Could be related to: learned residual scaling, gated residuals, attention-gated skip connections
- Similar to Highway Networks or ResNet modifications

Deliver:
1. What this technique likely is
2. How to implement in a transformer
3. Where in the architecture to add it (block level? attention?)
4. Pseudocode or PyTorch implementation
```

### Task 3: GPTQ-lite Research
```
RESEARCH TASK: GPTQ-lite Implementation

Research lightweight GPTQ (post-training quantization) that doesn't require full Hessian computation.

Context: PR #445 uses "GPTQ-lite" + EMA + TTT for 1.1232

Questions:
1. How does GPTQ differ from simple per-row quantization?
2. What is the "lite" version? (approximated Hessian?)
3. Can we implement this without external libraries?
4. Expected improvement over per-row int4/5/6?

Deliver working code or point to existing implementations we can adapt.
```

### Task 4: LN Scale Research
```
RESEARCH TASK: LayerNorm Scale (LN Scale)

Appears in PR #452 with 1.1365, combined with XSA + Partial RoPE + EMA + TTT.

Research adding learnable or fixed scaling factors after LayerNorm/RMSNorm.

Standard RMSNorm: x * rsqrt(mean(x^2))
With LN Scale: x * rsqrt(mean(x^2)) * scale

Questions:
1. Is scale learnable or fixed?
2. Per-channel or per-layer?
3. Initialization strategy?
4. Where to apply in the transformer block?

Deliver minimal PyTorch implementation.
```

### Task 5: XSA Research
```
RESEARCH TASK: XSA (Cross-Layer/Scale Attention)

Appears in PR #452. Could be:
- Cross-Layer Attention
- Extended Self-Attention  
- Multi-Scale Attention

Research what XSA means in recent transformer literature.

If unclear, search for alternative interpretations.

Deliver:
1. Most likely definition
2. Implementation complexity
3. Whether it's worth trying for Parameter Golf
```

### Task 6: Hyperparameter Optimization
```
RESEARCH TASK: Hyperparameter Optimization for 10-minute training

Current settings:
- train_batch_tokens: 786,432
- train_seq_len: 2048
- warmdown_iters: 3000
- matrix_lr: 0.02
- tied_embed_lr: 0.03
- muon_momentum: 0.99 (warmup from 0.92)
- swa_start_frac: 0.4
- ema_decay: 0.999, start_step: 100

Research optimal hyperparameters for:
1. Very short training (10 minutes = ~7000-8000 steps)
2. Small model (11 layers, 512 dim, ~24M params)
3. Aggressive quantization (int4/5/6)

Look at:
- Learning rate schedules (warmup, cool-down)
- Batch size effects
- Momentum schedules
- Weight decay tuning

Deliver recommended hyperparameter changes with justification.
```

---

## How to Use

1. Pick a task
2. Copy the prompt
3. Paste into Claude/Code/ChatGPT
4. Get detailed research + code
5. While training runs on Hyperbolic, iterate on improvements

## Pro Tip

Run multiple research tasks in parallel with different instances:
- Instance 1: Late QAT
- Instance 2: Catalytic Residuals
- Instance 3: Hyperparameter tuning

Combine best findings into improved submission v2.
