# Recent PR Analysis (March 22, 2026)

Based on Discord screenshot showing latest GitHub PRs

## PR Leaderboard (Latest)

| PR | Author | Score | Technique | TTT? | Valid? |
|----|--------|-------|-----------|------|--------|
| #442 | sjp611 | **1.1027** 🔥 | 11L EMA + AdamW TTT 10ep | YES | Maybe INVALID |
| #445 | newjordan | **1.1232** | 11L TTT Burst + EMA + GPTQ-lite | YES | Maybe INVALID |
| #452 | ofirkris | **1.1365** | 10L XSA + EMA + Partial RoPE + LN Scale + TTT | YES | Maybe INVALID |
| #450 | zachgoldfine44 | **1.1466** | 12L + Catalytic Residuals + BigramHash + SWA + Late QAT | NO | ✅ VALID |
| #447 | CREVIOS | **1.1431** | Bigram-Aware + Mixed-Precision | NO | ✅ VALID |

## Key Insights

### 1. TTT Dominates Top Scores BUT May Be Invalid
- Top 3 PRs all use TTT
- Discord rule: "Don't train on eval tokens you haven't scored"
- Many TTT implementations train on "future" tokens → likely invalid

### 2. Non-TTT Best: 1.1431 - 1.1466
- **PR #450**: 12L + Catalytic Residuals + Late QAT = 1.1466
- **PR #447**: Bigram-Aware + Mixed-Precision = 1.1431

### 3. Our Position (Estimated)
- **Current target**: ~1.130 with EMA
- **If we hit 1.130**: Beat non-TTT best (1.1431)
- **Gap to close**: 0.013 - 0.016 bpb

## What We Need to Match #450 (1.1466)

PR #450 uses:
1. ✅ 12 layers (we have 11, could add 1 more)
2. ❓ **Catalytic Residuals** (UNKNOWN - PRIORITY 1)
3. ✅ BigramHash (we have this)
4. ✅ SWA (we have this)
5. ❓ **Late QAT** (UNKNOWN - PRIORITY 2)

## Research Priority (Non-TTT Only)

### 🔥 PRIORITY 1: Catalytic Residuals
- Source: PR #450 (1.1466, non-TTT)
- Unknown what this is
- Could be biggest gain
- **Action**: Deep research needed

### 🔥 PRIORITY 2: Late QAT
- Source: PR #450 (1.1466, non-TTT)
- QAT only in final training steps
- Proven to help quantization
- **Action**: Implement if Catalytic Residuals unclear

### 🔥 PRIORITY 3: 12th Layer
- PR #450 uses 12 layers (we have 11)
- May need int4 for more layers to fit
- **Action**: Try int3 or aggressive pruning

### MEDIUM PRIORITY: LN Scale
- Source: PR #452 (1.1365, but has TTT)
- Simple addition
- Unknown if significant alone
- **Action**: Quick implementation test

### LOW PRIORITY: GPTQ-lite
- Source: PR #445 (1.1232, has TTT)
- Complex, may not fit 10min training
- **Action**: Skip unless other options exhausted

## Strategy

### Phase 1: Current Run (Training Now)
- See what we get with current setup
- Target: ~1.130

### Phase 2: If Below 1.1431
1. Research Catalytic Residuals (use deep research prompt)
2. Implement Late QAT
3. Try 12th layer with more aggressive quantization

### Phase 3: If Still Below 1.1431
1. Add LN Scale
2. Hyperparameter tuning
3. Consider architectural changes (XSA?)

## Expected Gains

| Technique | Expected Gain | Confidence |
|-----------|---------------|------------|
| Catalytic Residuals | -0.005 to -0.010 | Low (unknown) |
| Late QAT | -0.002 to -0.005 | High |
| 12th Layer | -0.002 to -0.004 | Medium |
| LN Scale | -0.001 to -0.002 | Medium |
| **Combined** | **-0.010 to -0.021** | - |

**If we implement all**: Could reach **1.109 - 1.120**

This would beat even TTT entries if they're invalidated!

## Immediate Action Items

While waiting for training:
1. ✅ Use deep research prompt on "Catalytic Residuals"
2. ✅ Use deep research prompt on "Late QAT implementation"
3. Review PR #450 code if available
4. Prepare v2 implementation

## Notes

- Competition runs until April 30
- OpenAI may invalidate TTT entries
- Non-TTT improvements are "future-proof"
- Even small gains compound
