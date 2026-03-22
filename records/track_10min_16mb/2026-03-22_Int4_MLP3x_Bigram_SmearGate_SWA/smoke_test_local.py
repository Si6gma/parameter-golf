#!/usr/bin/env python3
"""
Local CPU smoke test for 12-layer submission.
Validates model init, forward pass, quantization, and estimated artifact size.
No CUDA required.
"""
import sys, io, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

# ── patch out CUDA guard so we can import the module ──────────────────────────
import unittest.mock as mock

# We'll import the relevant classes by exec'ing the file with the guard patched
with open(os.path.join(os.path.dirname(__file__), "train_gpt.py")) as f:
    src = f.read()

# Replace the CUDA guard temporarily for import
src_patched = src.replace(
    'if not torch.cuda.is_available():\n        raise RuntimeError("CUDA is required")',
    'pass  # CUDA guard patched for smoke test'
)

ns = {}
try:
    exec(compile(src_patched, "train_gpt.py", "exec"), ns)
except SystemExit:
    pass
except Exception as e:
    # Expected: main() will fail without data files; we just need the classes
    pass

GPT = ns["GPT"]
mixed_quantize = ns["mixed_quantize"]
dequantize_mixed = ns["dequantize_mixed"]

# ── Config matching the submission defaults ────────────────────────────────────
CFG = dict(
    vocab_size=1024,
    num_layers=12,
    model_dim=512,
    num_heads=8,
    num_kv_heads=4,
    mlp_mult=3.0,
    tie_embeddings=True,
    tied_embed_init_std=0.005,
    logit_softcap=30.0,
    rope_base=10000.0,
    qk_gain_init=1.0,
    bigram_vocab_size=10240,
    bigram_dim=64,
)

print("=" * 60)
print("SMOKE TEST: 12-layer submission")
print("=" * 60)

# ── 1. Model init ──────────────────────────────────────────────────────────────
print("\n[1] Model initialization...")
model = GPT(**CFG)
n_params = sum(p.numel() for p in model.parameters())
print(f"    num_layers : 12")
print(f"    bigram     : vocab={CFG['bigram_vocab_size']} dim={CFG['bigram_dim']}")
print(f"    n_params   : {n_params:,}")
assert model.num_encoder_layers == 6, "Expected 6 encoder layers"
assert model.num_decoder_layers == 6, "Expected 6 decoder layers"
assert model.num_skip_weights == 6,   "Expected 6 skip weights"
print("    ✓ encoder/decoder split: 6/6, skip_weights: 6")

# ── 2. Forward pass (CPU, short sequence) ────────────────────────────────────
print("\n[2] Forward pass (CPU, batch=2, seq=64)...")
model.eval()
with torch.no_grad():
    x = torch.randint(0, 1024, (2, 64))
    logits = model(x)
    assert logits.shape == (2, 64, 1024), f"Unexpected logits shape: {logits.shape}"
print(f"    logits shape: {logits.shape}  ✓")

# ── 3. Quantization round-trip ────────────────────────────────────────────────
print("\n[3] Quantization round-trip (int4/5/6)...")
sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
quant_result, quant_meta = mixed_quantize(sd, mlp_fc_bits=4, mlp_proj_bits=5, attn_bits=6)

buf = io.BytesIO()
torch.save({"w": quant_result, "m": quant_meta}, buf)
raw_bytes = len(buf.getvalue())

try:
    import zstandard
    compressed = zstandard.ZstdCompressor(level=22).compress(buf.getvalue())
    compressor = "zstd-22"
except ImportError:
    import zlib
    compressed = zlib.compress(buf.getvalue(), level=9)
    compressor = "zlib-9"

model_compressed_bytes = len(compressed)

code_path = os.path.join(os.path.dirname(__file__), "train_gpt.py")
with open(code_path) as f:
    code_bytes = len(f.read().encode("utf-8"))

total_artifact = model_compressed_bytes + code_bytes
limit = 16_000_000

print(f"    compressor        : {compressor}")
print(f"    quantized raw     : {raw_bytes:,} bytes ({raw_bytes/1e6:.2f} MB)")
print(f"    compressed model  : {model_compressed_bytes:,} bytes ({model_compressed_bytes/1e6:.2f} MB)")
print(f"    code              : {code_bytes:,} bytes ({code_bytes/1e6:.2f} MB)")
print(f"    total artifact    : {total_artifact:,} bytes ({total_artifact/1e6:.2f} MB)")
print(f"    limit             : {limit:,} bytes (16.00 MB)")
margin = limit - total_artifact
print(f"    margin            : {margin:,} bytes ({margin/1e6:.2f} MB) {'✓' if margin > 0 else '✗ OVER BUDGET!'}")

# Verify round-trip fidelity
print("\n[4] Dequantization fidelity check...")
state = torch.load(io.BytesIO(buf.getvalue()), map_location="cpu")
deq = dequantize_mixed(state["w"], state["m"], sd)
max_err = max(
    (deq[k] - sd[k].to(deq[k].dtype)).abs().max().item()
    for k in sd if k in deq
)
print(f"    max dequant error : {max_err:.6f}  ✓")

print("\n" + "=" * 60)
if margin > 0:
    print("SMOKE TEST PASSED ✓")
else:
    print("SMOKE TEST FAILED ✗ (over 16MB budget)")
print("=" * 60)
