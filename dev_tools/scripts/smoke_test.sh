#!/bin/bash
# Quick smoke test - runs tiny training to verify config works

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBMISSION_DIR="${PROJECT_ROOT}/records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA"

echo "=========================================="
echo "Smoke Test - Quick Verification"
echo "=========================================="

# Check GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found - assuming CPU test"
    NUM_GPUS=1
else
    NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "Detected $NUM_GPUS GPUs"
fi

# Smoke test config (tiny model, few steps)
export RUN_ID="smoke_test"
export SEED=1337
export NUM_LAYERS=10
export BIGRAM_VOCAB_SIZE=10240
export MLP_FC_QUANT_BITS=4
export MLP_PROJ_QUANT_BITS=5
export ATTN_QUANT_BITS=6
export MLP_MULT=3
export SWA_ENABLED=1
export SWA_START_FRAC=0.4
export TRAIN_SEQ_LEN=512
export TRAIN_BATCH_TOKENS=65536
export ITERATIONS=50
export MAX_WALLCLOCK_SECONDS=0

echo ""
echo "Config:"
echo "  Layers: $NUM_LAYERS"
echo "  BigramHash: $BIGRAM_VOCAB_SIZE"
echo "  Iterations: $ITERATIONS"
echo ""
echo "Starting smoke test (should finish in ~30 seconds)..."
echo ""

cd "$PROJECT_ROOT"

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    $SUBMISSION_DIR/train_gpt.py 2>&1 | tee logs/smoke_test.txt

echo ""
echo "=========================================="
echo "Smoke Test Complete!"
echo "=========================================="

# Check for final val_bpb
VAL_BPB=$(grep "final_int8_zlib_roundtrip_exact" logs/smoke_test.txt | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
SIZE=$(grep "Total submission size:" logs/smoke_test.txt | tail -1 | grep -oP '\d+' || echo "N/A")

echo ""
echo "Results:"
echo "  Final val_bpb: $VAL_BPB (smoke test - not meaningful)"
echo "  Submission size: $SIZE bytes"
echo ""

if [ "$SIZE" != "N/A" ] && [ "$SIZE" -lt 16000000 ]; then
    echo "✅ Size check PASSED ($SIZE < 16,000,000)"
else
    echo "❌ Size check FAILED ($SIZE >= 16,000,000)"
    exit 1
fi

echo ""
echo "Ready for full training!"
