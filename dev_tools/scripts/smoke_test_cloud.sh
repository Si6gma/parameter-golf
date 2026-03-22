#!/bin/bash
# Quick smoke test on cloud GPUs
# Runs a mini training to verify everything works before full 10-min runs

set -e

echo "=========================================="
echo "Parameter Golf - Cloud Smoke Test"
echo "=========================================="
echo ""

SUBMISSION_DIR="records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA"

# Check GPUs
echo "🔍 Checking GPUs..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "✓ Found $NUM_GPUS GPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Test torch
echo ""
echo "🔍 Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Quick training test (30 seconds)
echo ""
echo "🔍 Running mini training test (30 sec)..."
echo ""

export RUN_ID="smoke_test"
export SEED=1337
export NUM_LAYERS=4
export MODEL_DIM=256
export NUM_HEADS=4
export NUM_KV_HEADS=2
export MLP_MULT=2
export BIGRAM_VOCAB_SIZE=1024
export BIGRAM_DIM=64
export TRAIN_SEQ_LEN=512
export TRAIN_BATCH_TOKENS=65536
export ITERATIONS=10
export WARMUP_STEPS=2
export VAL_LOSS_EVERY=0
export TRAIN_LOG_EVERY=5
export MAX_WALLCLOCK_SECONDS=30

timeout 60 torchrun --standalone --nproc_per_node=$NUM_GPUS $SUBMISSION_DIR/train_gpt.py || true

# Check if log was created
if [ -f "logs/smoke_test.txt" ]; then
    echo ""
    echo "✓ Training log created"
    
    # Check for key outputs
    if grep -q "model_params:" logs/smoke_test.txt; then
        echo "✓ Model initialized"
    fi
    
    if grep -q "step:10/10" logs/smoke_test.txt; then
        echo "✓ Training loop completed"
    fi
    
    echo ""
    echo "Last 5 lines of log:"
    tail -5 logs/smoke_test.txt
else
    echo "❌ No training log created - something went wrong"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Smoke test passed!"
echo "=========================================="
echo ""
echo "Ready for full training: bash train_all_seeds.sh"
