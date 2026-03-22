#!/bin/bash
# Train all 3 seeds for submission
# This runs the full 10-minute training 3 times with different seeds

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

SUBMISSION_DIR="${PROJECT_ROOT}/records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA"
SEEDS=(1337 42 7)

echo "=========================================="
echo "Parameter Golf - Multi-Seed Training"
echo "=========================================="
echo ""

# Check GPUs
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found"
    exit 1
fi

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "Detected $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs, found $NUM_GPUS"
    echo "Training will use available GPUs"
fi

# Create logs directory
mkdir -p logs
mkdir -p $SUBMISSION_DIR

echo ""
echo "Will train 3 seeds: ${SEEDS[@]}"
echo "Estimated time: ~35 minutes (3 × 10 min + overhead)"
echo ""

# Function to run training
run_seed() {
    local SEED=$1
    local RUN_ID="train_seed${SEED}"
    local LOG_FILE="logs/${RUN_ID}.txt"
    
    echo "=========================================="
    echo "Training seed $SEED"
    echo "Run ID: $RUN_ID"
    echo "Started: $(date)"
    echo "=========================================="
    
    # Environment variables for this run
    export RUN_ID
    export SEED
    export NUM_LAYERS=11
    export BIGRAM_VOCAB_SIZE=12288
    export MLP_FC_QUANT_BITS=4
    export MLP_PROJ_QUANT_BITS=5
    export ATTN_QUANT_BITS=6
    export MLP_MULT=3
    export SWA_ENABLED=1
    export SWA_START_FRAC=0.4
    export TRAIN_SEQ_LEN=2048
    export TRAIN_BATCH_TOKENS=786432
    export MAX_WALLCLOCK_SECONDS=600
    
    # Run training
    torchrun \
        --standalone \
        --nproc_per_node=$NUM_GPUS \
        $SUBMISSION_DIR/train_gpt.py 2>&1 | tee $LOG_FILE
    
    # Copy final log to submission directory
    cp $LOG_FILE $SUBMISSION_DIR/
    
    # Extract final val_bpb
    local VAL_BPB=$(grep "final_int8_zlib_roundtrip_exact" $LOG_FILE | tail -1 | grep -oP 'val_bpb:\K[0-9.]+' || echo "N/A")
    
    echo ""
    echo "Seed $SEED completed at $(date)"
    echo "Final val_bpb: $VAL_BPB"
    echo ""
}

# Run all seeds
for SEED in "${SEEDS[@]}"; do
    run_seed $SEED
done

echo "=========================================="
echo "All seeds completed!"
echo "=========================================="
echo ""
echo "Logs saved to:"
for SEED in "${SEEDS[@]}"; do
    echo "  - $SUBMISSION_DIR/train_seed${SEED}.txt"
done
echo ""
echo "Next step: Run 'python3 verify_submission.py' to check results"
