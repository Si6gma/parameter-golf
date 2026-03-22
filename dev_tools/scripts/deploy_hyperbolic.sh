#!/bin/bash
# Deployment script for Hyperbolic 8x H100
# Run this AFTER SSHing into the instance

set -e

echo "=========================================="
echo "Parameter Golf - Hyperbolic Deployment"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

# Check GPUs
echo "🔍 Checking GPUs..."
nvidia-smi --query-gpu=name,count --format=csv,noheader
echo ""

# Clone repo (if not already done)
if [ ! -d "parameter-golf" ]; then
    echo "📦 Cloning repository..."
    git clone https://github.com/Si6gma/parameter-golf.git
fi

cd parameter-golf

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q torch numpy sentencepiece zstandard tqdm

# Download dataset
echo ""
echo "📥 Downloading dataset..."
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo "✅ Dataset already present"
fi

# Quick smoke test
echo ""
echo "🧪 Running smoke test..."
cd records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA

# Run mini test
export NUM_LAYERS=2
export MODEL_DIM=128
export TRAIN_BATCH_TOKENS=8192
export ITERATIONS=5
export MAX_WALLCLOCK_SECONDS=60

timeout 90 torchrun --standalone --nproc_per_node=8 train_gpt.py 2>&1 | tail -20

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "=========================================="
echo ""
echo "To start full training:"
echo "  cd parameter-golf"
echo "  source .venv/bin/activate"
echo "  bash dev_tools/scripts/train_all_seeds.sh"
echo ""
