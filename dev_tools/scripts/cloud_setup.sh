#!/bin/bash
# Cloud Setup Script for Parameter Golf
# Run this on your cloud GPU instance (8xH100) to set everything up

set -e  # Exit on error

echo "=========================================="
echo "Parameter Golf - Cloud Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're on CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: nvidia-smi not found. Are you on a GPU instance?${NC}"
    exit 1
fi

echo -e "${GREEN}✓ GPUs detected:${NC}"
nvidia-smi --query-gpu=name,count --format=csv,noheader

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q torch numpy sentencepiece zstandard tqdm

# Download dataset if not present
if [ ! -d "./data/datasets/fineweb10B_sp1024" ]; then
    echo ""
    echo "Downloading FineWeb dataset..."
    mkdir -p data/datasets data/tokenizers
    python3 data/cached_challenge_fineweb.py --variant sp1024
else
    echo -e "${GREEN}✓ Dataset already present${NC}"
fi

# Verify submission files
echo ""
echo "Verifying submission files..."
SUBMISSION_DIR="records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA"

if [ ! -f "$SUBMISSION_DIR/train_gpt.py" ]; then
    echo -e "${RED}ERROR: train_gpt.py not found in submission directory${NC}"
    exit 1
fi

if [ ! -f "$SUBMISSION_DIR/submission.json" ]; then
    echo -e "${RED}ERROR: submission.json not found${NC}"
    exit 1
fi

if [ ! -f "$SUBMISSION_DIR/README.md" ]; then
    echo -e "${RED}ERROR: README.md not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Submission files verified${NC}"

# Create logs directory
mkdir -p logs
mkdir -p $SUBMISSION_DIR

echo ""
echo "=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Run smoke test:   bash smoke_test_cloud.sh"
echo "  2. Train 3 seeds:    bash train_all_seeds.sh"
echo "  3. Verify results:   python3 verify_submission.py"
echo ""
