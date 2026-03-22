# Hyperbolic 8x H100 Quickstart

Your instance: `medium-dracena-ostrich` @ uk-southeast-3

## Once SSH is Available

```bash
# 1. SSH into the instance
ssh root@<ip-address>  # (IP will show when ready)

# 2. Deploy everything
wget https://raw.githubusercontent.com/Si6gma/parameter-golf/main/dev_tools/scripts/deploy_hyperbolic.sh
bash deploy_hyperbolic.sh

# 3. Run full training
cd parameter-golf
bash dev_tools/scripts/train_all_seeds.sh

# 4. Check results
python3 dev_tools/utils/verify_submission.py
```

## Manual Training (if needed)

```bash
cd parameter-golf/records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA

# Single seed
torchrun --standalone --nproc_per_node=8 train_gpt.py

# With custom settings
EMA_DECAY=0.999 EMA_START_STEP=100 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Monitoring

```bash
# Watch GPUs
watch -n 1 nvidia-smi

# Watch training progress
tail -f logs/train_seed1337.txt | grep "step:"
```

## Expected Results

- **Time per seed**: ~10 minutes
- **Total for 3 seeds**: ~35-40 minutes
- **Target score**: ~1.130 bpb
- **Cost**: ~$12 (at $2.19/GPU/hr × 8 GPUs × 0.6 hr)

## If Something Fails

1. Check GPUs: `nvidia-smi`
2. Check PyTorch CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
3. Check dataset: `ls data/datasets/fineweb10B_sp1024/`
4. Run smoke test: `bash dev_tools/scripts/smoke_test_cloud.sh`

## Downloading Results

```bash
# From your local machine
scp -r root@<ip-address>:~/parameter-golf/records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/*.txt ./
```

Good luck! 🚀
