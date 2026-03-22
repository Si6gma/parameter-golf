# Development Tools

This folder contains helper scripts and documentation for developing and testing Parameter Golf submissions. These files are **NOT** part of the official submission.

## Folder Structure

```
dev_tools/
├── scripts/          # Shell scripts for cloud training
│   ├── cloud_setup.sh
│   ├── smoke_test_cloud.sh
│   └── train_all_seeds.sh
│
├── training/         # Experimental training scripts
│   ├── train_gpt_improved.py      # CUDA with int4/5/6
│   ├── train_gpt_advanced.py      # CUDA with Test-Time Training
│   └── train_gpt_mlx_improved.py  # MLX for Mac
│
├── utils/            # Helper utilities
│   ├── smoke_test.py              # Local smoke test
│   └── verify_submission.py       # Result verification
│
└── docs/             # Documentation
    ├── STRATEGY.md
    ├── CLOUD_SETUP.md
    └── SETUP_SUMMARY.md
```

## Official Submission

The actual submission is in:
```
records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/
├── README.md
├── submission.json
├── requirements.txt
└── train_gpt.py
```

## Quick Start

### Local Testing (Mac)
```bash
cd dev_tools
python3 utils/smoke_test.py
```

### Cloud Training
```bash
# On cloud instance
cd dev_tools
bash scripts/cloud_setup.sh
bash scripts/smoke_test_cloud.sh
bash scripts/train_all_seeds.sh
python3 utils/verify_submission.py
```

### Copy to Submission

If you improve the code, copy to submission:
```bash
cp dev_tools/training/train_gpt_improved.py \
   records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/train_gpt.py
```

## Documentation

- `docs/STRATEGY.md` - Original strategy document
- `docs/CLOUD_SETUP.md` - Full cloud training guide  
- `docs/SETUP_SUMMARY.md` - Quick reference
