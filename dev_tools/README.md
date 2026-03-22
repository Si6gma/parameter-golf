# Dev Tools

Helper scripts for training. Not part of official submission.

## Quick Start

```bash
# Cloud setup
bash scripts/cloud_setup.sh

# Train all 3 seeds
bash scripts/train_all_seeds.sh

# Verify results
python3 utils/verify_submission.py
```

## Structure

```
dev_tools/
├── NOTES.md           # Dev notes & hyperparameters
├── scripts/           # Cloud training scripts
│   ├── cloud_setup.sh
│   ├── deploy_hyperbolic.sh
│   ├── smoke_test_cloud.sh
│   └── train_all_seeds.sh
└── utils/             # Helpers
    ├── smoke_test.py
    └── verify_submission.py
```

## Official Submission

```
records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA/
├── train_gpt.py       # Training script
├── submission.json    # Results metadata
└── requirements.txt   # Dependencies
```
