#!/usr/bin/env python3
"""
Smoke test for Parameter Golf submission
Runs a tiny model locally to verify code works before burning $$$ on H100s
"""
import os
import sys
import subprocess

# Force tiny settings for fast local test
os.environ["NUM_LAYERS"] = "2"
os.environ["MODEL_DIM"] = "128"
os.environ["NUM_HEADS"] = "4"
os.environ["NUM_KV_HEADS"] = "2"
os.environ["MLP_MULT"] = "2"
os.environ["BIGRAM_VOCAB_SIZE"] = "512"
os.environ["BIGRAM_DIM"] = "32"
os.environ["TRAIN_SEQ_LEN"] = "128"
os.environ["TRAIN_BATCH_TOKENS"] = "4096"
os.environ["ITERATIONS"] = "10"
os.environ["WARMUP_STEPS"] = "2"
os.environ["VAL_LOSS_EVERY"] = "5"
os.environ["TRAIN_LOG_EVERY"] = "2"
os.environ["MLX_EAGER_EVAL"] = "1"

print("=" * 70)
print("PARAMETER GOLF SMOKE TEST")
print("=" * 70)
print("\nThis will run a tiny model to verify the code works.")
print("Expected: ~30-60 seconds on Mac\n")

# Use the improved MLX version
script_path = "train_gpt_mlx_improved.py"
if not os.path.exists(script_path):
    print(f"❌ {script_path} not found!")
    sys.exit(1)

print(f"Running: {script_path}")
print(f"Settings: 2 layers, 128 dim, 128 seq_len, 10 iterations\n")

try:
    result = subprocess.run(
        ["python3", script_path],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    # Check output
    if result.returncode == 0:
        print("✅ Script executed without errors!")
        
        # Look for key indicators in output
        output = result.stdout + result.stderr
        
        checks = [
            ("model_params", "Model initialized"),
            ("smeargate", "SmearGate working"),
            ("bigram", "BigramHash working"),
            ("swa", "SWA enabled"),
            ("step:10", "Completed 10 steps"),
        ]
        
        print("\n📊 Checks:")
        for key, desc in checks:
            if key in output.lower():
                print(f"  ✅ {desc}")
            else:
                print(f"  ⚠️  {desc} - not verified")
        
        # Show final loss if available
        for line in output.split('\n'):
            if 'step:10' in line and 'loss:' in line:
                print(f"\n📉 Final: {line.strip()}")
                break
        
        print("\n" + "=" * 70)
        print("SMOKE TEST PASSED ✅")
        print("=" * 70)
        print("\nCode looks good! Ready for H100 training.")
        print("Next step: Deploy to RunPod with 8xH100s")
        
    else:
        print("❌ Script failed!")
        print("\nSTDERR:")
        print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("❌ Timeout (>2 minutes) - something's wrong")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
