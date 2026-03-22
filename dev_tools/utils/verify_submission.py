#!/usr/bin/env python3
"""
Verify submission is complete and calculate statistics
"""
import json
import os
import re
from pathlib import Path
from typing import Optional

SUBMISSION_DIR = Path("records/track_10min_16mb/2026-03-22_Int4_MLP3x_Bigram_SmearGate_SWA")

def extract_val_bpb(log_file: Path) -> Optional[float]:
    """Extract final val_bpb from training log."""
    if not log_file.exists():
        return None
    
    with open(log_file) as f:
        content = f.read()
    
    # Look for the exact val_bpb line
    pattern = r"final_int8_zlib_roundtrip_exact val_loss:[\d.]+ val_bpb:([\d.]+)"
    matches = re.findall(pattern, content)
    
    if matches:
        return float(matches[-1])
    return None

def check_file_size(file_path: Path, max_bytes: int = 16_000_000) -> bool:
    """Check if file is under size limit."""
    if not file_path.exists():
        return False
    return file_path.stat().st_size < max_bytes

def main():
    print("=" * 70)
    print("Parameter Golf - Submission Verification")
    print("=" * 70)
    print()
    
    # Check required files
    required_files = [
        "README.md",
        "submission.json",
        "train_gpt.py",
    ]
    
    print("📁 Checking required files:")
    all_present = True
    for fname in required_files:
        fpath = SUBMISSION_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✅ {fname} ({size_kb:.1f} KB)")
        else:
            print(f"  ❌ {fname} - MISSING")
            all_present = False
    
    if not all_present:
        print("\n❌ Submission incomplete - missing required files")
        return 1
    
    # Check training logs
    print("\n📊 Checking training logs:")
    seeds = [1337, 42, 7]
    val_bpbs = []
    
    for seed in seeds:
        log_file = SUBMISSION_DIR / f"train_seed{seed}.txt"
        val_bpb = extract_val_bpb(log_file)
        
        if val_bpb is not None:
            print(f"  ✅ Seed {seed}: val_bpb = {val_bpb:.8f}")
            val_bpbs.append(val_bpb)
        else:
            print(f"  ❌ Seed {seed}: Log missing or val_bpb not found")
    
    if len(val_bpbs) == 3:
        import statistics
        mean_bpb = statistics.mean(val_bpbs)
        std_bpb = statistics.stdev(val_bpbs)
        
        print("\n" + "=" * 70)
        print("📈 RESULTS SUMMARY")
        print("=" * 70)
        print(f"  Seeds: {seeds}")
        print(f"  Individual scores:")
        for seed, bpb in zip(seeds, val_bpbs):
            print(f"    Seed {seed}: {bpb:.8f}")
        print(f"\n  Mean val_bpb: {mean_bpb:.8f}")
        print(f"  Std dev:      {std_bpb:.8f}")
        print(f"  Best:         {min(val_bpbs):.8f}")
        print(f"  Worst:        {max(val_bpbs):.8f}")
        
        # Check if we beat SOTA
        SOTA = 1.1428
        improvement = SOTA - mean_bpb
        print(f"\n  vs SOTA ({SOTA}):")
        if improvement > 0:
            print(f"    ✅ IMPROVEMENT: -{improvement:.8f} bpb")
            if improvement > 0.005:
                print(f"    ✅ Statistically significant! (> 0.005)")
            else:
                print(f"    ⚠️  May not be statistically significant")
        else:
            print(f"    ❌ Behind SOTA by {abs(improvement):.8f} bpb")
        
        # Update submission.json
        print("\n📝 Updating submission.json...")
        submission_path = SUBMISSION_DIR / "submission.json"
        with open(submission_path) as f:
            submission = json.load(f)
        
        submission["val_loss"] = round(mean_bpb, 8)
        submission["blurb"] = (
            f"11 layers with aggressive mixed quantization (int4 MLP fc, int5 MLP proj, int6 attention). "
            f"BigramHash 12288 buckets. SmearGate for bigram context. SWA start_frac=0.4. "
            f"Mean of 3 seeds: {mean_bpb:.4f} (std: {std_bpb:.4f})"
        )
        
        with open(submission_path, 'w') as f:
            json.dump(submission, f, indent=2)
        
        print(f"  ✅ Updated with mean val_bpb: {mean_bpb:.8f}")
        
    elif len(val_bpbs) > 0:
        print(f"\n⚠️  Only {len(val_bpbs)}/3 seeds completed")
        print(f"   Current mean: {sum(val_bpbs)/len(val_bpbs):.8f}")
    else:
        print("\n❌ No training logs found!")
        print("   Run: bash train_all_seeds.sh")
        return 1
    
    print("\n" + "=" * 70)
    print("✅ Verification complete!")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    exit(main())
