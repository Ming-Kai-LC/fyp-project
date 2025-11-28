"""
Automated Training Pipeline
============================
Runs weak baselines (VGG-16, MobileNetV2) followed by CrossViT enhanced training.

Author: Tan Ming Kai (24PMR12003)
Date: 2025-11-26
"""

import subprocess
import sys
import time
from pathlib import Path

def run_script(script_name):
    """Run a Python script and return success status."""
    print(f"\n{'='*70}")
    print(f"STARTING: {script_name}")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            cwd=Path(__file__).parent.parent,
            check=True
        )
        elapsed = (time.time() - start_time) / 60
        print(f"\n[OK] {script_name} completed in {elapsed:.1f} minutes")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {script_name} failed: {e}")
        return False

def main():
    print("="*70)
    print("AUTOMATED TRAINING PIPELINE")
    print("="*70)
    print("\nThis script will run:")
    print("  1. Weak baselines (VGG-16, MobileNetV2) - 5 seeds each")
    print("  2. CrossViT enhanced (Base, Small) - 5 seeds each")
    print("  3. Ensemble and TTA evaluation")
    print("\n" + "="*70)

    total_start = time.time()

    # Step 1: Run weak baselines
    print("\n" + "="*70)
    print("STEP 1/2: WEAK BASELINES TRAINING")
    print("="*70)

    weak_success = run_script("scripts/train_weak_baselines.py")

    if not weak_success:
        print("\n[WARNING] Weak baselines had issues, but continuing with CrossViT...")

    # Step 2: Run CrossViT enhanced
    print("\n" + "="*70)
    print("STEP 2/2: CROSSVIT ENHANCED TRAINING")
    print("="*70)

    crossvit_success = run_script("scripts/train_crossvit_enhanced.py")

    # Summary
    total_elapsed = (time.time() - total_start) / 60

    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"\nTotal time: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
    print(f"\nResults:")
    print(f"  Weak baselines: {'SUCCESS' if weak_success else 'FAILED'}")
    print(f"  CrossViT enhanced: {'SUCCESS' if crossvit_success else 'FAILED'}")
    print(f"\nCheck results in:")
    print(f"  - results/vgg16_results.csv")
    print(f"  - results/mobilenetv2_results.csv")
    print(f"  - results/crossvitbase_results.csv")
    print(f"  - results/crossvitsmall_results.csv")
    print("="*70)

if __name__ == "__main__":
    main()
