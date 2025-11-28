"""
Monitor and Run CrossViT
=========================
Waits for weak baselines to complete, then runs CrossViT training.

Usage: python scripts/run_crossvit_after_baselines.py
"""

import time
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
REQUIRED_FILES = [
    "vgg16_results.csv",
    "mobilenetv2_results.csv"
]

def check_baselines_complete():
    """Check if weak baselines results exist."""
    for f in REQUIRED_FILES:
        path = RESULTS_DIR / f
        if not path.exists():
            return False, f
    return True, None

def count_results(filepath):
    """Count number of results in CSV."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            return len(lines) - 1  # Subtract header
    except:
        return 0

def main():
    print("="*70)
    print("CROSSVIT TRAINING MONITOR")
    print("="*70)
    print("\nWaiting for weak baselines training to complete...")
    print("Required files:")
    for f in REQUIRED_FILES:
        print(f"  - {RESULTS_DIR / f}")
    print("\nChecking every 60 seconds...")
    print("="*70)

    check_count = 0
    while True:
        complete, missing = check_baselines_complete()

        if complete:
            # Verify all 5 seeds are complete
            vgg_count = count_results(RESULTS_DIR / "vgg16_results.csv")
            mobile_count = count_results(RESULTS_DIR / "mobilenetv2_results.csv")

            if vgg_count >= 5 and mobile_count >= 5:
                print(f"\n[OK] Weak baselines complete!")
                print(f"  VGG-16: {vgg_count} seeds")
                print(f"  MobileNetV2: {mobile_count} seeds")
                break
            else:
                print(f"\rWaiting... VGG-16: {vgg_count}/5, MobileNetV2: {mobile_count}/5", end="", flush=True)
        else:
            check_count += 1
            print(f"\rCheck #{check_count}: Waiting for {missing}...", end="", flush=True)

        time.sleep(60)

    print("\n" + "="*70)
    print("STARTING CROSSVIT ENHANCED TRAINING")
    print("="*70)

    # Run CrossViT training
    result = subprocess.run(
        [sys.executable, "scripts/train_crossvit_enhanced.py"],
        cwd=Path(__file__).parent.parent
    )

    if result.returncode == 0:
        print("\n[OK] CrossViT training complete!")
    else:
        print(f"\n[ERROR] CrossViT training failed with code {result.returncode}")

if __name__ == "__main__":
    main()
