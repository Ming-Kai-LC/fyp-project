"""
Auto-monitor and train remaining models sequentially
Monitors ViT training and automatically starts Swin when complete
"""

import time
import subprocess
import os
from pathlib import Path

def check_vit_complete():
    """Check if ViT has completed all 5 seeds"""
    vit_dir = Path("experiments/phase2_systematic/models/vit")
    if not vit_dir.exists():
        return False
    pth_files = list(vit_dir.glob("*.pth"))
    return len(pth_files) >= 5

def check_vit_running():
    """Check if ViT training process is still running"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        return 'train_all_models_safe.py vit' in result.stdout
    except:
        return True  # Assume running if can't check

def train_swin():
    """Start Swin training"""
    print("\n" + "="*70)
    print("ViT COMPLETE! Starting Swin-Tiny training...")
    print("="*70 + "\n")

    cmd = "python train_all_models_safe.py swin"
    subprocess.run(cmd, shell=True)

    print("\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)

def main():
    print("="*70)
    print("AUTO-TRAINING MONITOR")
    print("="*70)
    print("\nWaiting for ViT-Base to complete...")
    print("Then will automatically start Swin-Tiny\n")

    check_interval = 60  # Check every 60 seconds

    while True:
        if check_vit_complete():
            print("\n[OK] ViT-Base training complete!")
            break

        if not check_vit_running():
            print("\n[WARNING] ViT process not detected but models not complete")
            print("   Waiting for completion...")

        time.sleep(check_interval)

    # Wait a bit to ensure files are fully written
    time.sleep(5)

    # Start Swin
    train_swin()

if __name__ == "__main__":
    main()
