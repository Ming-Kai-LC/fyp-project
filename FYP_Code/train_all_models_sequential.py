"""
Automatic Sequential Model Training Script
Trains all remaining models one after another without manual intervention.

This script will:
1. Check which models are already complete (have 5 .pth files)
2. Train remaining models in sequence
3. Skip to next model if current one completes or fails
4. Automatically maximize GPU utilization

Usage: python train_all_models_sequential.py
"""

import os
import subprocess
import time
from pathlib import Path
import sys

# Model training order
MODELS = ['densenet121', 'efficientnet', 'vit', 'swin', 'crossvit']
MODELS_DIR = Path("experiments/phase2_systematic/models")

def check_model_complete(model_name):
    """Check if model has all 5 seed files"""
    model_dir = MODELS_DIR / model_name
    if not model_dir.exists():
        return False

    pth_files = list(model_dir.glob("*.pth"))
    return len(pth_files) >= 5

def get_pending_models():
    """Get list of models that haven't been fully trained yet"""
    pending = []
    for model in MODELS:
        if not check_model_complete(model):
            pending.append(model)
    return pending

def train_model(model_name):
    """Train a single model with all 5 seeds"""
    print(f"\n{'='*80}")
    print(f"[*] STARTING: {model_name.upper()}")
    print(f"{'='*80}\n")

    log_file = f"logs/{model_name}_live_training.log"

    # Run training
    cmd = f"python train_all_models_safe.py {model_name}"

    try:
        # Run with live output
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # Stream output to both console and log file
        with open(log_file, 'w') as f:
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()

        process.wait()

        if process.returncode == 0:
            print(f"\n[OK] {model_name.upper()} completed successfully!\n")
            return True
        else:
            print(f"\n[FAIL] {model_name.upper()} failed with return code {process.returncode}\n")
            return False

    except Exception as e:
        print(f"\n[ERROR] training {model_name}: {e}\n")
        return False

def main():
    """Main training loop"""
    print("\n" + "="*80)
    print("AUTOMATIC SEQUENTIAL MODEL TRAINING")
    print("="*80)

    # Check which models need training
    pending = get_pending_models()

    if not pending:
        print("\n[OK] All models already trained! Nothing to do.\n")
        return

    print(f"\n[*] Models to train: {', '.join([m.upper() for m in pending])}")
    print(f"[*] Total models: {len(pending)}")
    print(f"[*] Estimated time: ~{len(pending) * 2.5:.1f} hours\n")

    completed = []
    failed = []

    start_time = time.time()

    for i, model in enumerate(pending, 1):
        print(f"\n{'='*80}")
        print(f"Progress: {i}/{len(pending)} models")
        print(f"{'='*80}\n")

        success = train_model(model)

        if success:
            completed.append(model)
        else:
            failed.append(model)
            print(f"[WARN] {model.upper()} failed, continuing to next model...")

    # Final summary
    total_time = time.time() - start_time
    hours = total_time / 3600

    print("\n" + "="*80)
    print("[*] TRAINING COMPLETE!")
    print("="*80)
    print(f"\n[OK] Completed: {len(completed)}/{len(pending)} models")
    print(f"     {', '.join([m.upper() for m in completed])}")

    if failed:
        print(f"\n[FAIL] Failed: {len(failed)} models")
        print(f"       {', '.join([m.upper() for m in failed])}")

    print(f"\n[*] Total time: {hours:.2f} hours")
    print(f"[*] Results saved to: experiments/phase2_systematic/")
    print(f"[*] Logs saved to: logs/")
    print("\n" + "="*80 + "\n")

    # List all completed models
    print("\n[*] All Trained Models:")
    for model in MODELS:
        if check_model_complete(model):
            model_dir = MODELS_DIR / model
            pth_count = len(list(model_dir.glob("*.pth")))
            print(f"    [OK] {model.upper()}: {pth_count} seeds")
        else:
            print(f"    [..] {model.upper()}: Incomplete")

    print()

if __name__ == "__main__":
    main()
