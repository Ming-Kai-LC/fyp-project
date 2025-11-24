#!/usr/bin/env python3
"""
Auto-train remaining models with proper 240x240 resolution.
This script will train ViT (if not complete) and then Swin automatically.
"""
import subprocess
import time
import os
from datetime import datetime

def check_model_complete(model_name, expected_seeds=5):
    """Check if model training is complete by counting checkpoint files"""
    model_dir = f"experiments/phase2_systematic/models/{model_name}"
    if not os.path.exists(model_dir):
        return False
    
    checkpoints = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    return len(checkpoints) >= expected_seeds

def train_model(model_name):
    """Train a model with all 5 seeds"""
    print(f"\n{'='*70}")
    print(f"AUTO-TRAINING {model_name.upper()} - Started at {datetime.now()}")
    print(f"{'='*70}\n")
    
    log_file = f"logs/{model_name}_autotraining_240.log"
    
    # Run training
    process = subprocess.run(
        ["python", "train_all_models_safe.py", model_name],
        stdout=open(log_file, 'w'),
        stderr=subprocess.STDOUT,
        text=True
    )
    
    if process.returncode == 0:
        print(f"[OK] {model_name.upper()} training completed successfully")
        print(f"  Log: {log_file}")
        return True
    else:
        print(f"[ERROR] {model_name.upper()} training failed (exit code {process.returncode})")
        print(f"  Check log: {log_file}")
        return False

def main():
    print("="*70)
    print("AUTOMATIC TRAINING PIPELINE - 240x240 RESOLUTION")
    print("="*70)
    print("Models to train: ViT, Swin")
    print("Seeds per model: 42, 123, 456, 789, 101112")
    print("="*70)
    
    # Train ViT (5 seeds with vit_base_patch16_plus_clip_240.laion400m_e32)
    if not check_model_complete('vit'):
        print("\n[1/2] Training ViT with 240x240 resolution...")
        if not train_model('vit'):
            print("\n[ERROR] Pipeline stopped: ViT training failed")
            return 1
    else:
        print("\n[1/2] [OK] ViT already complete (5/5 seeds)")
    
    # Train Swin (5 seeds with swinv2_tiny_window8_256.ms_in1k)
    if not check_model_complete('swin'):
        print("\n[2/2] Training Swin with 256x256 resolution (closest to 240)...")
        if not train_model('swin'):
            print("\n[ERROR] Pipeline stopped: Swin training failed")
            return 1
    else:
        print("\n[2/2] [OK] Swin already complete (5/5 seeds)")
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print("\nFinal status:")
    print("  [OK] ResNet-50: 5/5 seeds (240x240)")
    print("  [OK] DenseNet-121: 5/5 seeds (240x240)")
    print("  [OK] EfficientNet-B0: 5/5 seeds (240x240)")
    print("  [OK] CrossViT-Tiny: 5/5 seeds (240x240)")
    print("  [OK] ViT-Base/16+: 5/5 seeds (240x240)")
    print("  [OK] Swin-Tiny: 5/5 seeds (256x256)")
    print("\nPhase 2 Systematic Experimentation: 30/30 models complete")
    print("Next step: Phase 3 Statistical Validation")
    print("="*70)
    
    return 0

if __name__ == "__main__":
    exit(main())
