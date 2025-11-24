#!/usr/bin/env python3
"""
Auto-continuation script: Starts Swin training after ViT completes.
Monitors ViT completion and automatically launches Swin.
"""
import subprocess
import time
import os
from datetime import datetime

def check_vit_complete():
    """Check if ViT training is complete (5 models saved)"""
    vit_dir = "experiments/phase2_systematic/models/vit"
    if not os.path.exists(vit_dir):
        return False
    
    checkpoints = [f for f in os.listdir(vit_dir) if f.endswith('.pth')]
    return len(checkpoints) >= 5

def is_training_running():
    """Check if any training process is running"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        return "train_all_models_safe.py vit" in result.stdout
    except:
        return False

def main():
    print("="*70)
    print("AUTO-CONTINUATION MONITOR - ViT -> Swin")
    print("="*70)
    print(f"Started at: {datetime.now()}")
    print("\nWaiting for ViT training to complete...")
    print("Checking every 5 minutes...\n")
    
    check_count = 0
    while True:
        check_count += 1
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if ViT is complete
        if check_vit_complete() and not is_training_running():
            print(f"\n[{current_time}] ViT training COMPLETED!")
            
            # Count models
            vit_models = len([f for f in os.listdir("experiments/phase2_systematic/models/vit") 
                            if f.endswith('.pth')])
            print(f"  Found {vit_models}/5 ViT models")
            
            print("\n" + "="*70)
            print("STARTING SWIN TRAINING AUTOMATICALLY")
            print("="*70)
            
            # Start Swin training
            log_file = "logs/swin_training_240_auto.log"
            print(f"Log file: {log_file}\n")
            
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    ["python", "train_all_models_safe.py", "swin"],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            print(f"Swin training started (PID: {process.pid})")
            print(f"Monitor with: tail -f {log_file}")
            print("\nAuto-continuation complete!")
            break
        
        # Status update
        vit_models = 0
        if os.path.exists("experiments/phase2_systematic/models/vit"):
            vit_models = len([f for f in os.listdir("experiments/phase2_systematic/models/vit") 
                            if f.endswith('.pth')])
        
        training_status = "RUNNING" if is_training_running() else "IDLE"
        print(f"[{current_time}] Check #{check_count}: ViT {vit_models}/5 models | Status: {training_status}")
        
        # Wait 5 minutes before next check
        time.sleep(300)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
