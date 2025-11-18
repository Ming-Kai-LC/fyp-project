"""
Shared Workstation GPU Resource Manager
Ensures fair GPU usage: 1 user = 80-90%, 2 users = 50%, 3 users = 33%
"""

import subprocess
import re
import torch

def get_gpu_processes():
    """Get list of GPU compute processes (excluding Windows GUI)"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        output = result.stdout

        # Count unique Python processes using GPU for compute
        python_processes = []
        for line in output.split('\n'):
            if 'python' in line.lower() and 'C' in line:  # C = Compute process
                python_processes.append(line)

        return len(python_processes)
    except:
        return 1  # Conservative estimate if can't detect

def get_gpu_memory_info():
    """Get total and used GPU memory in GB"""
    if not torch.cuda.is_available():
        return 0, 0

    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9

    return total_memory, reserved

def calculate_fair_batch_size(base_batch_size=64, num_users=None):
    """
    Calculate fair batch size based on number of users

    Workstation rules:
    - 1 user: use 80-90% of GPU (reduce batch slightly for safety)
    - 2 users: use 50% of GPU
    - 3 users: use 33% of GPU
    """
    if num_users is None:
        num_users = max(1, get_gpu_processes())

    total_vram, used_vram = get_gpu_memory_info()

    if num_users == 1:
        # Use 85% of resources (conservative for safety)
        scale = 0.85
        print(f"[OK] 1 user detected - Using 85% of GPU resources")
    elif num_users == 2:
        scale = 0.50
        print(f"[WARN] 2 users detected - Using 50% of GPU resources (fair share)")
    else:  # 3+ users
        scale = 0.33
        print(f"[WARN] {num_users} users detected - Using 33% of GPU resources (fair share)")

    # Calculate batch size
    fair_batch_size = max(8, int(base_batch_size * scale))  # Minimum batch size = 8

    print(f"   Total VRAM: {total_vram:.1f} GB")
    print(f"   Currently used: {used_vram:.1f} GB")
    print(f"   Fair allocation: {total_vram * scale:.1f} GB")
    print(f"   Recommended batch size: {fair_batch_size}")

    return fair_batch_size, num_users

def get_safe_config(model_name, base_config):
    """
    Get safe training configuration based on current GPU usage

    Model VRAM requirements (approximate for batch_size=64):
    - ResNet-50: ~6 GB
    - DenseNet-121: ~7 GB
    - EfficientNet-B0: ~5 GB
    - ViT-Base: ~10 GB
    - Swin-Tiny: ~8 GB
    - CrossViT-Tiny: ~7 GB
    """
    fair_batch, num_users = calculate_fair_batch_size(base_config.get('batch_size', 64))

    safe_config = base_config.copy()
    safe_config['batch_size'] = fair_batch
    safe_config['num_users_detected'] = num_users

    # Adjust num_workers based on batch size
    if fair_batch <= 16:
        safe_config['num_workers'] = 2
    elif fair_batch <= 32:
        safe_config['num_workers'] = 4
    else:
        safe_config['num_workers'] = 6

    return safe_config

def print_gpu_status():
    """Print current GPU status"""
    print("\n" + "="*70)
    print("GPU STATUS CHECK - Shared Workstation")
    print("="*70)

    total_vram, used_vram = get_gpu_memory_info()
    num_processes = get_gpu_processes()

    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"Total VRAM: {total_vram:.2f} GB")
    print(f"Used VRAM: {used_vram:.2f} GB ({used_vram/total_vram*100:.1f}%)")
    print(f"Free VRAM: {total_vram - used_vram:.2f} GB")
    print(f"Active compute processes: {num_processes}")

    if num_processes <= 1:
        print("[OK] You're the only user - Can use 80-90% of resources")
    elif num_processes == 2:
        print("[WARN] 2 users active - Fair share is 50% of resources")
    else:
        print(f"[WARN] {num_processes} users active - Fair share is 33% of resources")

    print("="*70 + "\n")

if __name__ == "__main__":
    print_gpu_status()

    # Example usage
    base_config = {
        'batch_size': 64,
        'num_workers': 4,
    }

    safe_config = get_safe_config("ResNet-50", base_config)
    print(f"\nSafe configuration: {safe_config}")
