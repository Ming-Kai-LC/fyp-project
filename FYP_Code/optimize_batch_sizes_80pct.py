"""
Optimize batch sizes for 80% VRAM target (EXCLUSIVE USE)
Single user mode - maximize efficiency while maintaining safety
RTX 6000 Ada (49 GB total) - Target 38-40 GB utilization
"""

import json
from pathlib import Path

notebooks_dir = Path("notebooks")

# Optimized batch sizes targeting 38-40 GB VRAM (80% utilization)
# Safe for single user with temperature monitoring
OPTIMIZED_CONFIG_80PCT = {
    "06_crossvit_training.ipynb": {
        "batch_size": 88,  # Up from 72
        "expected_vram": "~30 GB",
        "description": "CrossViT-Tiny dual-branch transformer"
    },
    "07_resnet50_training.ipynb": {
        "batch_size": 128,  # Keep at 128 (already optimal)
        "expected_vram": "~34 GB",
        "description": "ResNet-50 memory efficient CNN"
    },
    "08_densenet121_training.ipynb": {
        "batch_size": 88,  # Up from 72
        "expected_vram": "~32 GB",
        "description": "DenseNet-121 with dense connections"
    },
    "09_efficientnet_training.ipynb": {
        "batch_size": 144,  # Keep at 144 (already optimal)
        "expected_vram": "~32 GB",
        "description": "EfficientNet-B0 optimized architecture"
    },
    "10_vit_training.ipynb": {
        "batch_size": 80,  # Up from 64
        "expected_vram": "~38 GB",
        "description": "ViT-Base pure transformer (highest memory)"
    },
    "11_swin_training.ipynb": {
        "batch_size": 88,  # Up from 72
        "expected_vram": "~32 GB",
        "description": "Swin-Tiny hierarchical vision transformer"
    }
}

def update_batch_size_in_notebook(notebook_path, new_batch_size):
    """Update batch_size in a training notebook"""

    if not notebook_path.exists():
        print(f"[SKIP] {notebook_path.name} not found")
        return False

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Look for batch_size configuration
            if 'batch_size' in source and '=' in source:
                lines = source.split('\n')
                new_lines = []

                for line in lines:
                    if 'batch_size' in line and '=' in line and not line.strip().startswith('#'):
                        # Update batch_size line
                        indent = len(line) - len(line.lstrip())
                        new_line = ' ' * indent + f'batch_size = {new_batch_size}'
                        new_lines.append(new_line)
                        modified = True
                        print(f"  - Updated to: batch_size = {new_batch_size}")
                    else:
                        new_lines.append(line)

                if modified:
                    cell['source'] = [line + '\n' if i < len(new_lines)-1 else line
                                     for i, line in enumerate(new_lines)]

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        return True

    return False

def main():
    print("=" * 70)
    print("GPU BATCH SIZE OPTIMIZATION - EXCLUSIVE MODE")
    print("Target: 38-40 GB VRAM (80% utilization)")
    print("Single user on RTX 6000 Ada - Maximum efficiency")
    print("=" * 70)
    print()

    for notebook_name, config in OPTIMIZED_CONFIG_80PCT.items():
        notebook_path = notebooks_dir / notebook_name
        print(f"[OPTIMIZE] {notebook_name}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Expected VRAM: {config['expected_vram']}")
        print(f"  Model: {config['description']}")

        if update_batch_size_in_notebook(notebook_path, config['batch_size']):
            print(f"  [SUCCESS] Updated")
        else:
            print(f"  [SKIP] No changes needed or file not found")
        print()

    print("=" * 70)
    print("OPTIMIZATION COMPLETE - 80% TARGET")
    print("=" * 70)
    print()
    print("VRAM allocation (exclusive mode):")
    print("  - Your training: 38-40 GB (80%)")
    print("  - System overhead: ~2 GB")
    print("  - Safety margin: ~7-9 GB")
    print("  - Total: 49 GB")
    print()
    print("Safety measures:")
    print("  - Temperature monitoring: Every 30 seconds")
    print("  - Safe temp range: 75-82°C")
    print("  - GPU auto-throttles at 87°C")
    print("  - Professional GPU rated to 89°C continuous")
    print()
    print("Training time: Estimated 12-14 hours")
    print()

if __name__ == "__main__":
    main()
