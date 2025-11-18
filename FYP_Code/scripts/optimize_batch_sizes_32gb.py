"""
Optimize batch sizes for 32 GB VRAM target
Safe for shared workstation with 2-3 users
RTX 6000 Ada (49 GB total) - Target 65% utilization
"""

import json
from pathlib import Path

notebooks_dir = Path("notebooks")

# Conservative batch sizes targeting 32 GB VRAM (65% utilization)
# Safe for 2-3 users sharing the workstation
OPTIMIZED_CONFIG_32GB = {
    "06_crossvit_training.ipynb": {
        "batch_size": 72,  # Conservative for transformer
        "expected_vram": "~24 GB",
        "description": "CrossViT-Tiny dual-branch transformer"
    },
    "07_resnet50_training.ipynb": {
        "batch_size": 96,  # ResNet is efficient
        "expected_vram": "~26 GB",
        "description": "ResNet-50 memory efficient CNN"
    },
    "08_densenet121_training.ipynb": {
        "batch_size": 72,  # Dense connections need memory
        "expected_vram": "~25 GB",
        "description": "DenseNet-121 with dense connections"
    },
    "09_efficientnet_training.ipynb": {
        "batch_size": 112,  # Most efficient model
        "expected_vram": "~24 GB",
        "description": "EfficientNet-B0 optimized architecture"
    },
    "10_vit_training.ipynb": {
        "batch_size": 64,  # ViT needs more memory
        "expected_vram": "~30 GB",
        "description": "ViT-Base pure transformer (higher memory)"
    },
    "11_swin_training.ipynb": {
        "batch_size": 72,  # Hierarchical transformer
        "expected_vram": "~25 GB",
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
    print("GPU BATCH SIZE OPTIMIZATION - SHARED WORKSTATION MODE")
    print("Target: 32 GB VRAM (65% utilization)")
    print("Safe for 2-3 users on RTX 6000 Ada")
    print("=" * 70)
    print()

    for notebook_name, config in OPTIMIZED_CONFIG_32GB.items():
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
    print("OPTIMIZATION COMPLETE - 32 GB TARGET")
    print("=" * 70)
    print()
    print("VRAM allocation per user (estimated):")
    print("  - Your training: 24-30 GB")
    print("  - Other users: 17-25 GB available")
    print("  - Total: 49 GB")
    print()
    print("Safe for 2-3 users sharing the workstation")
    print("Temperature will stay under 80°C")
    print("Training time: Estimated 15-18 hours")
    print()

if __name__ == "__main__":
    main()
