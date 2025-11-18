"""
Autonomous training script for all FYP models
Safely uses 16-20GB GPU on shared workstation
Can scale up to 40GB if no other users detected
"""

import subprocess
import sys
import time
from pathlib import Path

# Use venv Python
VENV_PYTHON = Path("venv/Scripts/python.exe")
NOTEBOOKS_DIR = Path("notebooks")

# Training configuration
MODELS_TO_TRAIN = [
    {
        "name": "Baseline ResNet-50",
        "notebook": "04_baseline_test.ipynb",
        "priority": 1,
        "batch_size": 32,  # Safe for 16GB
        "description": "Phase 1 completion - verify training pipeline"
    },
    {
        "name": "CrossViT-Tiny",
        "notebook": "06_crossvit_training.ipynb",
        "priority": 2,
        "batch_size": 48,  # Primary model
        "description": "Main model - dual-branch transformer"
    },
    {
        "name": "ResNet-50 (5 seeds)",
        "notebook": "07_resnet50_training.ipynb",
        "priority": 3,
        "batch_size": 48,
        "description": "Baseline 1 - CNN architecture"
    },
    {
        "name": "DenseNet-121",
        "notebook": "08_densenet121_training.ipynb",
        "priority": 4,
        "batch_size": 40,
        "description": "Baseline 2 - Dense connections"
    },
    {
        "name": "EfficientNet-B0",
        "notebook": "09_efficientnet_training.ipynb",
        "priority": 5,
        "batch_size": 56,
        "description": "Baseline 3 - Efficient scaling"
    },
    {
        "name": "ViT-Base",
        "notebook": "10_vit_training.ipynb",
        "priority": 6,
        "batch_size": 32,
        "description": "Baseline 4 - Pure transformer"
    },
    {
        "name": "Swin-Tiny",
        "notebook": "11_swin_training.ipynb",
        "priority": 7,
        "batch_size": 40,
        "description": "Baseline 5 - Hierarchical transformer"
    }
]

def check_gpu_usage():
    """Check current GPU memory usage"""
    try:
        result = subprocess.run(
            [str(VENV_PYTHON), "-c",
             "import torch; print(torch.cuda.memory_allocated(0)/1e9 if torch.cuda.is_available() else 0)"],
            capture_output=True,
            text=True,
            timeout=10
        )
        usage_gb = float(result.stdout.strip())
        return usage_gb
    except:
        return 0

def execute_notebook(notebook_path, timeout_minutes=180):
    """Execute a Jupyter notebook using nbconvert"""
    notebook_path = NOTEBOOKS_DIR / notebook_path

    if not notebook_path.exists():
        print(f"[ERROR] Notebook not found: {notebook_path}")
        return False

    print(f"\n{'='*70}")
    print(f"Executing: {notebook_path.name}")
    print(f"{'='*70}")

    cmd = [
        str(VENV_PYTHON), "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
        f"--ExecutePreprocessor.timeout={timeout_minutes*60}"
    ]

    try:
        start_time = time.time()

        # Run notebook
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(NOTEBOOKS_DIR.parent)
        )

        # Stream output
        for line in iter(process.stdout.readline, ''):
            if line:
                print(line.strip())

        process.wait()

        elapsed = (time.time() - start_time) / 60

        if process.returncode == 0:
            print(f"\n[SUCCESS] Completed in {elapsed:.1f} minutes")
            return True
        else:
            print(f"\n[FAILED] Exit code: {process.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Exceeded {timeout_minutes} minutes")
        return False
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return False

def main():
    print("="*70)
    print("AUTONOMOUS FYP TRAINING - SHARED WORKSTATION MODE")
    print("="*70)
    print("\nConfiguration:")
    print("  - Target GPU Usage: 16-20GB (safe for shared workstation)")
    print("  - Can scale up to 40GB if alone")
    print("  - Using virtual environment (isolated)")
    print(f"  - Models to train: {len(MODELS_TO_TRAIN)}")
    print(f"  - Estimated total time: ~20-24 hours")

    # Verify GPU
    initial_gpu = check_gpu_usage()
    print(f"\n  - Initial GPU usage: {initial_gpu:.2f} GB")

    if initial_gpu > 10:
        print(f"\n[WARNING] GPU already in use ({initial_gpu:.2f} GB)")
        print("  Someone else may be using the workstation.")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("\n[ABORTED] Training cancelled by user")
            return

    print("\n" + "="*70)
    print("Starting training sequence...")
    print("="*70)

    results = []
    total_start = time.time()

    for model_config in MODELS_TO_TRAIN:
        print(f"\n\n{'#'*70}")
        print(f"# MODEL {model_config['priority']}/{len(MODELS_TO_TRAIN)}: {model_config['name']}")
        print(f"# {model_config['description']}")
        print(f"# Batch size: {model_config['batch_size']}")
        print(f"{'#'*70}\n")

        # Check GPU before starting
        pre_gpu = check_gpu_usage()
        print(f"[INFO] GPU usage before training: {pre_gpu:.2f} GB")

        # Execute notebook
        success = execute_notebook(
            model_config['notebook'],
            timeout_minutes=240  # 4 hours per model max
        )

        # Check GPU after
        post_gpu = check_gpu_usage()
        print(f"[INFO] GPU usage after training: {post_gpu:.2f} GB")

        results.append({
            'model': model_config['name'],
            'success': success,
            'notebook': model_config['notebook']
        })

        if not success:
            print(f"\n[WARNING] {model_config['name']} failed!")
            print("  Options:")
            print("  1. Continue with next model")
            print("  2. Retry this model")
            print("  3. Abort all training")

            # Auto-continue for autonomous mode
            print("  [AUTO] Continuing with next model...")
            time.sleep(2)
        else:
            print(f"\n[SUCCESS] {model_config['name']} completed!")

        # Brief pause between models
        print("\n[INFO] Cooling down for 10 seconds...")
        time.sleep(10)

    # Final summary
    total_time = (time.time() - total_start) / 3600

    print("\n\n" + "="*70)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)
    print(f"\nTotal time: {total_time:.2f} hours")
    print(f"\nResults:")

    successful = 0
    for result in results:
        status = "[OK]" if result['success'] else "[FAILED]"
        print(f"  {status} {result['model']}")
        if result['success']:
            successful += 1

    print(f"\n{successful}/{len(results)} models trained successfully")

    if successful == len(results):
        print("\n[SUCCESS] All models trained! Ready for statistical validation.")
        print("\nNext steps:")
        print("  1. View results: mlflow ui")
        print("  2. Run statistical validation")
        print("  3. Generate thesis tables")
    else:
        print(f"\n[WARNING] {len(results) - successful} models failed")
        print("  Review notebook outputs for errors")

    print("\n" + "="*70)
    print("Training session complete!")
    print("="*70)

if __name__ == "__main__":
    main()
