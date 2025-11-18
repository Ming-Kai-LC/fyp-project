"""
Autonomous FYP Workflow Manager
Monitors preprocessing, then automatically starts training all models
Safe for shared workstation (16-20GB GPU target)
"""

import subprocess
import time
from pathlib import Path
import sys

VENV_PYTHON = Path("venv/Scripts/python.exe")
NOTEBOOKS_DIR = Path("notebooks")

def check_preprocessing_done():
    """Check if CLAHE preprocessing is complete"""
    processed_dir = Path("data/processed/clahe_enhanced")

    if not processed_dir.exists():
        return False, 0

    # Count processed images
    count = 0
    for class_dir in ["COVID", "Normal", "Lung_Opacity", "Viral Pneumonia"]:
        train_dir = processed_dir / "train" / class_dir
        if train_dir.exists():
            count += len(list(train_dir.glob("*.png")))

    # Should have ~16,931 training images
    expected = 16931
    progress = (count / expected) * 100 if expected > 0 else 0

    return count >= expected * 0.95, progress  # 95% threshold

def run_notebook(notebook_name, description, timeout_minutes=240):
    """Execute a notebook using venv Python"""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"Notebook: {notebook_name}")
    print(f"{'='*70}\n")

    notebook_path = NOTEBOOKS_DIR / notebook_name

    cmd = [
        str(VENV_PYTHON), "-m", "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        str(notebook_path),
        f"--ExecutePreprocessor.timeout={timeout_minutes*60}"
    ]

    try:
        start = time.time()
        result = subprocess.run(
            cmd,
            cwd=str(Path.cwd()),
            capture_output=False,
            text=True,
            timeout=timeout_minutes*60 + 300  # 5min buffer
        )

        elapsed = (time.time() - start) / 60

        if result.returncode == 0:
            print(f"\n[SUCCESS] Completed in {elapsed:.1f} minutes\n")
            return True
        else:
            print(f"\n[FAILED] Exit code {result.returncode}\n")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] Exceeded {timeout_minutes} minutes\n")
        return False
    except Exception as e:
        print(f"\n[ERROR] {e}\n")
        return False

def main():
    print("="*70)
    print("AUTONOMOUS FYP WORKFLOW - RTX 6000 Ada (Shared Workstation)")
    print("="*70)
    print("\nSafe GPU Usage: 16-20GB target")
    print("Virtual Environment: Isolated from other users")
    print("\n" + "="*70)

    # Step 1: Wait for preprocessing to complete
    print("\nSTEP 1: Waiting for CLAHE preprocessing to complete...")
    print("Processing 21,165 images (this may take 30-60 minutes)")

    wait_start = time.time()
    while True:
        done, progress = check_preprocessing_done()

        if done:
            wait_time = (time.time() - wait_start) / 60
            print(f"\n[SUCCESS] Preprocessing complete ({progress:.1f}%)")
            print(f"Wait time: {wait_time:.1f} minutes\n")
            break

        print(f"[PROGRESS] {progress:.1f}% complete... checking again in 60s")
        time.sleep(60)

        # Safety timeout (4 hours)
        if (time.time() - wait_start) > 14400:
            print("\n[TIMEOUT] Preprocessing taking too long (>4 hours)")
            print("Please check 02_data_cleaning.ipynb manually")
            return

    # Step 2: Train baseline (Phase 1 completion)
    print("\n" + "="*70)
    print("STEP 2: Training Baseline ResNet-50 (Phase 1 Completion)")
    print("="*70)

    success = run_notebook(
        "04_baseline_test.ipynb",
        "Baseline ResNet-50 Training",
        timeout_minutes=120  # 2 hours max
    )

    if not success:
        print("[WARNING] Baseline training failed - check notebook for errors")
        print("Continuing with Phase 2 anyway...")

    # Step 3: Ask user if they want to continue with Phase 2
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE!")
    print("="*70)
    print("\nReady to start Phase 2: Train all 6 models × 5 seeds (30 runs)")
    print("Estimated time: 20-24 hours")
    print("\nThis will run autonomously in the background.")
    print("=" *70)

    # For autonomous mode, just continue
    print("\n[AUTO] Starting Phase 2 in 10 seconds...")
    print("(Press Ctrl+C to cancel)")
    try:
        time.sleep(10)
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] User interrupted")
        return

    # Step 4: Train all Phase 2 models
    print("\n" + "="*70)
    print("STEP 3: Phase 2 - Systematic Experimentation")
    print("="*70)

    phase2_notebooks = [
        ("06_crossvit_training.ipynb", "CrossViT-Tiny (Main Model)", 240),
        ("07_resnet50_training.ipynb", "ResNet-50 Baseline (5 seeds)", 240),
        ("08_densenet121_training.ipynb", "DenseNet-121 Baseline", 240),
        ("09_efficientnet_training.ipynb", "EfficientNet-B0 Baseline", 240),
        ("10_vit_training.ipynb", "ViT-Base Baseline", 240),
        ("11_swin_training.ipynb", "Swin-Tiny Baseline", 240),
    ]

    results = []
    phase2_start = time.time()

    for i, (notebook, description, timeout) in enumerate(phase2_notebooks, 1):
        print(f"\n{'#'*70}")
        print(f"# MODEL {i}/6: {description}")
        print(f"{'#'*70}")

        success = run_notebook(notebook, description, timeout_minutes=timeout)
        results.append({
            'model': description,
            'notebook': notebook,
            'success': success
        })

        if not success:
            print(f"[WARNING] {description} failed!")
            print("  Continuing with next model...")

        # Brief pause between models
        print("\n[INFO] Cooling down for 30 seconds...")
        time.sleep(30)

    # Final summary
    phase2_time = (time.time() - phase2_start) / 3600
    total_time = (time.time() - wait_start) / 3600

    print("\n\n" + "="*70)
    print("ALL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nPhase 2 time: {phase2_time:.2f} hours")
    print(f"Total time: {total_time:.2f} hours")
    print(f"\nResults:")

    successful = sum(1 for r in results if r['success'])
    for result in results:
        status = "[OK]" if result['success'] else "[FAILED]"
        print(f"  {status} {result['model']}")

    print(f"\n{successful}/{len(results)} models trained successfully")

    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. View all results:")
    print("   mlflow ui")
    print("   Open: http://localhost:5000")
    print("\n2. Run statistical validation:")
    print("   python -m jupyter nbconvert --execute 12_statistical_validation.ipynb")
    print("\n3. Generate thesis tables:")
    print("   python -m jupyter nbconvert --execute 15_thesis_content.ipynb")
    print("\n" + "="*70)
    print("FYP Training Complete! Ready for thesis writing.")
    print("="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Workflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
