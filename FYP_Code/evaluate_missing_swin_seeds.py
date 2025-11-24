"""
Evaluate missing Swin seeds (789, 101112) to complete results CSV
This recovers metrics from model checkpoints trained on Nov 21
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time

# Import from training script
import sys
sys.path.append('.')
from train_all_models_safe import get_model, COVID19DatasetRaw, validate, set_seed, device, CSV_DIR

def evaluate_checkpoint(model_name, seed, test_df, image_size=256):
    """Evaluate a saved model checkpoint"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name.upper()} - Seed {seed}")
    print(f"{'='*60}")

    set_seed(seed)

    # Load model
    model = get_model(model_name, num_classes=4)
    checkpoint_path = Path(f"experiments/phase2_systematic/models/{model_name}/{model_name}_best_seed{seed}.pth")

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None

    print(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(device)
    model.eval()

    # Create test dataset and loader
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = COVID19DatasetRaw(test_df, transform=val_transform, image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=42, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate
    class_weights = torch.tensor([1.47, 0.52, 0.88, 3.95], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    start_time = time.time()
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device, desc="Test")
    eval_time = time.time() - start_time

    print(f"Results: Acc={test_acc:.2f}%, Loss={test_loss:.6f}, Time={eval_time:.1f}s")

    # Note: We don't have the original training time, so we'll use a placeholder
    # The training time is not critical for statistical analysis
    training_time = 0.0  # Unknown - model was already trained

    return {
        'seed': seed,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time': training_time  # Set to 0 since we don't know original value
    }

def main():
    print("\n" + "="*70)
    print("RECOVERING MISSING SWIN RESULTS (Seeds 789, 101112)")
    print("="*70)

    # Load test data
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    print(f"\nTest set size: {len(test_df):,} images")

    # Read current CSV
    results_csv = Path("experiments/phase2_systematic/results/metrics/swin_results.csv")
    existing_results = pd.read_csv(results_csv)
    print(f"\nCurrent CSV has {len(existing_results)} seeds: {existing_results['seed'].tolist()}")

    # Evaluate missing seeds
    missing_seeds = [789, 101112]
    new_results = []

    for seed in missing_seeds:
        result = evaluate_checkpoint('swin', seed, test_df, image_size=256)
        if result:
            new_results.append(result)

    if new_results:
        # Combine with existing results
        new_df = pd.DataFrame(new_results)
        combined_df = pd.concat([existing_results, new_df], ignore_index=True)

        # Sort by seed for consistency
        combined_df = combined_df.sort_values('seed').reset_index(drop=True)

        print(f"\n{'='*60}")
        print("UPDATED RESULTS:")
        print(f"{'='*60}")
        print(combined_df.to_string(index=False))

        # Calculate statistics
        accuracies = combined_df['test_acc'].values
        print(f"\n{'='*60}")
        print(f"Swin-Tiny Summary ({len(combined_df)} seeds):")
        print(f"   Mean ± Std: {np.mean(accuracies):.2f}% ± {np.std(accuracies, ddof=1):.2f}%")
        print(f"   Range: [{np.min(accuracies):.2f}%, {np.max(accuracies):.2f}%]")
        print(f"{'='*60}")

        # Save updated CSV
        combined_df.to_csv(results_csv, index=False)
        print(f"\n✓ Updated CSV saved to: {results_csv}")
        print(f"✓ CSV now contains all 5 seeds: {combined_df['seed'].tolist()}")
    else:
        print("\nERROR: Could not evaluate any models")

if __name__ == "__main__":
    main()
