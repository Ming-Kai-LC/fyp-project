"""
CrossViT Enhanced Training Script
==================================
Train CrossViT-Base and CrossViT-Small with 5 random seeds.
Includes Ensemble and Test-Time Augmentation (TTA) evaluation.

Author: Tan Ming Kai (24PMR12003)
Date: 2025-11-26
Hardware: NVIDIA RTX 6000 Ada (51GB VRAM)
"""

import os
import sys
import random
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

import cv2
from PIL import Image

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# Configuration
# ============================================================================

# Directories
CSV_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Common configuration
COMMON_CONFIG = {
    'num_classes': 4,
    'image_size': 240,
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],
    'num_workers': 0,  # Windows compatibility with CLAHE
    'learning_rate': 5e-5,
    'weight_decay': 0.05,
    'max_epochs': 50,
    'early_stopping_patience': 15,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],
}

# Model-specific configurations (optimized for 24GB VRAM)
CONFIG_BASE = {
    **COMMON_CONFIG,
    'model_name': 'CrossViT-Base',
    'timm_model': 'crossvit_base_240',
    'batch_size': 64,  # Increased for faster training (uses ~20GB VRAM)
}

CONFIG_SMALL = {
    **COMMON_CONFIG,
    'model_name': 'CrossViT-Small',
    'timm_model': 'crossvit_small_240',
    'batch_size': 128,  # Increased for faster training (uses ~18GB VRAM)
}

# ============================================================================
# Dataset with On-the-fly CLAHE
# ============================================================================

class COVID19Dataset(Dataset):
    """Dataset with on-the-fly CLAHE enhancement."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_paths = dataframe['image_path'].values
        self.labels = dataframe['label'].values
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load: {img_path}")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply CLAHE
        enhanced = self.clahe.apply(gray)

        # Convert to RGB
        rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        # Convert to PIL Image
        image = Image.fromarray(rgb_image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# ============================================================================
# Transforms
# ============================================================================

def get_transforms(image_size, mean, std):
    """Get train and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform

# ============================================================================
# Training Functions
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]")

    for batch_idx, (images, labels) in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, desc="Val"):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"[{desc}]"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(loader), 100. * correct / total, np.array(all_preds), np.array(all_labels)

# ============================================================================
# Single Seed Training
# ============================================================================

def train_single_seed(seed, config, train_df, val_df, test_df, device):
    """Train model with a single seed."""
    print(f"\n{'='*70}")
    print(f"TRAINING {config['model_name']} WITH SEED {seed}")
    print(f"{'='*70}")

    set_seed(seed)

    # Get transforms
    train_transform, val_transform = get_transforms(
        config['image_size'], config['mean'], config['std']
    )

    # Create datasets
    train_dataset = COVID19Dataset(train_df, transform=train_transform)
    val_dataset = COVID19Dataset(val_df, transform=val_transform)
    test_dataset = COVID19Dataset(test_df, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'],
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'],
        pin_memory=True
    )

    # Load model
    model = timm.create_model(
        config['timm_model'],
        pretrained=True,
        num_classes=config['num_classes']
    )
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"[OK] {config['model_name']} loaded: {param_count:,} parameters")

    # Loss, optimizer, scheduler
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.amp.GradScaler('cuda') if config['mixed_precision'] else None

    # MLflow logging
    if MLFLOW_AVAILABLE:
        mlflow.end_run()
        model_tag = config['model_name'].lower().replace('-', '_').replace(' ', '_')
        mlflow.start_run(run_name=f"{model_tag}-seed-{seed}")
        mlflow.log_param("model", config['model_name'])
        mlflow.log_param("timm_model", config['timm_model'])
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("batch_size", config['batch_size'])
        mlflow.log_param("learning_rate", config['learning_rate'])
        mlflow.log_param("weight_decay", config['weight_decay'])
        mlflow.log_param("max_epochs", config['max_epochs'])
        mlflow.log_param("optimizer", "AdamW")
        mlflow.log_param("scheduler", "CosineAnnealingWarmRestarts")
        mlflow.set_tag("phase", "Phase 2 - CrossViT Enhancement")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    model_filename = config['model_name'].lower().replace('-', '').replace(' ', '_')
    best_model_path = MODELS_DIR / f"{model_filename}_best_seed{seed}.pth"

    start_time = time.time()

    for epoch in range(config['max_epochs']):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, epoch
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step()

        if MLFLOW_AVAILABLE:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | "
              f"Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("[OK] Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"[STOP] Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time

    # Test evaluation
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, desc="Test"
    )

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config['class_names'],
                yticklabels=config['class_names'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f"{config['model_name']} Confusion Matrix (Seed {seed})")
    cm_path = RESULTS_DIR / f"{model_filename}_cm_seed{seed}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Classification report
    report = classification_report(test_labels, test_preds,
                                   target_names=config['class_names'],
                                   output_dict=True)

    if MLFLOW_AVAILABLE:
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("training_time_minutes", training_time / 60)
        mlflow.log_artifact(str(cm_path))

        # Log per-class metrics
        for cls in config['class_names']:
            mlflow.log_metric(f"precision_{cls}", report[cls]['precision'])
            mlflow.log_metric(f"recall_{cls}", report[cls]['recall'])
            mlflow.log_metric(f"f1_{cls}", report[cls]['f1-score'])

        mlflow.end_run()

    print(f"\n[OK] Seed {seed} complete: Test Acc = {test_acc:.2f}% | Time = {training_time/60:.1f} min")

    return {
        'seed': seed,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time': training_time,
        'model_path': str(best_model_path)
    }

# ============================================================================
# Ensemble Prediction
# ============================================================================

def load_trained_models(model_name, timm_model, num_classes, seeds, models_dir, device):
    """Load all trained models for ensemble."""
    models_list = []
    model_filename = model_name.lower().replace('-', '').replace(' ', '_')

    for seed in seeds:
        model_path = models_dir / f"{model_filename}_best_seed{seed}.pth"
        if model_path.exists():
            model = timm.create_model(timm_model, pretrained=False, num_classes=num_classes)
            model.load_state_dict(torch.load(model_path))
            model = model.to(device)
            model.eval()
            models_list.append(model)
            print(f"  Loaded: {model_path.name}")
        else:
            print(f"  [WARNING] Not found: {model_path}")

    return models_list

def ensemble_predict(models_list, loader, device, config):
    """Ensemble prediction by averaging probabilities."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Ensemble Inference"):
            images = images.to(device)

            # Get predictions from all models
            all_probs = []
            for model in models_list:
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs)

            # Average probabilities
            avg_probs = torch.stack(all_probs).mean(dim=0)
            preds = avg_probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

# ============================================================================
# Test-Time Augmentation (TTA)
# ============================================================================

def get_tta_transforms(image_size, mean, std):
    """Get TTA transforms."""
    base_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    tta_transforms = [
        # Original
        base_transform,
        # Horizontal flip
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        # Rotation +5
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation((5, 5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        # Rotation -5
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomRotation((-5, -5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
        # Brightness adjustment
        transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(brightness=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]),
    ]

    return tta_transforms

def tta_predict(model, test_df, device, config):
    """Test-Time Augmentation prediction."""
    tta_transforms = get_tta_transforms(
        config['image_size'], config['mean'], config['std']
    )

    all_preds = []
    all_labels = test_df['label'].values

    model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_df)), desc="TTA Inference"):
            img_path = test_df.iloc[idx]['image_path']

            # Load and preprocess image
            image = cv2.imread(img_path)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Get predictions from all TTA transforms
            all_probs = []
            for transform in tta_transforms:
                img_tensor = transform(pil_image).unsqueeze(0).to(device)
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                all_probs.append(probs)

            # Average probabilities
            avg_probs = torch.stack(all_probs).mean(dim=0)
            pred = avg_probs.argmax(dim=1).item()
            all_preds.append(pred)

    return np.array(all_preds), all_labels

def ensemble_tta_predict(models_list, test_df, device, config):
    """Combined Ensemble + TTA prediction."""
    tta_transforms = get_tta_transforms(
        config['image_size'], config['mean'], config['std']
    )

    all_preds = []
    all_labels = test_df['label'].values

    for model in models_list:
        model.eval()

    with torch.no_grad():
        for idx in tqdm(range(len(test_df)), desc="Ensemble+TTA Inference"):
            img_path = test_df.iloc[idx]['image_path']

            # Load and preprocess image
            image = cv2.imread(img_path)
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(rgb_image)

            # Get predictions from all models and all TTA transforms
            all_probs = []
            for model in models_list:
                for transform in tta_transforms:
                    img_tensor = transform(pil_image).unsqueeze(0).to(device)
                    output = model(img_tensor)
                    probs = torch.softmax(output, dim=1)
                    all_probs.append(probs)

            # Average all probabilities (5 models × 5 TTA = 25 predictions)
            avg_probs = torch.stack(all_probs).mean(dim=0)
            pred = avg_probs.argmax(dim=1).item()
            all_preds.append(pred)

    return np.array(all_preds), all_labels

# ============================================================================
# Main Training Function
# ============================================================================

def train_crossvit_variant(config, train_df, val_df, test_df, device):
    """Train a CrossViT variant with all seeds."""
    print(f"\n{'='*70}")
    print(f"STARTING {config['model_name']} TRAINING")
    print(f"{'='*70}")
    print(f"Seeds: {config['seeds']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Max epochs: {config['max_epochs']}\n")

    all_results = []

    for seed in config['seeds']:
        try:
            result = train_single_seed(seed, config, train_df, val_df, test_df, device)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Statistical analysis
    if all_results:
        accuracies = [r['test_acc'] for r in all_results]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)

        print(f"\n{'='*70}")
        print(f"{config['model_name']} RESULTS ({len(all_results)} seeds)")
        print(f"{'='*70}")
        print(f"Mean ± Std: {mean_acc:.2f}% ± {std_acc:.2f}%")
        print(f"Range: [{np.min(accuracies):.2f}%, {np.max(accuracies):.2f}%]")

        # Save results
        results_df = pd.DataFrame(all_results)
        model_filename = config['model_name'].lower().replace('-', '').replace(' ', '_')
        results_path = RESULTS_DIR / f"{model_filename}_results.csv"
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

    return all_results

# ============================================================================
# Main
# ============================================================================

def main():
    print("="*70)
    print("CROSSVIT ENHANCED TRAINING")
    print("="*70)

    # Hardware verification
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("crossvit-covid19-classification")
        mlflow.set_tracking_uri("file:./mlruns")
        print("\n[OK] MLflow configured")

    # Load data
    print("\n[INFO] Loading data...")
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Verify paths
    sample_path = train_df.iloc[0]['image_path']
    if not Path(sample_path).exists():
        print(f"[ERROR] Sample path not found: {sample_path}")
        return
    print("[OK] Path verification passed")

    # ========================================================================
    # PHASE 1: Train CrossViT-Base (5 seeds)
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: CROSSVIT-BASE TRAINING")
    print("="*70)

    base_results = train_crossvit_variant(CONFIG_BASE, train_df, val_df, test_df, device)

    # ========================================================================
    # PHASE 2: Train CrossViT-Small (5 seeds)
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: CROSSVIT-SMALL TRAINING")
    print("="*70)

    small_results = train_crossvit_variant(CONFIG_SMALL, train_df, val_df, test_df, device)

    # ========================================================================
    # PHASE 3: Ensemble Evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 3: ENSEMBLE EVALUATION")
    print("="*70)

    _, val_transform = get_transforms(
        COMMON_CONFIG['image_size'], COMMON_CONFIG['mean'], COMMON_CONFIG['std']
    )
    test_dataset = COVID19Dataset(test_df, transform=val_transform)
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # CrossViT-Base Ensemble
    print("\nLoading CrossViT-Base models for ensemble...")
    base_models = load_trained_models(
        CONFIG_BASE['model_name'], CONFIG_BASE['timm_model'],
        CONFIG_BASE['num_classes'], CONFIG_BASE['seeds'], MODELS_DIR, device
    )

    if len(base_models) > 0:
        base_preds, base_labels = ensemble_predict(base_models, test_loader, device, CONFIG_BASE)
        base_ensemble_acc = accuracy_score(base_labels, base_preds) * 100
        print(f"\nCrossViT-Base Ensemble Accuracy: {base_ensemble_acc:.2f}%")

        # Save ensemble results
        cm = confusion_matrix(base_labels, base_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CONFIG_BASE['class_names'],
                    yticklabels=CONFIG_BASE['class_names'])
        plt.title("CrossViT-Base Ensemble Confusion Matrix")
        plt.savefig(RESULTS_DIR / "crossvit_base_ensemble_cm.png", dpi=300, bbox_inches='tight')
        plt.close()

    # CrossViT-Small Ensemble
    print("\nLoading CrossViT-Small models for ensemble...")
    small_models = load_trained_models(
        CONFIG_SMALL['model_name'], CONFIG_SMALL['timm_model'],
        CONFIG_SMALL['num_classes'], CONFIG_SMALL['seeds'], MODELS_DIR, device
    )

    if len(small_models) > 0:
        small_preds, small_labels = ensemble_predict(small_models, test_loader, device, CONFIG_SMALL)
        small_ensemble_acc = accuracy_score(small_labels, small_preds) * 100
        print(f"\nCrossViT-Small Ensemble Accuracy: {small_ensemble_acc:.2f}%")

        # Save ensemble results
        cm = confusion_matrix(small_labels, small_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CONFIG_SMALL['class_names'],
                    yticklabels=CONFIG_SMALL['class_names'])
        plt.title("CrossViT-Small Ensemble Confusion Matrix")
        plt.savefig(RESULTS_DIR / "crossvit_small_ensemble_cm.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ========================================================================
    # PHASE 4: TTA Evaluation (Best single model)
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 4: TEST-TIME AUGMENTATION (TTA) EVALUATION")
    print("="*70)

    # Use best CrossViT-Base model (seed 42)
    if len(base_models) > 0:
        print("\nTTA with CrossViT-Base (seed 42)...")
        tta_preds, tta_labels = tta_predict(base_models[0], test_df, device, CONFIG_BASE)
        tta_acc = accuracy_score(tta_labels, tta_preds) * 100
        print(f"CrossViT-Base TTA Accuracy: {tta_acc:.2f}%")

    # ========================================================================
    # PHASE 5: Ensemble + TTA (Best combination)
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 5: ENSEMBLE + TTA EVALUATION")
    print("="*70)

    if len(base_models) > 0:
        print("\nEnsemble + TTA with CrossViT-Base (5 models × 5 TTA = 25 predictions)...")
        ensemble_tta_preds, ensemble_tta_labels = ensemble_tta_predict(
            base_models, test_df, device, CONFIG_BASE
        )
        ensemble_tta_acc = accuracy_score(ensemble_tta_labels, ensemble_tta_preds) * 100
        print(f"CrossViT-Base Ensemble + TTA Accuracy: {ensemble_tta_acc:.2f}%")

        # Save final results
        cm = confusion_matrix(ensemble_tta_labels, ensemble_tta_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CONFIG_BASE['class_names'],
                    yticklabels=CONFIG_BASE['class_names'])
        plt.title("CrossViT-Base Ensemble + TTA Confusion Matrix")
        plt.savefig(RESULTS_DIR / "crossvit_base_ensemble_tta_cm.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*70)

    if base_results:
        base_accs = [r['test_acc'] for r in base_results]
        print(f"\nCrossViT-Base:")
        print(f"  Single model: {np.mean(base_accs):.2f}% ± {np.std(base_accs, ddof=1):.2f}%")
        if len(base_models) > 0:
            print(f"  Ensemble (5 models): {base_ensemble_acc:.2f}%")
            print(f"  TTA (5 augmentations): {tta_acc:.2f}%")
            print(f"  Ensemble + TTA: {ensemble_tta_acc:.2f}%")

    if small_results:
        small_accs = [r['test_acc'] for r in small_results]
        print(f"\nCrossViT-Small:")
        print(f"  Single model: {np.mean(small_accs):.2f}% ± {np.std(small_accs, ddof=1):.2f}%")
        if len(small_models) > 0:
            print(f"  Ensemble (5 models): {small_ensemble_acc:.2f}%")

    print("\n" + "="*70)
    print("[OK] ALL TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
