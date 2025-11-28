#!/usr/bin/env python
"""
Train VGG-16 and MobileNetV2 weak baselines with 5 random seeds.

Configuration matched with other training notebooks:
- Optimizer: Adam
- Scheduler: ReduceLROnPlateau
- Learning rate: 1e-4
- Weight decay: 1e-4
- Max epochs: 30
- Early stopping: 10

Author: Tan Ming Kai (24PMR12003)
Date: 2025-11-26
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
from torchvision import transforms, models

import cv2
from PIL import Image

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARNING] MLflow not available")

from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration - MATCHED WITH OTHER TRAINING NOTEBOOKS
# =============================================================================

# Get the project root directory
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

CSV_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "experiments" / "phase2_models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "phase2_results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'device': device,
    'num_classes': 4,
    'image_size': 224,
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],

    # OPTIMIZED FOR 24GB+ VRAM (RTX 6000 Ada)
    'batch_size': 64,
    'num_workers': 0,  # Windows has issues with multiprocessing and cv2.CLAHE

    # Training hyperparameters - MATCHED WITH OTHER MODELS
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,

    # Normalization (ImageNet)
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],

    # Mixed precision for speed
    'mixed_precision': True,

    # Seeds - same as all other models
    'seeds': [42, 123, 456, 789, 101112],
}

# =============================================================================
# Dataset class with on-the-fly CLAHE preprocessing
# =============================================================================

class COVID19Dataset(Dataset):
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
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        enhanced = self.clahe.apply(gray)
        rgb_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        image = Image.fromarray(rgb_image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# =============================================================================
# Transforms
# =============================================================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
])

# =============================================================================
# Training functions
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, epoch=0):
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
            with torch.cuda.amp.autocast():
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

        progress_bar.set_postfix({'loss': running_loss / (batch_idx + 1), 'acc': 100. * correct / total})

    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device, desc="Val"):
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

# =============================================================================
# Single seed training function
# =============================================================================

def train_model_single_seed(model_name, model_fn, seed, config, train_df, val_df, test_df):
    print(f"\n{'='*70}\nTRAINING {model_name.upper()} WITH SEED {seed}\n{'='*70}")

    set_seed(seed)

    # Create dataloaders
    train_dataset = COVID19Dataset(train_df, transform=train_transform)
    val_dataset = COVID19Dataset(val_df, transform=val_transform)
    test_dataset = COVID19Dataset(test_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['num_workers'] > 0 else False
    )

    # Load model
    model = model_fn()
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] {model_name} loaded: {total_params:,} parameters")

    # Loss, optimizer, scheduler - MATCHED WITH OTHER MODELS
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

    # MLflow - end any active run first
    if MLFLOW_AVAILABLE:
        try:
            mlflow.end_run()  # End any active run
        except:
            pass
        mlflow.start_run(run_name=f"{model_name.lower().replace('-', '')}-seed-{seed}")
        mlflow.log_param("model", model_name)
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("batch_size", config['batch_size'])
        mlflow.log_param("learning_rate", config['learning_rate'])
        mlflow.log_param("weight_decay", config['weight_decay'])
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("scheduler", "ReduceLROnPlateau")
        mlflow.set_tag("phase", "Phase 2 - Weak Baselines")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = MODELS_DIR / f"{model_name.lower().replace('-', '')}_seed{seed}.pth"

    start_time = time.time()

    for epoch in range(config['max_epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if MLFLOW_AVAILABLE:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_acc", train_acc, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} Acc={train_acc:.2f}% | Val Loss={val_loss:.4f} Acc={val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[OK] Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"[STOP] Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time

    # Test evaluation
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device, desc="Test")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config['class_names'], yticklabels=config['class_names'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f"{model_name} Confusion Matrix (Seed {seed})")
    cm_path = RESULTS_DIR / f"{model_name.lower().replace('-', '')}_cm_seed{seed}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    if MLFLOW_AVAILABLE:
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("training_time_minutes", training_time / 60)
        mlflow.log_artifact(str(cm_path))
        mlflow.end_run()

    print(f"[OK] Seed {seed} complete: Test Acc = {test_acc:.2f}% | Time = {training_time/60:.1f} min")

    del model
    torch.cuda.empty_cache()

    return {
        'model': model_name,
        'seed': seed,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'training_time_min': training_time / 60,
        'best_val_loss': best_val_loss
    }

# =============================================================================
# Model definitions
# =============================================================================

def create_vgg16():
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Linear(4096, 4)
    return model

def create_mobilenetv2():
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    return model

# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("WEAK BASELINES TRAINING: VGG-16 & MobileNetV2")
    print("="*70)

    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print(f"  Learning rate: {CONFIG['learning_rate']}")
    print(f"  Max epochs: {CONFIG['max_epochs']}")
    print(f"  Seeds: {CONFIG['seeds']}")

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("crossvit-covid19-classification")
        print("\n[OK] MLflow configured")

    # Load data
    print("\n[INFO] Loading data...")
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Verify paths
    sample_path = train_df['image_path'].iloc[0]
    if not Path(sample_path).exists():
        print(f"[ERROR] Sample path does not exist: {sample_path}")
        sys.exit(1)
    print(f"[OK] Path verification passed")

    # ==========================================================================
    # Train VGG-16
    # ==========================================================================
    print(f"\n{'='*70}\nSTARTING VGG-16 TRAINING\n{'='*70}")

    vgg16_results = []
    for seed in CONFIG['seeds']:
        try:
            result = train_model_single_seed('VGG-16', create_vgg16, seed, CONFIG, train_df, val_df, test_df)
            vgg16_results.append(result)
        except Exception as e:
            print(f"[ERROR] VGG-16 seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    # VGG-16 Statistics
    if vgg16_results:
        vgg16_accuracies = [r['test_acc'] for r in vgg16_results]
        print(f"\n[STATS] VGG-16 Results ({len(vgg16_results)} seeds):")
        print(f"   Mean ± Std: {np.mean(vgg16_accuracies):.2f}% ± {np.std(vgg16_accuracies, ddof=1):.2f}%")

        vgg16_df = pd.DataFrame(vgg16_results)
        vgg16_df.to_csv(RESULTS_DIR / "vgg16_results.csv", index=False)
        print(f"[OK] Results saved to {RESULTS_DIR / 'vgg16_results.csv'}")

    # ==========================================================================
    # Train MobileNetV2
    # ==========================================================================
    print(f"\n{'='*70}\nSTARTING MOBILENETV2 TRAINING\n{'='*70}")

    mobilenetv2_results = []
    for seed in CONFIG['seeds']:
        try:
            result = train_model_single_seed('MobileNetV2', create_mobilenetv2, seed, CONFIG, train_df, val_df, test_df)
            mobilenetv2_results.append(result)
        except Exception as e:
            print(f"[ERROR] MobileNetV2 seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    # MobileNetV2 Statistics
    if mobilenetv2_results:
        mobilenetv2_accuracies = [r['test_acc'] for r in mobilenetv2_results]
        print(f"\n[STATS] MobileNetV2 Results ({len(mobilenetv2_results)} seeds):")
        print(f"   Mean ± Std: {np.mean(mobilenetv2_accuracies):.2f}% ± {np.std(mobilenetv2_accuracies, ddof=1):.2f}%")

        mobilenetv2_df = pd.DataFrame(mobilenetv2_results)
        mobilenetv2_df.to_csv(RESULTS_DIR / "mobilenetv2_results.csv", index=False)
        print(f"[OK] Results saved to {RESULTS_DIR / 'mobilenetv2_results.csv'}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    if vgg16_results:
        print(f"\nVGG-16: {np.mean(vgg16_accuracies):.2f}% ± {np.std(vgg16_accuracies, ddof=1):.2f}%")
    if mobilenetv2_results:
        print(f"MobileNetV2: {np.mean(mobilenetv2_accuracies):.2f}% ± {np.std(mobilenetv2_accuracies, ddof=1):.2f}%")

    print(f"\n[OK] Training complete!")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == '__main__':
    main()
