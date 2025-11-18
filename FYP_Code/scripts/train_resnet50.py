"""
ResNet-50 Training Script for Phase 2
Train with 5 random seeds for statistical validation
"""

import os, sys, random, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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
    print("WARNING: MLflow not available")

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Configuration
CSV_DIR = Path("data/processed")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'device': device,
    'model_name': 'ResNet-50',
    'num_classes': 4,
    'image_size': 240,
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],
    'batch_size': 64,
    'num_workers': 4,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],
}

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")

# MLflow setup
if MLFLOW_AVAILABLE:
    mlflow.set_experiment("crossvit-covid19-classification")
    mlflow.set_tracking_uri("file:./mlruns")
    print("MLflow configured")

# Load data
train_df = pd.read_csv(CSV_DIR / "train_processed.csv")
val_df = pd.read_csv(CSV_DIR / "val_processed.csv")
test_df = pd.read_csv(CSV_DIR / "test_processed.csv")
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

# Dataset class
class COVID19Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_paths = dataframe['processed_path'].values
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
])

val_transform = transforms.Compose([
    transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIG['mean'], std=CONFIG['std'])
])

# Training functions
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

# Single seed training
def train_resnet_single_seed(seed, config):
    print(f"\n{'='*70}\nTRAINING RESNET-50 WITH SEED {seed}\n{'='*70}")

    set_seed(seed)

    # Create dataloaders
    train_dataset = COVID19Dataset(train_df, transform=train_transform)
    val_dataset = COVID19Dataset(val_df, transform=val_transform)
    test_dataset = COVID19Dataset(test_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # Load model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, config['num_classes'])
    model = model.to(device)

    print(f"ResNet-50 loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Loss, optimizer
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.start_run(run_name=f"resnet50-seed-{seed}")
        mlflow.log_param("model", config['model_name'])
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("batch_size", config['batch_size'])
        mlflow.log_param("learning_rate", config['learning_rate'])
        mlflow.set_tag("phase", "Phase 2 - Baseline")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = MODELS_DIR / f"resnet50_best_seed{seed}.pth"

    start_time = time.time()

    for epoch in range(config['max_epochs']):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if MLFLOW_AVAILABLE:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_acc", val_acc, step=epoch)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("Best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time

    # Test
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device, desc="Test")

    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config['class_names'], yticklabels=config['class_names'])
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.title(f"ResNet-50 Confusion Matrix (Seed {seed})")
    cm_path = RESULTS_DIR / f"resnet50_cm_seed{seed}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    if MLFLOW_AVAILABLE:
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("training_time_minutes", training_time / 60)
        mlflow.log_artifact(str(cm_path))
        mlflow.end_run()

    print(f"Seed {seed} complete: Test Acc = {test_acc:.2f}%")

    return {'seed': seed, 'test_acc': test_acc, 'test_loss': test_loss, 'training_time': training_time}

# Main training loop
print(f"\n{'='*70}\nSTARTING MULTI-SEED RESNET-50 TRAINING\n{'='*70}")
print(f"Seeds: {CONFIG['seeds']}\n")

all_results = []
for seed in CONFIG['seeds']:
    try:
        result = train_resnet_single_seed(seed, CONFIG)
        all_results.append(result)
    except Exception as e:
        print(f"ERROR with seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        continue

print(f"\n{'='*70}\nALL SEEDS COMPLETED\n{'='*70}")

# Statistical analysis
if all_results:
    accuracies = [r['test_acc'] for r in all_results]
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies, ddof=1)

    print(f"\nResNet-50 Results ({len(all_results)} seeds):")
    print(f"   Mean ± Std: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"   Range: [{np.min(accuracies):.2f}%, {np.max(accuracies):.2f}%]")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "resnet50_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    print(results_df)
else:
    print("\nNo results - all seeds failed!")
