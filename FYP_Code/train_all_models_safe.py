"""
Safe Multi-Model Training Script for Shared Workstation
- Automatic GPU resource management
- Uses raw images (applies CLAHE on-the-fly)
- Proper MLflow tracking
- Fair resource allocation

Usage: python train_all_models_safe.py [model_name]
  model_name: resnet50, densenet121, efficientnet, vit, swin, crossvit, or "all"
"""

import os, sys, random, time, warnings, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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

from sklearn.metrics import confusion_matrix
from shared_gpu_config import get_safe_config, print_gpu_status

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# Paths
CSV_DIR = Path("data/processed")
MODELS_DIR = Path("experiments/phase2_systematic/models")
RESULTS_DIR = Path("experiments/phase2_systematic/results/confusion_matrices")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Base configuration
BASE_CONFIG = {
    'device': device,
    'num_classes': 4,
    'image_size': 240,
    'class_names': ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia'],
    'class_weights': [1.47, 0.52, 0.88, 3.95],
    'batch_size': 380,  # Maximized for RTX 6000 Ada (48GB VRAM, ~85-90% usage target)
    'num_workers': 8,  # Maximum data loading performance
    'learning_rate': 1.78e-4,  # Scaled for larger batch size (1.2e-4 * 380/256)
    'weight_decay': 1e-4,
    'max_epochs': 30,
    'early_stopping_patience': 10,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'mixed_precision': True,
    'seeds': [42, 123, 456, 789, 101112],
}

# Dataset class - uses RAW images with on-the-fly CLAHE
class COVID19DatasetRaw(Dataset):
    """Load raw images and apply CLAHE on-the-fly"""
    def __init__(self, dataframe, transform=None, apply_clahe=True):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.apply_clahe = apply_clahe
        self.image_paths = dataframe['image_path'].values  # Use raw image paths
        self.labels = dataframe['label'].values

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load grayscale image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Apply CLAHE if enabled
        if self.apply_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = clahe.apply(image)

        # Resize to 240x240
        image = cv2.resize(image, (240, 240))

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Convert to PIL for transforms
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# Transforms
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=BASE_CONFIG['mean'], std=BASE_CONFIG['std'])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=BASE_CONFIG['mean'], std=BASE_CONFIG['std'])
])

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
    correct, total = 0, 0

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
    correct, total = 0, 0
    all_preds, all_labels = [], []

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

def get_model(model_name, num_classes):
    """Load model architecture"""
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'vit':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=240, num_classes=num_classes)
    elif model_name == 'swin':
        import timm
        model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, img_size=240, num_classes=num_classes)
    elif model_name == 'crossvit':
        import timm
        model = timm.create_model('crossvit_tiny_240', pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model

def train_model_single_seed(model_name, seed, config, train_df, val_df, test_df):
    """Train a single model with one seed"""
    print(f"\n{'='*70}\nTRAINING {model_name.upper()} WITH SEED {seed}\n{'='*70}")

    set_seed(seed)

    # Create datasets
    train_dataset = COVID19DatasetRaw(train_df, transform=train_transform)
    val_dataset = COVID19DatasetRaw(val_df, transform=val_transform)
    test_dataset = COVID19DatasetRaw(test_df, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                             num_workers=config['num_workers'], pin_memory=True)

    # Load model
    model = get_model(model_name, config['num_classes'])
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"{model_name.upper()} loaded: {num_params:,} parameters")

    # Loss, optimizer
    class_weights = torch.tensor(config['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] else None

    # MLflow
    if MLFLOW_AVAILABLE:
        try:
            if mlflow.active_run():
                mlflow.end_run()  # End any active run first

            mlflow.start_run(run_name=f"{model_name}-seed-{seed}")
            mlflow.log_param("model", model_name)
            mlflow.log_param("random_seed", seed)
            mlflow.log_param("batch_size", config['batch_size'])
            mlflow.log_param("learning_rate", config['learning_rate'])
            mlflow.log_param("num_users", config.get('num_users_detected', 1))
            mlflow.set_tag("phase", "Phase 2")
        except Exception as e:
            print(f"MLflow warning: {e}")

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = MODELS_DIR / model_name / f"{model_name}_best_seed{seed}.pth"

    start_time = time.time()

    try:
        for epoch in range(config['max_epochs']):
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
            val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
            scheduler.step(val_loss)

            if MLFLOW_AVAILABLE and mlflow.active_run():
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
        plt.title(f"{model_name.upper()} Confusion Matrix (Seed {seed})")
        cm_path = RESULTS_DIR / f"{model_name}_cm_seed{seed}.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("training_time_minutes", training_time / 60)
            mlflow.log_artifact(str(cm_path))

        print(f"Seed {seed} complete: Test Acc = {test_acc:.2f}%")

        return {'seed': seed, 'test_acc': test_acc, 'test_loss': test_loss, 'training_time': training_time}

    finally:
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run()

def train_model_all_seeds(model_name):
    """Train one model with all 5 seeds"""
    print(f"\n{'='*70}\nSTARTING {model_name.upper()} TRAINING (5 SEEDS)\n{'='*70}")

    # Check GPU and get safe config
    print_gpu_status()
    config = get_safe_config(model_name, BASE_CONFIG)
    print(f"\nUsing configuration: batch_size={config['batch_size']}, num_workers={config['num_workers']}")

    # Load data
    train_df = pd.read_csv(CSV_DIR / "train.csv")
    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("crossvit-covid19-classification")
        mlflow.set_tracking_uri("file:./experiments/phase2_systematic/mlruns")

    all_results = []
    for seed in config['seeds']:
        try:
            result = train_model_single_seed(model_name, seed, config, train_df, val_df, test_df)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR with seed {seed}: {e}")
            import traceback
            traceback.print_exc()

    # Statistics
    if all_results:
        accuracies = [r['test_acc'] for r in all_results]
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies, ddof=1)

        print(f"\n{model_name.upper()} Results ({len(all_results)} seeds):")
        print(f"   Mean +/- Std: {mean_acc:.2f}% +/- {std_acc:.2f}%")
        print(f"   Range: [{np.min(accuracies):.2f}%, {np.max(accuracies):.2f}%]")

        results_df = pd.DataFrame(all_results)
        results_path = Path("experiments/phase2_systematic/results/metrics") / f"{model_name}_results.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nResults saved to {results_path}")

    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train COVID-19 classification models')
    parser.add_argument('model', type=str, default='resnet50',
                        help='Model to train: resnet50, densenet121, efficientnet, vit, swin, crossvit, or "all"')
    args = parser.parse_args()

    if args.model == 'all':
        models = ['resnet50', 'densenet121', 'efficientnet', 'vit', 'swin', 'crossvit']
        for model in models:
            train_model_all_seeds(model)
    else:
        train_model_all_seeds(args.model)
