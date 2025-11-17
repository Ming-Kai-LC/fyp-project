#!/usr/bin/env python3
"""
Environment Setup Verification
Checks Python environment, packages, and hardware for FYP

Author: Tan Ming Kai (24PMR12003)
FYP: CrossViT for COVID-19 Classification
"""

import sys
import subprocess
from importlib import import_module


def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(title)
    print("="*60)


def check_python_version():
    """Check Python version"""
    print_section("PYTHON VERSION")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    print(f"Python version: {version_str}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python 3.8+ required")
        return False


def check_package(package_name, import_name=None, min_version=None):
    """
    Check if package is installed and optionally check version
    
    Args:
        package_name: Package name for display
        import_name: Name to use for import (if different from package_name)
        min_version: Minimum version required (tuple)
    """
    if import_name is None:
        import_name = package_name.lower().replace('-', '_')
    
    try:
        module = import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        
        print(f"   ✅ {package_name:20s} {version}")
        
        if min_version and version != 'unknown':
            # Parse version
            try:
                current_version = tuple(map(int, version.split('.')[:len(min_version)]))
                if current_version >= min_version:
                    return True
                else:
                    print(f"      ⚠️  Version {version} < required {'.'.join(map(str, min_version))}")
                    return False
            except:
                return True  # Can't parse version, assume OK
        return True
        
    except ImportError:
        print(f"   ❌ {package_name:20s} NOT INSTALLED")
        return False


def check_required_packages():
    """Check all required packages"""
    print_section("REQUIRED PACKAGES")
    
    packages = {
        'Core Deep Learning': [
            ('PyTorch', 'torch', (2, 0)),
            ('torchvision', 'torchvision', (0, 15)),
            ('timm', 'timm', (0, 9)),
        ],
        'Computer Vision': [
            ('OpenCV', 'cv2', None),
            ('Albumentations', 'albumentations', (1, 3)),
            ('scikit-image', 'skimage', None),
        ],
        'Data Science': [
            ('NumPy', 'numpy', (1, 24)),
            ('pandas', 'pandas', (2, 0)),
            ('matplotlib', 'matplotlib', None),
            ('seaborn', 'seaborn', None),
        ],
        'Scientific Computing': [
            ('SciPy', 'scipy', (1, 11)),
            ('scikit-learn', 'sklearn', (1, 3)),
        ],
        'Utilities': [
            ('tqdm', 'tqdm', None),
            ('Pillow', 'PIL', None),
        ],
    }
    
    all_installed = True
    
    for category, pkg_list in packages.items():
        print(f"\n{category}:")
        for pkg_info in pkg_list:
            if len(pkg_info) == 2:
                pkg_name, import_name = pkg_info
                min_ver = None
            else:
                pkg_name, import_name, min_ver = pkg_info
            
            if not check_package(pkg_name, import_name, min_ver):
                all_installed = False
    
    return all_installed


def check_cuda():
    """Check CUDA availability and version"""
    print_section("CUDA & GPU")
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"cuDNN Version: {torch.backends.cudnn.version()}")
            
            num_gpus = torch.cuda.device_count()
            print(f"Number of GPUs: {num_gpus}")
            
            for i in range(num_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   Compute Capability: {props.major}.{props.minor}")
                print(f"   Total Memory: {props.total_memory / 1e9:.2f} GB")
                print(f"   Multi-processors: {props.multi_processor_count}")
            
            # Test GPU computation
            print("\n🧪 Testing GPU computation...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            print("   ✅ GPU computation successful")
            
            # Check for RTX 4060 specifically
            if 'RTX 4060' in torch.cuda.get_device_name(0):
                print("\n✅ RTX 4060 detected - perfect for FYP!")
                total_gb = props.total_memory / 1e9
                if total_gb >= 7.5:
                    print(f"   ✅ {total_gb:.1f} GB VRAM available (8GB model)")
                else:
                    print(f"   ⚠️  {total_gb:.1f} GB VRAM (expected ~8GB)")
            
            return True
        else:
            print("❌ CUDA not available - will run on CPU (VERY SLOW)")
            print("   Please check:")
            print("   1. NVIDIA GPU is installed")
            print("   2. CUDA drivers are installed")
            print("   3. PyTorch was installed with CUDA support")
            return False
            
    except ImportError:
        print("❌ PyTorch not installed - cannot check CUDA")
        return False


def check_timm_models():
    """Check if CrossViT model is available in timm"""
    print_section("CROSSVIT MODEL AVAILABILITY")
    
    try:
        import timm
        
        # Check if crossvit models are available
        available_models = timm.list_models('crossvit*')
        
        print("Available CrossViT models:")
        for model_name in available_models:
            print(f"   ✅ {model_name}")
        
        if 'crossvit_tiny_240' in available_models:
            print("\n✅ crossvit_tiny_240 available - ready for FYP!")
            
            # Test model loading
            print("\n🧪 Testing model loading...")
            model = timm.create_model('crossvit_tiny_240', pretrained=False, num_classes=4)
            num_params = sum(p.numel() for p in model.parameters())
            print(f"   ✅ Model loaded successfully")
            print(f"   Parameters: {num_params:,} (~{num_params/1e6:.1f}M)")
            
            return True
        else:
            print("\n❌ crossvit_tiny_240 not found")
            print("   Update timm: pip install --upgrade timm")
            return False
            
    except ImportError:
        print("❌ timm not installed")
        return False


def check_disk_space():
    """Check available disk space"""
    print_section("DISK SPACE")
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage("/")
        
        total_gb = total / 1e9
        used_gb = used / 1e9
        free_gb = free / 1e9
        
        print(f"Total Disk Space: {total_gb:.1f} GB")
        print(f"Used: {used_gb:.1f} GB")
        print(f"Free: {free_gb:.1f} GB")
        
        # Dataset is ~15GB, models ~5GB, outputs ~2GB = ~25GB needed
        if free_gb > 30:
            print("\n✅ Sufficient disk space for FYP (~30GB recommended)")
            return True
        elif free_gb > 20:
            print("\n⚠️  Low disk space. Consider cleaning up (30GB+ recommended)")
            return True
        else:
            print("\n❌ Insufficient disk space. Need at least 30GB free")
            return False
    except:
        print("⚠️  Could not check disk space")
        return True


def print_summary(results):
    """Print overall summary"""
    print_section("ENVIRONMENT SETUP SUMMARY")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("🎉 Environment setup COMPLETE!")
        print("✅ Ready to start FYP implementation")
    else:
        print("⚠️  Some issues detected")
        print("Please resolve the issues marked with ❌ before proceeding")
        
        # Suggest installation commands
        if not results['Required Packages']:
            print("\n📦 Install missing packages:")
            print("   pip install -r requirements.txt")
        
        if not results['CUDA & GPU']:
            print("\n🖥️  Install PyTorch with CUDA:")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*60)


def main():
    """Run all environment checks"""
    print("="*60)
    print("TAR UMT FYP - ENVIRONMENT SETUP VERIFICATION")
    print("CrossViT for COVID-19 Chest X-ray Classification")
    print("Student: Tan Ming Kai (24PMR12003)")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Required Packages': check_required_packages(),
        'CUDA & GPU': check_cuda(),
        'CrossViT Model': check_timm_models(),
        'Disk Space': check_disk_space(),
    }
    
    print_summary(results)
    
    # Generate requirements.txt if needed
    if not results['Required Packages']:
        print("\n📝 Generating requirements.txt...")
        requirements = """# Core Deep Learning
torch==2.0.1
torchvision==0.15.2
timm==0.9.2

# Computer Vision
opencv-python==4.8.0.74
albumentations==1.3.1
scikit-image==0.21.0

# Data Science
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2

# Scientific Computing
scipy==1.11.0
scikit-learn==1.3.0

# Utilities
tqdm==4.65.0
pillow==10.0.0

# Optional
tensorboard==2.13.0
"""
        with open('requirements.txt', 'w') as f:
            f.write(requirements)
        print("   ✅ requirements.txt generated")


if __name__ == "__main__":
    main()
