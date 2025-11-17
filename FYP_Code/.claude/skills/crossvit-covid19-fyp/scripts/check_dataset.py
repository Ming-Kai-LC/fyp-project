#!/usr/bin/env python3
"""
Dataset Validation Utility
Checks integrity and statistics of COVID-19 Radiography Database

Author: Tan Ming Kai (24PMR12003)
FYP: CrossViT for COVID-19 Classification
"""

import os
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm


class DatasetChecker:
    """Validate COVID-19 Radiography Database"""
    
    def __init__(self, data_dir):
        """
        Initialize dataset checker
        
        Args:
            data_dir: Path to dataset root directory
        """
        self.data_dir = Path(data_dir)
        self.class_names = ['COVID', 'Normal', 'Lung_Opacity', 'Viral Pneumonia']
        self.stats = defaultdict(dict)
        
    def check_directory_structure(self):
        """Verify directory structure exists"""
        print("📁 Checking directory structure...")
        
        all_exist = True
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            exists = class_dir.exists()
            status = "✅" if exists else "❌"
            print(f"   {status} {class_name}: {class_dir}")
            
            if exists:
                num_files = len(list(class_dir.glob('*.png')))
                print(f"      Found {num_files} PNG files")
                self.stats[class_name]['count'] = num_files
            else:
                all_exist = False
                self.stats[class_name]['count'] = 0
        
        return all_exist
    
    def check_image_integrity(self, max_per_class=100):
        """
        Check if images can be loaded and are valid
        
        Args:
            max_per_class: Maximum images to check per class (for speed)
        """
        print(f"\n🖼️  Checking image integrity (sampling {max_per_class} per class)...")
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            image_files = list(class_dir.glob('*.png'))[:max_per_class]
            corrupted = []
            
            for img_path in tqdm(image_files, desc=f"   {class_name}", leave=False):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        corrupted.append(img_path.name)
                    elif img.size == 0:
                        corrupted.append(img_path.name)
                except Exception as e:
                    corrupted.append(img_path.name)
            
            if corrupted:
                print(f"   ❌ {class_name}: {len(corrupted)} corrupted images")
                print(f"      Examples: {corrupted[:3]}")
                self.stats[class_name]['corrupted'] = corrupted
            else:
                print(f"   ✅ {class_name}: All sampled images OK")
                self.stats[class_name]['corrupted'] = []
    
    def analyze_image_statistics(self, sample_size=100):
        """
        Analyze image properties (size, intensity, etc.)
        
        Args:
            sample_size: Number of images to sample per class
        """
        print(f"\n📊 Analyzing image statistics (sampling {sample_size} per class)...")
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                continue
            
            image_files = list(class_dir.glob('*.png'))[:sample_size]
            
            widths, heights, means, stds = [], [], [], []
            
            for img_path in tqdm(image_files, desc=f"   {class_name}", leave=False):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        h, w = img.shape
                        widths.append(w)
                        heights.append(h)
                        means.append(img.mean())
                        stds.append(img.std())
                except:
                    continue
            
            if widths:
                print(f"\n   {class_name}:")
                print(f"      Resolution: {np.mean(widths):.0f}×{np.mean(heights):.0f} "
                      f"(min: {min(widths)}×{min(heights)}, max: {max(widths)}×{max(heights)})")
                print(f"      Intensity: μ={np.mean(means):.1f}, σ={np.mean(stds):.1f}")
                
                self.stats[class_name]['resolution'] = (np.mean(widths), np.mean(heights))
                self.stats[class_name]['intensity_mean'] = np.mean(means)
                self.stats[class_name]['intensity_std'] = np.mean(stds)
    
    def check_class_distribution(self):
        """Verify class distribution matches expected values"""
        print("\n📈 Class Distribution Analysis:")
        
        expected = {
            'COVID': 3616,
            'Normal': 10192,
            'Lung_Opacity': 6012,
            'Viral Pneumonia': 1345
        }
        
        total_expected = sum(expected.values())
        total_found = sum(self.stats[cls]['count'] for cls in self.class_names)
        
        print(f"\n   Expected total: {total_expected}")
        print(f"   Found total:    {total_found}")
        
        if total_found == total_expected:
            print("   ✅ Total count matches!")
        else:
            print(f"   ⚠️  Mismatch: {abs(total_found - total_expected)} images difference")
        
        print("\n   Per-class distribution:")
        for class_name in self.class_names:
            exp = expected[class_name]
            found = self.stats[class_name]['count']
            percentage = (found / total_found * 100) if total_found > 0 else 0
            
            match = "✅" if found == exp else "⚠️"
            print(f"      {match} {class_name:20s}: {found:5d} / {exp:5d} ({percentage:.1f}%)")
    
    def calculate_imbalance_ratios(self):
        """Calculate class imbalance ratios"""
        print("\n⚖️  Class Imbalance Ratios:")
        
        counts = {cls: self.stats[cls]['count'] for cls in self.class_names}
        
        if counts['Viral Pneumonia'] > 0:
            ratio_normal_viral = counts['Normal'] / counts['Viral Pneumonia']
            print(f"   Normal : Viral Pneumonia = {ratio_normal_viral:.2f}:1")
        
        if counts['COVID'] > 0:
            ratio_normal_covid = counts['Normal'] / counts['COVID']
            print(f"   Normal : COVID-19        = {ratio_normal_covid:.2f}:1")
        
        if counts['Viral Pneumonia'] > 0:
            ratio_opacity_viral = counts['Lung_Opacity'] / counts['Viral Pneumonia']
            print(f"   Lung Opacity : Viral Pneumonia = {ratio_opacity_viral:.2f}:1")
        
        # Calculate class weights for loss function
        total = sum(counts.values())
        weights = {}
        print(f"\n   Recommended Class Weights for nn.CrossEntropyLoss:")
        for class_name in self.class_names:
            weight = total / (len(self.class_names) * counts[class_name]) if counts[class_name] > 0 else 0
            weights[class_name] = weight
            print(f"      {class_name:20s}: {weight:.2f}")
        
        return weights
    
    def run_full_check(self, quick=False):
        """
        Run complete dataset validation
        
        Args:
            quick: If True, sample fewer images for faster checking
        """
        print("="*60)
        print("COVID-19 RADIOGRAPHY DATABASE VALIDATION")
        print("="*60)
        print(f"Dataset Directory: {self.data_dir}\n")
        
        # Check directory structure
        if not self.check_directory_structure():
            print("\n❌ ERROR: Missing directories. Please check dataset path.")
            return False
        
        # Check image integrity
        sample_size = 50 if quick else 100
        self.check_image_integrity(max_per_class=sample_size)
        
        # Analyze statistics
        self.analyze_image_statistics(sample_size=sample_size)
        
        # Check distribution
        self.check_class_distribution()
        
        # Calculate imbalance
        weights = self.calculate_imbalance_ratios()
        
        # Summary
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        total_corrupted = sum(len(self.stats[cls].get('corrupted', [])) for cls in self.class_names)
        
        if total_corrupted == 0:
            print("✅ All sampled images are valid")
        else:
            print(f"⚠️  Found {total_corrupted} corrupted images")
        
        print("✅ Dataset structure is correct")
        print("✅ Ready for training!\n")
        
        return True


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate COVID-19 Radiography Database')
    parser.add_argument('data_dir', type=str, help='Path to dataset directory')
    parser.add_argument('--quick', action='store_true', help='Quick check (sample fewer images)')
    
    args = parser.parse_args()
    
    checker = DatasetChecker(args.data_dir)
    checker.run_full_check(quick=args.quick)


if __name__ == "__main__":
    # Example usage
    print("Dataset Validation Utility")
    print("Usage: python check_dataset.py /path/to/dataset")
    print("\nFor this FYP, your dataset should be structured as:")
    print("   dataset_root/")
    print("   ├── COVID/")
    print("   ├── Normal/")
    print("   ├── Lung_Opacity/")
    print("   └── Viral Pneumonia/")
    print("\nEach directory should contain PNG images.")
    print("\nRun with actual path to perform validation.")
