#!/usr/bin/env python3
"""
Check Topological Features

This script checks if the topological features directory has the correct structure
and if the feature files are properly formatted.
"""

import os
import numpy as np
from pathlib import Path
import argparse
import random
import sys

def check_directory_structure(data_dir, topo_dir=None):
    """Check if directory structure is correct"""
    if topo_dir is None:
        topo_dir = os.path.join(data_dir, "topo_features")
    
    # Check if base directories exist
    print(f"Checking directory: {topo_dir}")
    if not os.path.exists(topo_dir):
        print(f"ERROR: Topo features directory {topo_dir} not found!")
        return False
    
    # Check train/test directories
    for mode in ['train', 'test']:
        mode_dir = os.path.join(topo_dir, mode)
        if not os.path.exists(mode_dir):
            print(f"ERROR: {mode} directory not found in {topo_dir}")
            return False
        print(f"✓ Found {mode} directory: {mode_dir}")
        
        # Check class directories
        class_dirs = [d for d in os.listdir(mode_dir) if os.path.isdir(os.path.join(mode_dir, d))]
        if not class_dirs:
            print(f"ERROR: No class directories found in {mode_dir}")
            return False
        print(f"✓ Found {len(class_dirs)} class directories: {', '.join(class_dirs)}")
        
        # Sample check a few files
        for class_name in class_dirs:
            class_dir = os.path.join(mode_dir, class_name)
            files = [f for f in os.listdir(class_dir) if f.endswith(".npy")]
            
            if not files:
                print(f"ERROR: No .npy files found in {class_dir}")
                return False
            
            print(f"✓ Found {len(files)} .npy files in {class_dir}")
            
            # Check a few random files
            sample_size = min(5, len(files))
            sampled_files = random.sample(files, sample_size)
            
            for file in sampled_files:
                file_path = os.path.join(class_dir, file)
                try:
                    # Load and check dimensions
                    features = np.load(file_path)
                    if features.shape != (2, 56, 56):
                        print(f"ERROR: File {file_path} has incorrect shape: {features.shape}, expected (2, 56, 56)")
                        return False
                except Exception as e:
                    print(f"ERROR: Failed to load {file_path}: {e}")
                    return False
            
            print(f"✓ Sampled {sample_size} files from {class_dir} verified correctly")
    
    # Check for matching images and topo features
    print("\nChecking for matching image and feature files...")
    for mode in ['train', 'test']:
        img_mode_dir = os.path.join(data_dir, mode)
        topo_mode_dir = os.path.join(topo_dir, mode)
        
        for class_name in os.listdir(img_mode_dir):
            if not os.path.isdir(os.path.join(img_mode_dir, class_name)):
                continue
                
            img_class_dir = os.path.join(img_mode_dir, class_name)
            topo_class_dir = os.path.join(topo_mode_dir, class_name)
            
            # Get image files
            img_files = [os.path.splitext(f)[0] for f in os.listdir(img_class_dir) if f.endswith('.jpg')]
            topo_files = [os.path.splitext(f)[0] for f in os.listdir(topo_class_dir) if f.endswith('.npy')]
            
            img_set = set(img_files)
            topo_set = set(topo_files)
            
            missing = img_set - topo_set
            if missing:
                print(f"WARNING: {len(missing)} images in {img_class_dir} do not have corresponding topo features")
                if len(missing) < 10:
                    print(f"Missing files: {', '.join(list(missing))}")
                else:
                    print(f"First 10 missing files: {', '.join(list(missing)[:10])}")
            else:
                print(f"✓ All {len(img_files)} images in {class_name}/{mode} have corresponding topo features")
    
    print("\nDirectory structure check completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Check if topological features are properly structured")
    parser.add_argument("--data_dir", default="/mnt/f/Datasets/physionet.org/files/mimic_part_jpg", 
                        help="Base data directory containing images and topo_features")
    parser.add_argument("--topo_dir", default=None,
                        help="Alternative topo_features directory if not in data_dir")
    
    args = parser.parse_args()
    
    result = check_directory_structure(args.data_dir, args.topo_dir)
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 