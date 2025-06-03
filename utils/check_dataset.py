#!/usr/bin/env python3
"""
Simple script to check dataset composition without torch/cv2 dependencies.
This helps debug issues where certain classes might be missing.
"""

import os
import yaml
import argparse
from collections import defaultdict

def check_dataset_structure(data_dir):
    """Check the structure and composition of the dataset."""
    print(f"=== Analyzing Dataset Structure in {data_dir} ===")
    
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return
    
    # Check for standard dataset structure
    subdirs = ['train', 'test', 'val']
    found_subdirs = []
    
    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if os.path.exists(subdir_path):
            found_subdirs.append(subdir)
            print(f"✓ Found {subdir} directory")
        else:
            print(f"❌ Missing {subdir} directory")
    
    if not found_subdirs:
        print("❌ No standard train/test/val directories found")
        print("Checking for other directory structure...")
        
        # List all directories
        try:
            items = os.listdir(data_dir)
            dirs = [item for item in items if os.path.isdir(os.path.join(data_dir, item))]
            print(f"Found directories: {dirs}")
        except Exception as e:
            print(f"Error listing directories: {e}")
        return
    
    # Analyze each split
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    
    for split in found_subdirs:
        print(f"\n--- Analyzing {split} split ---")
        split_path = os.path.join(data_dir, split)
        
        try:
            # Check if it's organized by class folders
            items = os.listdir(split_path)
            class_folders = [item for item in items if os.path.isdir(os.path.join(split_path, item))]
            
            if class_folders:
                print(f"Found class folders: {class_folders}")
                
                total_samples = 0
                for class_folder in class_folders:
                    class_path = os.path.join(split_path, class_folder)
                    try:
                        files = [f for f in os.listdir(class_path) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                        count = len(files)
                        total_samples += count
                        print(f"  {class_folder}: {count} images")
                        
                        if count == 0:
                            print(f"    ⚠️  No images found in {class_folder}!")
                        elif count < 10:
                            print(f"    ⚠️  Very few samples in {class_folder}")
                            
                    except Exception as e:
                        print(f"  Error reading {class_folder}: {e}")
                
                print(f"Total {split} samples: {total_samples}")
                
                # Check for class imbalance
                if class_folders:
                    class_counts = []
                    for class_folder in class_folders:
                        class_path = os.path.join(split_path, class_folder)
                        try:
                            files = [f for f in os.listdir(class_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                            class_counts.append(len(files))
                        except:
                            class_counts.append(0)
                    
                    if max(class_counts) > 0:
                        min_count = min(class_counts)
                        max_count = max(class_counts)
                        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                        
                        if imbalance_ratio > 10:
                            print(f"    ⚠️  Severe class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
                        elif imbalance_ratio > 3:
                            print(f"    ⚠️  Class imbalance detected. Ratio: {imbalance_ratio:.1f}:1")
                        
                        # Check for missing classes
                        expected_classes = ['Normal', 'CHF', 'pneumonia']
                        for expected in expected_classes:
                            found = any(expected.lower() in folder.lower() for folder in class_folders)
                            if not found:
                                print(f"    ❌ Missing expected class: {expected}")
            
            else:
                # Check if images are directly in the split folder
                files = [f for f in items if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if files:
                    print(f"Found {len(files)} images directly in {split} folder")
                    print("⚠️  Images not organized by class - may need different data loading approach")
                else:
                    print(f"No images or class folders found in {split}")
                    
        except Exception as e:
            print(f"Error analyzing {split}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Check dataset composition without torch dependencies.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to experiment results directory')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Override data directory path')
    
    args = parser.parse_args()
    
    # Load configuration to get data directory
    config_path = os.path.join(args.results_dir, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_dir = args.data_dir if args.data_dir else config.get('data_dir')
    
    if not data_dir:
        print("❌ No data directory specified in config or arguments")
        return
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Model: {config.get('model_type', 'unknown')} ({config.get('model_mode', 'unknown')})")
    print(f"Classes: {config.get('num_classes', 'unknown')}")
    print(f"Data directory: {data_dir}")
    
    check_dataset_structure(data_dir)

if __name__ == "__main__":
    main() 