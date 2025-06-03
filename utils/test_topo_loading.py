#!/usr/bin/env python3
"""
Test Topological Feature Loading

This script tests loading topological features with the updated dataset class.
It verifies that:
1. The dataset can be created
2. Individual items can be loaded
3. DataLoader batches can be created without errors
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add parent directory to path to import modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import the dataset class
from utils.read_data_topo import dataset, read_mimic

def inspect_topo_files(data_dir, mode='train'):
    """Inspect a few topological feature files to check for corruption"""
    print(f"\n--- Inspecting Topological Feature Files ({mode}) ---")
    
    # Check topo_features directory structure
    topo_dir = os.path.join(data_dir, "topo_features", mode)
    alt_topo_dir = os.path.join(data_dir, "topo_features_new", "topo_features", mode)
    
    # Check primary directory
    if os.path.exists(topo_dir):
        print(f"Primary topo directory exists: {topo_dir}")
        label_dirs = [d for d in os.listdir(topo_dir) if os.path.isdir(os.path.join(topo_dir, d))]
        print(f"Found label directories: {label_dirs}")
        
        # Check one file from each label
        for label in label_dirs:
            label_dir = os.path.join(topo_dir, label)
            files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
            
            if files:
                # Check a random file
                sample_file = files[0]
                sample_path = os.path.join(label_dir, sample_file)
                try:
                    data = np.load(sample_path)
                    print(f"Label: {label}, Sample file: {sample_file}")
                    print(f"  - Shape: {data.shape}")
                    print(f"  - Min/Max: {data.min():.4f}/{data.max():.4f}")
                    print(f"  - Any NaN: {np.isnan(data).any()}")
                    print(f"  - Any Inf: {np.isinf(data).any()}")
                except Exception as e:
                    print(f"Error loading {sample_path}: {e}")
    else:
        print(f"Primary topo directory does not exist: {topo_dir}")
    
    # Check alternate directory
    if os.path.exists(alt_topo_dir):
        print(f"\nAlternate topo directory exists: {alt_topo_dir}")
        label_dirs = [d for d in os.listdir(alt_topo_dir) if os.path.isdir(os.path.join(alt_topo_dir, d))]
        print(f"Found label directories: {label_dirs}")
        
        # Check one file from each label
        for label in label_dirs:
            label_dir = os.path.join(alt_topo_dir, label)
            files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
            
            if files:
                # Check a random file
                sample_file = files[0]
                sample_path = os.path.join(label_dir, sample_file)
                try:
                    data = np.load(sample_path)
                    print(f"Label: {label}, Sample file: {sample_file}")
                    print(f"  - Shape: {data.shape}")
                    print(f"  - Min/Max: {data.min():.4f}/{data.max():.4f}")
                    print(f"  - Any NaN: {np.isnan(data).any()}")
                    print(f"  - Any Inf: {np.isinf(data).any()}")
                except Exception as e:
                    print(f"Error loading {sample_path}: {e}")
    else:
        print(f"Alternate topo directory does not exist: {alt_topo_dir}")

def test_single_item_loading(data_dir):
    """Test loading a single item from the dataset"""
    print("\n--- Testing Single Item Loading ---")
    
    from torchvision import transforms
    simple_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    
    # Create dataset with minimal transforms
    train_dataset = dataset(data_dir=data_dir, mode='train', transform=simple_transform)
    print(f"Dataset created with {len(train_dataset)} items")
    
    # Try to load first few items from the dataset
    for idx in range(min(3, len(train_dataset))):
        try:
            print(f"\nLoading item {idx}:")
            img, label, topo = train_dataset[idx]
            
            print(f"  - Image shape: {img.shape}")
            print(f"  - Image min/max: {img.min():.4f}/{img.max():.4f}")
            print(f"  - Label: {label}")
            print(f"  - Topo shape: {topo.shape}")
            print(f"  - Topo min/max: {topo.min():.4f}/{topo.max():.4f}")
            print(f"  - Topo has NaN: {torch.isnan(topo).any().item()}")
            print(f"  - Topo has Inf: {torch.isinf(topo).any().item()}")
            
        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            import traceback
            traceback.print_exc()

def test_dataloader(data_dir, batch_size=4):
    """Test creating and iterating through a DataLoader"""
    print(f"\n--- Testing DataLoader with batch_size={batch_size} ---")
    
    try:
        # Get DataLoaders
        train_loader, test_loader = read_mimic(batchsize=batch_size, data_dir=data_dir)
        
        print(f"Created DataLoaders:")
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Test loader: {len(test_loader)} batches")
        
        # Try to load first batch from train_loader
        print("\nTrying to load first batch from train_loader:")
        batch = next(iter(train_loader))
        imgs, labels, topo_features = batch
        
        print(f"  - Batch shapes:")
        print(f"    - Images: {imgs.shape}")
        print(f"    - Labels: {labels.shape}")
        print(f"    - Topo: {topo_features.shape}")
        
        # Try to load first batch from test_loader
        print("\nTrying to load first batch from test_loader:")
        batch = next(iter(test_loader))
        imgs, labels, topo_features = batch
        
        print(f"  - Batch shapes:")
        print(f"    - Images: {imgs.shape}")
        print(f"    - Labels: {labels.shape}")
        print(f"    - Topo: {topo_features.shape}")
        
        print("\nSuccessfully loaded batches from both DataLoaders!")
        
    except Exception as e:
        print(f"Error in DataLoader test: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Set default data directory (should be changed to match your actual path)
    default_data_dir = '/mnt/f/Datasets/physionet.org/files/mimic_part_jpg'
    
    # Allow overriding from command line
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = default_data_dir
        
    print(f"Using data directory: {data_dir}")
    
    # If directory doesn't exist, look for a better default
    if not os.path.isdir(data_dir):
        print(f"Warning: Directory not found: {data_dir}")
        # Try to find an alternative path
        alt_paths = [
            os.path.expanduser("~/mimic_part_jpg"),
            os.path.expanduser("~/data/mimic_part_jpg"),
            os.path.expanduser("~/datasets/mimic_part_jpg"),
            "../data/mimic_part_jpg",
            "../mimic_part_jpg",
        ]
        
        for path in alt_paths:
            if os.path.isdir(path):
                print(f"Found alternative data directory: {path}")
                data_dir = path
                break
        else:
            print("Error: Could not find a valid data directory.")
            print("Please provide a valid data directory path as an argument.")
            return 1
    
    # Run the tests
    inspect_topo_files(data_dir, mode='train')
    inspect_topo_files(data_dir, mode='test')
    test_single_item_loading(data_dir)
    test_dataloader(data_dir, batch_size=4)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 