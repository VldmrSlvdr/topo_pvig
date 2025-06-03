from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import numpy as np
import os
from glob import glob
import sys


def read_mimic(batchsize, data_dir='/mnt/f/Datasets/physionet.org/files/mimic_part_jpg', num_workers=0):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-5,5)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4763, 0.4763, 0.4763], 
                                 std=[0.2988, 0.2988, 0.2988]) # from calculate_normalize.py
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #                      std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.476, 0.476, 0.476], 
            #                     std=[0.299, 0.299, 0.299]
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4763, 0.4763, 0.4763], 
                                 std=[0.2988, 0.2988, 0.2988]) # from calculate_normalize.py
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            #             std=[0.229, 0.224, 0.225])
            # transforms.Normalize(mean=[0.476, 0.476, 0.476], 
            #                     std=[0.299, 0.299, 0.299]
        ]),
    }
    
    # use pin_memory=True if using GPU
    pin_memory_flag = torch.cuda.is_available()
    
    # Create datasets
    try:
        image_datasets = {x: dataset(mode=x, transform=data_transforms[x], data_dir=data_dir)
                      for x in ['train', 'test']}
        
        print(f"Created datasets. Train size: {len(image_datasets['train'])}, Test size: {len(image_datasets['test'])}")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        raise

    # Explicitly disable multiprocessing for DataLoader
    # This is crucial for MPS compatibility and to avoid pickle errors
    data_loader_train = DataLoader(
        dataset=image_datasets['train'],
        batch_size=batchsize,
        shuffle=True,
        pin_memory=pin_memory_flag,
        num_workers=0,  # Force 0 to avoid multiprocessing issues
        persistent_workers=False  # Ensure workers don't persist between epochs
    )
    
    data_loader_test = DataLoader(
        dataset=image_datasets['test'],
        batch_size=batchsize,
        shuffle=False,
        pin_memory=pin_memory_flag,
        num_workers=0,  # Force 0 to avoid multiprocessing issues
        persistent_workers=False  # Ensure workers don't persist between epochs
    )
    
    return data_loader_train, data_loader_test

class dataset(Dataset):

    def __init__(self, data_dir='../mimic_part_jpg', mode="train", transform=None):
        """
        Dataset class for loading medical images with topological features.
        
        Args:
            data_dir (str): Base data directory
            mode (str): 'train' or 'test'
            transform: Image transformations to apply
        """
        self.root = data_dir
        self.mode = mode
        self.T = transform
        self.labels = ["CHF", "Normal", "pneumonia"]
        self.labelsdict = {"CHF": 0, "Normal": 1, "pneumonia": 2}
        
        # Collect all image paths and verify topo features exist
        self.idlist = []
        self.topo_paths = []  # Store topo paths alongside image paths
        
        for label in self.labels:
            pattern = os.path.join(self.root, self.mode, label, "*.jpg")
            paths = glob(pattern)
            if len(paths) == 0:
                print(f"Warning: No images found for {label} in {pattern}")
                continue
                
            for imgpath in paths:
                img_id = os.path.splitext(os.path.basename(imgpath))[0]
                topo_path = os.path.join(self.root, "topo_features", self.mode, label, f"{img_id}.npy")
                
                # Only add if both image and topo features exist
                if os.path.exists(topo_path):
                    self.idlist.append(imgpath)
                    self.topo_paths.append(topo_path)
                else:
                    print(f"Warning: Missing topo features for {img_id}, skipping")
        
        if len(self.idlist) == 0:
            raise ValueError(f"No valid image-topo pairs found in {data_dir}/{mode}/*/*.jpg")
        
        print(f"Found {len(self.idlist)} valid image-topo pairs in {mode} mode")
        
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, idx):
        """Get item by index"""
        # Handle out of bounds idx
        if idx >= len(self.idlist):
            print(f"Warning: Index {idx} out of bounds (dataset size: {len(self.idlist)})")
            idx = idx % len(self.idlist)

        # --- Get image path and ID ---
        imgpath = self.idlist[idx]
        topopath = self.topo_paths[idx]
        parts = imgpath.split(os.path.sep)
        label_name = parts[-2]

        # --- Extract image safely ---
        try:
            with open(imgpath, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
        except Exception as e:
            print(f"Error loading image {imgpath}: {e}")
            # Return a blank image as fallback
            img = Image.new('RGB', (224, 224), color='gray')
            
        # --- Extract label ---
        label = self.labelsdict.get(label_name, 0)  # Default to 0 if label not found

        # --- Load topo features ---
        try:
            topo_features_np = np.load(topopath, allow_pickle=False)
            if topo_features_np.shape != (2, 56, 56):
                raise ValueError(f"Unexpected topo feature shape: {topo_features_np.shape}")
            if np.isnan(topo_features_np).any():
                raise ValueError("NaN values found in topo features")
            topo_features = torch.from_numpy(topo_features_np).float()
        except Exception as e:
            print(f"Error loading topo features from {topopath}: {e}")
            topo_features = torch.zeros((2, 56, 56), dtype=torch.float32)

        # --- Apply image transform safely ---
        if self.T is not None:
            try:
                # Save random state to ensure consistent augmentation if needed
                state = torch.get_rng_state()
                img = self.T(img)
                torch.set_rng_state(state)
            except Exception as e:
                print(f"Error applying transform to {imgpath}: {e}")
                # Return a blank transformed tensor as fallback
                img = torch.zeros((3, 224, 224), dtype=torch.float32)

        # --- Return data ---
        return img, label, topo_features