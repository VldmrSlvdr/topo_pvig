from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
import numpy as np
import os
from glob import glob


def read_mimic_topo(batchsize, data_dir = '/mnt/f/Datasets/physionet.org/files/mimic_part_jpg'):
    """Reads MIMIC images and corresponding pre-computed topo features."""
    # Kept transforms same as original read_data.py for the image
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation((-5,5)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.476, 0.476, 0.476], 
                                 std=[0.299, 0.299, 0.299]) # from calculate_normalize.py
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.476, 0.476, 0.476], 
                                 std=[0.299, 0.299, 0.299]) # from calculate_normalize.py
        ]),
    }
    
    image_datasets = {x: TopoDataset(mode=x, transform=data_transforms[x], data_dir=data_dir)
                      for x in ['train', 'test']}
    
    # Use pin_memory=True if using GPU
    pin_memory_flag = torch.cuda.is_available()
    
    data_loader_train = DataLoader(dataset=image_datasets['train'],
                                   batch_size=batchsize,
                                   shuffle=True,
                                   pin_memory=pin_memory_flag,
                                   num_workers=4 # Adjust num_workers based on your system
                                   )
    data_loader_test = DataLoader(dataset=image_datasets['test'],
                                  batch_size=batchsize,
                                  shuffle=False,
                                  pin_memory=pin_memory_flag,
                                  num_workers=4 # Adjust num_workers based on your system
                                  )
    
    return data_loader_train, data_loader_test

class TopoDataset(Dataset):
    """Dataset class for loading images and their corresponding topo features."""

    def __init__(self, data_dir='/mnt/f/Datasets/physionet.org/files/mimic_part_jpg', mode="train", transform=None):

        self.root = data_dir
        self.mode = mode
        self.T = transform # Image transform
        # No longer need fixations.csv or topo.csv here
        # self.csv = pd.read_csv(os.path.join(self.root, "gaze", "fixations.csv"))
        # self.topo = pd.read_csv(os.path.join(self.root, "topo_heatmap", "topography.csv"))
        self.labels = ["CHF", "Normal", "pneumonia"]
        self.labelsdict = {"CHF": 0, "Normal": 1, "pneumonia": 2}
        self.idlist = []
        # Find all image files
        for label_name in self.labels:
            self.idlist.extend(glob(os.path.join(self.root, self.mode, label_name, "*.jpg")))
        
        print(f"Found {len(self.idlist)} images for mode '{self.mode}'")
        
    def __len__(self):
        return len(self.idlist)

    def __getitem__(self, idx):

        # --- Get image path and ID ---
        imgpath = self.idlist[idx]
        # Use os.path.sep for cross-platform compatibility
        parts = imgpath.split(os.path.sep)
        img_filename = parts[-1]
        label_name = parts[-2]
        id = os.path.splitext(img_filename)[0] # Get ID without extension

        # --- Construct topo feature path ---
        topopath = os.path.join(self.root, "topo_features", self.mode, label_name, f"{id}.npy")

        # --- Extract image --- 
        try:
            with open(imgpath, "rb") as f:
                img = Image.open(f)
                img = img.convert("RGB")
        except Exception as e:
            print(f"Error loading image {imgpath}: {e}")
            # Return dummy data or raise error
            # For simplicity, we might return None and handle it in DataLoader collation
            # Or, return a dummy tensor of the expected shape
            dummy_img = torch.zeros((3, 224, 224))
            dummy_topo = torch.zeros((2, 56, 56))
            return dummy_img, -1, dummy_topo # Indicate error with label -1

        # --- Extract label --- 
        label = self.labelsdict[label_name]

        # --- Load topo features --- 
        try:
            if os.path.exists(topopath):
                topo_features_np = np.load(topopath)
                # Ensure correct shape
                if topo_features_np.shape == (2, 56, 56):
                    topo_features = torch.from_numpy(topo_features_np).float() # Ensure float tensor
                else:
                    print(f"Warning: Unexpected shape {topo_features_np.shape} for {topopath}. Expected (2, 56, 56). Returning zeros.")
                    topo_features = torch.zeros((2, 56, 56))
            else:
                print(f"Warning: Topo feature file not found: {topopath}. Returning zeros.")
                topo_features = torch.zeros((2, 56, 56))
        except Exception as e:
             print(f"Error loading topo features {topopath}: {e}")
             topo_features = torch.zeros((2, 56, 56))

        # --- Apply image transform --- 
        # NOTE: No random state preservation needed as only image is transformed randomly
        if self.T:
            img = self.T(img)
        # The normalization from the original script seems specific to ImageNet?
        # Re-check if the normalization constants are correct for your data
        # img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # --- Return data --- 
        return img, label, topo_features

    # getPatchGaze method is removed as it's no longer used
    # def getPatchGaze(self, gaze):
    #     ...

# Example usage (optional)
if __name__ == '__main__':
    print("Testing DataLoader creation...")
    # Example: Create dataloaders with batch size 4
    data_dir = '/mnt/f/Datasets/physionet.org/files/mimic_part_jpg' 
    try:
        train_loader, test_loader = read_mimic_topo(batchsize=4, data_dir=data_dir)
        
        print("Iterating through one batch of train data...")
        # Get one batch
        img_batch, label_batch, topo_batch = next(iter(train_loader))
        
        # Print shapes
        print(f"Image batch shape: {img_batch.shape}")     # Should be [4, 3, 224, 224]
        print(f"Label batch shape: {label_batch.shape}")     # Should be [4]
        print(f"Topo feature batch shape: {topo_batch.shape}") # Should be [4, 2, 56, 56]
        
        print("\nIterating through one batch of test data...")
        img_batch_test, label_batch_test, topo_batch_test = next(iter(test_loader))
        print(f"Image batch shape (test): {img_batch_test.shape}")
        print(f"Label batch shape (test): {label_batch_test.shape}")
        print(f"Topo feature batch shape (test): {topo_batch_test.shape}")
        
        print("\nDataLoader setup successful.")
        
    except FileNotFoundError as e:
        print(f"\nError during testing: {e}")
        print("Please ensure the data directory and required subdirectories exist.")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
        import traceback
        traceback.print_exc()


