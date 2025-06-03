#!/usr/bin/env python3
"""
Visualize attention/gating maps from trained TopoGNN models.
This script helps understand what features the model is focusing on,
particularly how topological dimensions are being weighted.
"""

import os
import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import importlib
import sys
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
import cv2
from pathlib import Path
import random
from skimage import transform
import matplotlib.cm as cm
from PIL import ImageDraw, ImageFont

# Ensure the script can find other modules in the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.read_data_topo import read_mimic, Dataset

# --- Configuration and Utility Functions ---

def load_config(results_dir):
    """Load configuration from results directory."""
    config_path = os.path.join(results_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")
    return config

def load_model(config, results_dir, device):
    """Load model from best accuracy checkpoint."""
    model_type = config.get('model_type', 'pvig_ti')
    model_mode = config.get('model_mode', 'proj')
    num_classes = config['num_classes']
    drop_path_rate = config.get('drop_path_rate', 0.0)
    
    model_func_name = f"{model_type}_224_gelu_{model_mode}"
    module_name = f"models.pvig_topo_{model_mode}"
    
    print(f"Loading model function '{model_func_name}' from module '{module_name}'...")
    try:
        model_module = importlib.import_module(module_name)
        model_func = getattr(model_module, model_func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load model function '{model_func_name}' from {module_name}. {e}")
        sys.exit(1)
        
    model = model_func(num_classes=num_classes, drop_path_rate=drop_path_rate, pretrained=False)
    
    # Try to load best accuracy model first
    checkpoint_path = os.path.join(results_dir, 'best_acc_model.pth')
    if not os.path.exists(checkpoint_path):
        # Fallback to other checkpoints if best_acc doesn't exist
        for alternative in ['best_auc_model.pth', 'final_model.pth']:
            alt_path = os.path.join(results_dir, alternative)
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                print(f"best_acc_model.pth not found. Using {alternative} instead.")
                break
        else:
            raise FileNotFoundError(f"No checkpoint found in {results_dir}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle potential DataParallel prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    is_data_parallel = False
    for k, v in state_dict.items():
        if k.startswith('module.'):
            is_data_parallel = True
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
            
    if is_data_parallel:
        print("Checkpoint was saved using DataParallel. Removing 'module.' prefix.")
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully. Epoch: {checkpoint.get('epoch', 'unknown')}")
    return model, checkpoint.get('epoch', 0)

def analyze_test_dataset(model, data_dir, device):
    """Analyze the test dataset to show class distribution and model performance."""
    print("=== Analyzing Test Dataset ===")
    
    try:
        _, test_loader = read_mimic(batchsize=1, data_dir=data_dir)
        
        class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
        class_counts = {0: 0, 1: 0, 2: 0}
        correct_predictions = {0: 0, 1: 0, 2: 0}
        total_samples = 0
        
        model.eval()
        
        print("Scanning test dataset...")
        for i, (images, labels, topo_features) in enumerate(test_loader):
            if i % 100 == 0 and i > 0:
                print(f"Processed {i} samples...")
                
            label = labels[0].item()
            class_counts[label] += 1
            total_samples += 1
            
            # Check prediction
            img_tensor = images[0].unsqueeze(0).to(device)
            topo_tensor = topo_features[0].unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(img_tensor, topo_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item()
                
                if predicted_label == label:
                    correct_predictions[label] += 1
            
            # Don't scan the entire dataset if it's very large
            if i >= 500:  # Limit to first 500 samples for analysis
                print(f"Analyzed first {i+1} samples (stopping early to save time)")
                break
        
        print(f"\n=== Dataset Analysis Results ===")
        print(f"Total samples analyzed: {total_samples}")
        
        for class_idx, class_name in class_names.items():
            count = class_counts[class_idx]
            correct = correct_predictions[class_idx]
            accuracy = (correct / count * 100) if count > 0 else 0
            
            print(f"{class_name}: {count} samples, {correct} correctly predicted ({accuracy:.1f}% accuracy)")
            
            if count == 0:
                print(f"  ⚠️  No {class_name} samples found in test set!")
            elif correct == 0:
                print(f"  ⚠️  No correctly predicted {class_name} samples found!")
        
        return class_counts, correct_predictions
        
    except Exception as e:
        print(f"Error analyzing test dataset: {e}")
        return None, None

def get_random_test_sample(model, data_dir, device, target_class=None, max_attempts=100, allow_incorrect=False):
    """Get a random sample from the test dataset that the model predicts correctly.
    
    Args:
        model: The trained model
        data_dir: Path to data directory
        device: Device to run inference on
        target_class: Specific class to find (0=Normal, 1=CHF, 2=pneumonia). If None, any correct prediction.
        max_attempts: Maximum number of samples to try
        allow_incorrect: If True, return incorrect predictions if no correct ones found
    """
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    
    if target_class is not None:
        target_name = class_names.get(target_class, f'Class {target_class}')
        print(f"Looking for correctly predicted {target_name} sample...")
    else:
        print("Loading test data to select a random correctly-predicted sample...")
    
    try:
        # Only need the test loader
        _, test_loader = read_mimic(batchsize=1, data_dir=data_dir)
        
        # Try random samples until finding a correctly predicted one or reaching max attempts
        attempts = 0
        samples_checked = 0
        found_class_samples = 0
        last_sample_of_class = None
        
        model.eval()
        
        for i, (images, labels, topo_features) in enumerate(test_loader):
            label = labels[0].item()
            
            # If we're looking for a specific class, skip samples that don't match
            if target_class is not None and label != target_class:
                continue
                
            found_class_samples += 1
            img_tensor = images[0].unsqueeze(0).to(device)
            topo_tensor = topo_features[0].unsqueeze(0).to(device)
            
            # Store this sample in case we need it as fallback
            last_sample_of_class = (img_tensor, topo_tensor, class_names.get(label, f"Class {label}"))
            
            # Only consider some samples randomly to avoid checking every single one
            if random.random() < 0.15 or attempts >= max_attempts-10:  # Higher chance in later attempts
                
                # Run inference to check prediction
                with torch.no_grad():
                    outputs = model(img_tensor, topo_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_label = predicted.item()
                
                # Check if prediction matches the true label
                is_correct = (predicted_label == label)
                attempts += 1
                
                if is_correct:
                    print(f"Found correctly predicted {class_names.get(label, f'Class {label}')} sample on attempt {attempts}")
                    return img_tensor, topo_tensor, class_names.get(label, f"Class {label}")
                
                if attempts >= max_attempts:
                    break
                
                if attempts % 10 == 0:
                    target_desc = class_names.get(target_class, 'any class') if target_class is not None else 'any class'
                    print(f"Tried {attempts} samples for {target_desc}, still looking...")
            
            samples_checked += 1
            if samples_checked >= 1000:  # Don't check too many samples
                break
        
        # If we didn't find a correctly predicted sample
        if target_class is not None:
            target_name = class_names.get(target_class, f'Class {target_class}')
            if found_class_samples == 0:
                print(f"❌ No {target_name} samples found in test dataset!")
                return None, None, None
            elif allow_incorrect and last_sample_of_class is not None:
                print(f"⚠️  No correctly predicted {target_name} samples found after {attempts} attempts.")
                print(f"   Using an incorrectly predicted {target_name} sample instead.")
                return last_sample_of_class
            else:
                print(f"❌ No correctly predicted {target_name} samples found after {attempts} attempts.")
                print(f"   Found {found_class_samples} {target_name} samples total.")
                return None, None, None
                
    except Exception as e:
        print(f"Error getting test sample for {class_names.get(target_class, 'any class')}: {e}")
        return None, None, None
    
    # If looking for any class, try with the first sample
    if target_class is None:
        for images, labels, topo_features in test_loader:
            img_tensor = images[0].unsqueeze(0).to(device)
            topo_tensor = topo_features[0].unsqueeze(0).to(device)
            label = labels[0].item()
            class_name = class_names.get(label, f"Class {label}")
            print("Warning: Couldn't find correctly predicted sample. Using first sample from dataset.")
            return img_tensor, topo_tensor, class_name
    
    return None, None, None

def prepare_custom_image(image_path, device, model, size=224):
    """Prepare a custom image for model input and check if it's correctly predicted."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1)
        # Normalize to match ImageNet stats (used in model training)
        img_tensor = img_tensor / 255.0
        img_tensor = torch.unsqueeze(img_tensor, 0).to(device)
        
        # We can't easily generate topo features here, so return None
        # The caller will need to use sample topo features
        return img_tensor, None, "Custom Image"
    except Exception as e:
        print(f"Error loading custom image: {e}")
        return None, None, None

# --- Hook and Activation Extraction ---

class FeatureExtractor:
    """Extract intermediate activations from the model."""
    def __init__(self, model, layer_names=None):
        self.model = model
        self.activations = {}
        self.hooks = []
        self.mode = None
        
        # Determine model mode
        if hasattr(model, 'topo_gate_combine'):
            self.mode = 'gated'
        elif hasattr(model, 'topo_proj'):
            self.mode = 'proj'
        else:
            for name, module in model.named_modules():
                if 'dim0' in name:
                    self.mode = 'dim0'
                    break
                elif 'dim1' in name:
                    self.mode = 'dim1'
                    break
        
        if not self.mode:
            self.mode = 'unknown'
        print(f"Detected model mode: {self.mode}")
        
        # Set up hooks
        self._register_hooks(layer_names)
    
    def _register_hooks(self, layer_names):
        """Register forward hooks to extract activations."""
        # Always get gate values for gated models
        if self.mode == 'gated' and hasattr(self.model, 'topo_gate_combine'):
            def gate_hook(module, input, output):
                # Store the gate values
                self.activations['gate'] = module.gate(input[0]).detach()
            self.hooks.append(self.model.topo_gate_combine.register_forward_hook(gate_hook))
            
            # Also get the raw topo features and fused output
            def topo_hook(module, input, output):
                # Store the input topo features
                self.activations['topo_raw'] = input[0].detach()
                # Store the fused+projected output
                self.activations['topo_fused'] = output.detach()
            self.hooks.append(self.model.topo_gate_combine.register_forward_hook(topo_hook))
        
        # Get backbone features at pyramid levels
        if hasattr(self.model, 'pyramid_levels'):
            pyramid_levels = self.model.pyramid_levels
            
            # Register hooks for each pyramid level
            idx = 0
            for i, module in enumerate(self.model.backbone):
                def hook_fn(module, input, output, level_idx=i):
                    if hasattr(self.model, 'pyramid_levels') and level_idx in self.model.pyramid_levels:
                        level = self.model.pyramid_levels.index(level_idx)
                        self.activations[f'pyramid_level_{level}'] = output.detach()
                        
                self.hooks.append(module.register_forward_hook(hook_fn))
        
        # Additional hooks for attention visualization
        # For GCN attention in Grapher modules, we need to access internal components
        # of the model's backbone
        idx = 0
        for name, module in self.model.named_modules():
            if 'grapher.grapher' in name:
                # Hook to capture graph attention scores inside Grapher modules
                def gcn_hook(module, input, output, idx=idx):
                    # The attention scores might be internal to the Grapher
                    # We can't easily access them directly via hooks, but can try to
                    # visualize the key information exchanged in graph edges later
                    self.activations[f'grapher_{idx}'] = output.detach()
                    
                self.hooks.append(module.register_forward_hook(gcn_hook))
                idx += 1
    
    def remove_hooks(self):
        """Remove all hooks to clean up."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        """Clean up hooks when object is deleted."""
        self.remove_hooks()

# --- Visualization Functions ---

def visualize_gate_values(gate_values, output_dir, prefix=''):
    """Visualize gate values that weight dim0 vs dim1 features."""
    # gate_values shape: [1, 1, H, W]
    if gate_values is None:
        print("No gate values available to visualize.")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Get the gate weights (probability of choosing dim0 over dim1)
    gate = gate_values.squeeze().cpu().numpy()
    
    # Create custom colormap (red = dim1, blue = dim0)
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue -> White -> Red
    cmap = LinearSegmentedColormap.from_list('dim0_vs_dim1', colors, N=256)
    
    plt.imshow(gate, cmap=cmap, vmin=0, vmax=1)
    plt.colorbar(label='Weight (0=dim1, 1=dim0)')
    plt.title('Gate Values: Which Dimension is Emphasized')
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', label='Dimension 0 (Birth-Death)'),
        Patch(facecolor='red', label='Dimension 1 (Loops)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}gate_values.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Gate visualization saved to: {save_path}")

def visualize_topo_features(topo_features, output_dir, prefix=''):
    """Visualize raw topological features."""
    if topo_features is None:
        print("No topological features available to visualize.")
        return
    
    # topo_features shape: [1, 2, H, W]
    topo = topo_features.squeeze().cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Visualize dimension 0
    plt.subplot(1, 2, 1)
    plt.imshow(topo[0], cmap='viridis')
    plt.colorbar()
    plt.title('Dimension 0 (Birth-Death)')
    
    # Visualize dimension 1
    plt.subplot(1, 2, 2)
    plt.imshow(topo[1], cmap='plasma')
    plt.colorbar()
    plt.title('Dimension 1 (Loops)')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}topo_features.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Topological features visualization saved to: {save_path}")

def visualize_input_and_prediction(image, topo_features, model, device, output_dir, class_name, prefix=''):
    """Visualize input image, prediction, and attention overlay."""
    if image is None:
        print("No image available to visualize.")
        return
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(image, topo_features)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Get class names and probabilities
    pred_idx = predicted.item()
    pred_prob = probs[0, pred_idx].item()
    
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    pred_name = class_names.get(pred_idx, f"Class {pred_idx}")
    
    # Convert image for visualization
    img = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)  # Ensure values are between 0 and 1
    
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Input Image\nTrue Class: {class_name}')
    plt.axis('off')
    
    # Prediction probabilities
    plt.subplot(1, 2, 2)
    classes = list(class_names.values())
    class_probs = [probs[0, i].item() for i in range(len(classes))]
    
    colors = ['green' if classes[i] == pred_name else 'gray' for i in range(len(classes))]
    plt.bar(classes, class_probs, color=colors)
    plt.ylim(0, 1)
    plt.title(f'Prediction: {pred_name} ({pred_prob:.2%})')
    plt.ylabel('Probability')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}input_and_prediction.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Input and prediction visualization saved to: {save_path}")

def visualize_activations(activations, output_dir, prefix=''):
    """Visualize different activation maps from the model."""
    if not activations:
        print("No activations available to visualize.")
        return
    
    # Visualize pyramid level activations
    pyramid_keys = [k for k in activations.keys() if 'pyramid_level' in k]
    if pyramid_keys:
        plt.figure(figsize=(15, 5 * len(pyramid_keys)))
        
        for i, key in enumerate(sorted(pyramid_keys)):
            # Get activation and compute average across channels
            act = activations[key].squeeze().cpu().numpy()
            avg_act = np.mean(act, axis=0)  # Average across channels
            
            plt.subplot(len(pyramid_keys), 1, i + 1)
            plt.imshow(avg_act, cmap='inferno')
            plt.colorbar()
            plt.title(f'Pyramid Level {key.split("_")[-1]} Activation')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{prefix}pyramid_activations.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Pyramid activations saved to: {save_path}")
    
    # Visualize grapher module outputs if available
    grapher_keys = [k for k in activations.keys() if 'grapher' in k]
    if grapher_keys and len(grapher_keys) > 0:
        # Show a sample of grapher outputs (the first few and last few)
        samples = []
        if len(grapher_keys) > 6:
            samples = grapher_keys[:3] + grapher_keys[-3:]
        else:
            samples = grapher_keys
            
        plt.figure(figsize=(15, 5 * len(samples)))
        
        for i, key in enumerate(sorted(samples)):
            # Get activation and compute average across channels
            act = activations[key].squeeze().cpu().numpy()
            avg_act = np.mean(act, axis=0)  # Average across channels
            
            plt.subplot(len(samples), 1, i + 1)
            plt.imshow(avg_act, cmap='viridis')
            plt.colorbar()
            plt.title(f'GCN Module {key} Output')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{prefix}gcn_activations.png')
        plt.savefig(save_path)
        plt.close()
        print(f"GCN module activations saved to: {save_path}")

def visualize_fused_topo(fused_topo, output_dir, prefix=''):
    """Visualize the fused topological features after gating/projection."""
    if fused_topo is None:
        print("No fused topological features available to visualize.")
        return
    
    # fused_topo shape might be [1, C, H, W] where C is the channel dimension
    fused = fused_topo.squeeze().cpu().numpy()
    
    # If there are multiple channels, compute the mean
    if len(fused.shape) == 3:
        fused = np.mean(fused, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(fused, cmap='inferno')
    plt.colorbar()
    plt.title('Fused Topological Features (After Projection)')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}fused_topo.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Fused topological features saved to: {save_path}")

def combine_visualizations(results_dir, prefix='', run_name=''):
    """Combine key visualizations into a summary image."""
    # Find the visualization files
    files = {
        'input': f'{prefix}input_and_prediction.png',
        'topo': f'{prefix}topo_features.png',
        'gate': f'{prefix}gate_values.png',
        'fused': f'{prefix}fused_topo.png',
        'pyramid': f'{prefix}pyramid_activations.png'
    }
    
    # Check which files exist
    existing = {}
    for key, filename in files.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            existing[key] = path
    
    if len(existing) <= 1:
        print("Not enough visualizations to create a combined view.")
        return
    
    # Load images using PIL
    images = {}
    for key, path in existing.items():
        try:
            # Load image and convert to RGB mode to ensure 3 channels (remove alpha if exists)
            img = Image.open(path).convert('RGB')
            images[key] = np.array(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not images:
        print("Failed to load any visualization images.")
        return
    
    # Create a grid layout based on what's available
    rows = []
    
    # Row 1: Input & Topo
    row1_imgs = []
    if 'input' in images:
        row1_imgs.append(images['input'])
    if 'topo' in images:
        row1_imgs.append(images['topo'])
    if row1_imgs:
        # Resize to same height
        height = min(img.shape[0] for img in row1_imgs)
        row1_imgs = [transform.resize(img, (height, int(img.shape[1] * height / img.shape[0]), 3), 
                                     preserve_range=True, anti_aliasing=True).astype(np.uint8) 
                    if img.shape[0] != height else img for img in row1_imgs]
        
        # Debug info
        for i, img in enumerate(row1_imgs):
            print(f"Row 1 image {i} shape: {img.shape}")
            
        rows.append(np.hstack(row1_imgs))
    
    # Row 2: Gate & Fused
    row2_imgs = []
    if 'gate' in images:
        row2_imgs.append(images['gate'])
    if 'fused' in images:
        row2_imgs.append(images['fused'])
    if row2_imgs:
        # Resize to same height
        height = min(img.shape[0] for img in row2_imgs)
        row2_imgs = [transform.resize(img, (height, int(img.shape[1] * height / img.shape[0]), 3), 
                                     preserve_range=True, anti_aliasing=True).astype(np.uint8)
                    if img.shape[0] != height else img for img in row2_imgs]
        
        # Debug info
        for i, img in enumerate(row2_imgs):
            print(f"Row 2 image {i} shape: {img.shape}")
            
        rows.append(np.hstack(row2_imgs))
    
    # Pyramid by itself if it exists
    if 'pyramid' in images:
        # Ensure it's also RGB
        pyramid_img = images['pyramid']
        # Add debug info
        print(f"Pyramid image shape: {pyramid_img.shape}")
        rows.append(pyramid_img)
    
    if not rows:
        print("No rows to combine.")
        return
    
    # Resize rows to the same width
    max_width = max(row.shape[1] for row in rows)
    resized_rows = []
    for row in rows:
        if row.shape[1] != max_width:
            h, w = row.shape[:2]
            aspect = h / w
            new_height = int(aspect * max_width)
            resized_row = transform.resize(row, (new_height, max_width, 3), 
                                         preserve_range=True, anti_aliasing=True).astype(np.uint8)
            resized_rows.append(resized_row)
        else:
            resized_rows.append(row)
    
    # Combine rows
    try:
        combined = np.vstack(resized_rows)
        
        # Add a title bar with experiment name
        title_bar_height = 50
        title_bar = np.ones((title_bar_height, combined.shape[1], 3), dtype=np.uint8) * 255
        
        # Add text to title bar
        title_img = Image.fromarray(title_bar)
        draw = ImageDraw.Draw(title_img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
            
        title = f"Attention Visualization - {run_name}"
        text_width = draw.textlength(title, font=font)
        x_pos = (title_bar.shape[1] - text_width) // 2
        draw.text((x_pos, 10), title, fill=(0, 0, 0), font=font)
        
        title_bar = np.array(title_img)
        combined = np.vstack([title_bar, combined])
        
        # Save the combined image
        save_path = os.path.join(results_dir, f'{prefix}combined_visualization.png')
        Image.fromarray(combined).save(save_path)
        print(f"Combined visualization saved to: {save_path}")
    except Exception as e:
        print(f"Error creating combined visualization: {e}")
        # Print more detailed error information
        import traceback
        traceback.print_exc()

def visualize_fusion_attention_heatmap(img_tensor, activations, output_dir, prefix=''):
    """Create and save fusion attention heatmaps for each pyramid level activation, overlaying on the input image."""
    # Convert input image to numpy (H, W, 3), normalized to [0, 1]
    img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    img_h, img_w = img.shape[:2]

    pyramid_keys = [k for k in activations.keys() if 'pyramid_level' in k]
    if not pyramid_keys:
        print("No pyramid level activations found for fusion attention heatmap.")
        return

    for key in sorted(pyramid_keys):
        act = activations[key].squeeze().cpu().numpy()
        # Average across channels to get a 2D map
        if act.ndim == 3:
            avg_act = np.mean(act, axis=0)
        else:
            avg_act = act
        # Normalize to [0, 1]
        avg_act -= avg_act.min()
        if avg_act.max() > 0:
            avg_act /= avg_act.max()
        # Resize to input image size
        heatmap = cv2.resize(avg_act, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
        # Apply 'jet' colormap (blue-green-red)
        heatmap_color = cm.get_cmap('jet')(heatmap)[:, :, :3]  # Drop alpha
        # Blend heatmap with input image
        overlay = (0.5 * img + 0.5 * heatmap_color)
        overlay = np.clip(overlay, 0, 1)
        # Plot and save
        plt.figure(figsize=(8, 6))
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f'Fusion Attention Heatmap - {key.replace("_", " ").title()} (Jet)')
        save_path = os.path.join(output_dir, f'{prefix}fusion_attention_{key}_jet.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Fusion attention heatmap (jet) saved to: {save_path}")

def create_master_summary(output_dir, processed_prefixes, run_name, prefix=''):
    """Create a master summary combining key visualizations from all classes."""
    # Files to combine for each class
    key_files = ['input_and_prediction.png', 'gate_values.png', 'topo_features.png']
    
    class_images = {}
    
    # Load images for each class
    for class_prefix in processed_prefixes:
        class_name = class_prefix.replace(prefix, '').replace('_', ' ').title().rstrip(' ')
        class_images[class_name] = {}
        
        for file_type in key_files:
            file_path = os.path.join(output_dir, f'{class_prefix}{file_type}')
            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path).convert('RGB')
                    class_images[class_name][file_type] = np.array(img)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    if not class_images:
        print("No images found to create master summary.")
        return
    
    # Create a grid layout: rows for classes, columns for visualization types
    class_names = list(class_images.keys())
    available_types = set()
    for images in class_images.values():
        available_types.update(images.keys())
    available_types = sorted(list(available_types))
    
    if not available_types:
        print("No visualization types available for master summary.")
        return
    
    # Create title mapping for cleaner display
    title_map = {
        'input_and_prediction.png': 'Input & Prediction',
        'gate_values.png': 'Gate Values',
        'topo_features.png': 'Topological Features'
    }
    
    # Calculate grid dimensions
    n_rows = len(class_names)
    n_cols = len(available_types)
    
    # Target size for each cell
    target_height = 300
    target_width = 400
    
    # Create master grid
    master_grid = []
    
    # Add title row
    title_height = 40
    title_row = np.ones((title_height, target_width * n_cols, 3), dtype=np.uint8) * 240
    
    # Add column titles
    for col, vis_type in enumerate(available_types):
        title_img = Image.fromarray(title_row[:, col*target_width:(col+1)*target_width])
        draw = ImageDraw.Draw(title_img)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        title_text = title_map.get(vis_type, vis_type.replace('_', ' ').title())
        text_width = draw.textlength(title_text, font=font)
        x_pos = (target_width - text_width) // 2
        draw.text((x_pos, 10), title_text, fill=(0, 0, 0), font=font)
        
        title_row[:, col*target_width:(col+1)*target_width] = np.array(title_img)
    
    master_grid.append(title_row)
    
    # Add rows for each class
    for row, class_name in enumerate(class_names):
        row_images = []
        
        for col, vis_type in enumerate(available_types):
            if vis_type in class_images[class_name]:
                # Resize image to target size
                img = class_images[class_name][vis_type]
                img_resized = transform.resize(img, (target_height, target_width, 3), 
                                             preserve_range=True, anti_aliasing=True).astype(np.uint8)
            else:
                # Create placeholder image
                img_resized = np.ones((target_height, target_width, 3), dtype=np.uint8) * 200
                
                # Add "Not Available" text
                placeholder_img = Image.fromarray(img_resized)
                draw = ImageDraw.Draw(placeholder_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except:
                    font = ImageFont.load_default()
                    
                text = "Not Available"
                text_width = draw.textlength(text, font=font)
                x_pos = (target_width - text_width) // 2
                y_pos = target_height // 2 - 10
                draw.text((x_pos, y_pos), text, fill=(100, 100, 100), font=font)
                img_resized = np.array(placeholder_img)
            
            row_images.append(img_resized)
        
        # Combine row images horizontally
        row_combined = np.hstack(row_images)
        
        # Add class label on the left
        label_width = 100
        label_img = np.ones((target_height, label_width, 3), dtype=np.uint8) * 220
        
        label_pil = Image.fromarray(label_img)
        draw = ImageDraw.Draw(label_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
        
        # Rotate text 90 degrees for vertical label
        text_img = Image.new('RGB', (200, 30), color=(220, 220, 220))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((10, 5), class_name, fill=(0, 0, 0), font=font)
        text_img = text_img.rotate(90, expand=True)
        
        # Paste rotated text onto label
        text_resized = text_img.resize((label_width-10, min(target_height-20, text_img.height)))
        label_pil.paste(text_resized, (5, (target_height - text_resized.height) // 2))
        
        label_img = np.array(label_pil)
        
        # Combine label with row
        row_with_label = np.hstack([label_img, row_combined])
        master_grid.append(row_with_label)
    
    # Combine all rows vertically
    try:
        master_image = np.vstack(master_grid)
        
        # Add main title
        main_title_height = 60
        main_title_img = np.ones((main_title_height, master_image.shape[1], 3), dtype=np.uint8) * 250
        
        title_pil = Image.fromarray(main_title_img)
        draw = ImageDraw.Draw(title_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        main_title = f"Attention Visualization Summary - {run_name}"
        text_width = draw.textlength(main_title, font=font)
        x_pos = (master_image.shape[1] - text_width) // 2
        draw.text((x_pos, 15), main_title, fill=(0, 0, 0), font=font)
        
        main_title_img = np.array(title_pil)
        
        # Final combined image
        final_image = np.vstack([main_title_img, master_image])
        
        # Save the master summary
        save_path = os.path.join(output_dir, f'{prefix}master_summary.png')
        Image.fromarray(final_image).save(save_path)
        print(f"Master summary saved to: {save_path}")
        
    except Exception as e:
        print(f"Error creating master summary: {e}")
        import traceback
        traceback.print_exc()

# --- Main Function ---

def process_single_sample(img_tensor, topo_tensor, class_name, model, device, output_dir, prefix=''):
    """Process a single sample and generate all visualizations."""
    print(f"\n--- Processing {class_name} sample ---")
    
    # Setup feature extractor with hooks
    extractor = FeatureExtractor(model)
    
    # Run inference to capture activations
    with torch.no_grad():
        outputs = model(img_tensor, topo_tensor)
        # Get softmax probabilities
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        pred_idx = predicted.item()
        pred_prob = probs[0, pred_idx].item()
        
        class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
        pred_name = class_names.get(pred_idx, f"Class {pred_idx}")
        
        print(f"Model predicted: {pred_name} with confidence: {pred_prob:.4f}")
        print(f"True class: {class_name}")
        
        # Bold indication if prediction is correct
        if pred_name == class_name:
            print("✓ CORRECT PREDICTION")
        else:
            print("✗ INCORRECT PREDICTION")
    
    # Get activations
    activations = extractor.activations
    
    # Generate visualizations with class-specific prefix
    class_prefix = f"{prefix}{class_name.lower().replace(' ', '_')}_"
    
    # Visualize input image and model prediction
    visualize_input_and_prediction(img_tensor, topo_tensor, model, device, output_dir, class_name, class_prefix)
    
    # Visualize topological features
    visualize_topo_features(topo_tensor, output_dir, class_prefix)
    
    # Visualize gate values if available (gated model)
    if 'gate' in activations:
        visualize_gate_values(activations['gate'], output_dir, class_prefix)
    
    # Visualize fused topological features if available
    if 'topo_fused' in activations:
        visualize_fused_topo(activations['topo_fused'], output_dir, class_prefix)
    
    # Visualize other activations
    visualize_activations(activations, output_dir, class_prefix)
    
    # Fusion attention heatmap visualization
    visualize_fusion_attention_heatmap(img_tensor, activations, output_dir, class_prefix)
    
    # Clean up hooks
    extractor.remove_hooks()
    
    return class_prefix

def main(args):
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Create output directory if specified
    output_dir = args.output_dir if args.output_dir else results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Get prefix for saved files
    prefix = args.prefix + '_' if args.prefix else ''
    
    # Load configuration
    config = load_config(results_dir)
    
    # Setup device
    device_req = config.get('device', 'auto')
    if device_req == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_req)
    print(f'Using device: {device}')
    
    # Load model
    model, epoch = load_model(config, results_dir, device)
    
    # Class names and indices
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    
    # Process samples for each class
    processed_prefixes = []
    
    if args.image_path:
        # Use provided image (single sample mode)
        img_tensor, _, class_name = prepare_custom_image(args.image_path, device, model)
        if img_tensor is None:
            print("Using random test sample instead.")
            img_tensor, topo_tensor, class_name = get_random_test_sample(model, config['data_dir'], device)
        else:
            # For custom images, we need to get topo features separately
            _, topo_tensor, _ = get_random_test_sample(model, config['data_dir'], device)
            print("Using topological features from a random test sample for custom image.")
            
            # Check if the custom image is correctly predicted
            with torch.no_grad():
                outputs = model(img_tensor, topo_tensor)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item()
                
                pred_name = class_names.get(predicted_label, f"Class {predicted_label}")
                
                # We don't know the true label for custom images, so just inform the user
                print(f"Custom image prediction: {pred_name} with confidence: {probs[0, predicted_label].item():.4f}")
        
        # Process the single sample
        sample_prefix = process_single_sample(img_tensor, topo_tensor, class_name, model, device, output_dir, prefix)
        processed_prefixes.append(sample_prefix)
        
    else:
        # Process one sample for each class
        print("=== Finding correctly predicted samples for all 3 classes ===")
        
        # First, analyze the test dataset to understand what's available
        class_counts, correct_predictions = analyze_test_dataset(model, config['data_dir'], device)
        
        if class_counts is None:
            print("Failed to analyze dataset. Proceeding with original approach...")
        else:
            print(f"\n=== Proceeding with sample selection ===")
        
        for class_idx, class_name in class_names.items():
            print(f"\n--- Looking for {class_name} sample ---")
            
            # Check if this class exists in the dataset
            if class_counts is not None and class_counts.get(class_idx, 0) == 0:
                print(f"❌ Skipping {class_name} - no samples found in test dataset")
                continue
            
            if args.allow_incorrect:
                # Original behavior - don't filter for correct predictions, just find any sample of this class
                _, test_loader = read_mimic(batchsize=1, data_dir=config['data_dir'])
                found = False
                for images, labels, topo_features in test_loader:
                    if labels[0].item() == class_idx:
                        img_tensor = images[0].unsqueeze(0).to(device)
                        topo_tensor = topo_features[0].unsqueeze(0).to(device)
                        print(f"Found {class_name} sample (prediction not verified)")
                        found = True
                        break
                
                if not found:
                    print(f"Warning: No {class_name} samples found in test set.")
                    continue
            else:
                # Try to find correctly predicted sample first
                result = get_random_test_sample(
                    model, config['data_dir'], device, target_class=class_idx, allow_incorrect=False
                )
                
                if result[0] is None:
                    # If no correctly predicted samples, ask user if they want incorrect ones
                    print(f"\n⚠️  No correctly predicted {class_name} samples found.")
                    
                    # Check if we have any samples of this class at all
                    if class_counts is not None and correct_predictions is not None:
                        total_class = class_counts.get(class_idx, 0)
                        correct_class = correct_predictions.get(class_idx, 0)
                        if total_class > 0:
                            print(f"   Found {total_class} total {class_name} samples, but only {correct_class} correctly predicted.")
                            print(f"   Trying to use an incorrectly predicted {class_name} sample instead...")
                            
                            # Try to get an incorrect sample
                            result = get_random_test_sample(
                                model, config['data_dir'], device, target_class=class_idx, allow_incorrect=True
                            )
                            
                            if result[0] is None:
                                print(f"   ❌ Failed to find any {class_name} sample. Skipping this class.")
                                continue
                        else:
                            print(f"   ❌ No {class_name} samples exist in test dataset. Skipping this class.")
                            continue
                    else:
                        # Fallback: try to get any sample of this class
                        print(f"   Trying to find any {class_name} sample...")
                        result = get_random_test_sample(
                            model, config['data_dir'], device, target_class=class_idx, allow_incorrect=True
                        )
                        
                        if result[0] is None:
                            print(f"   ❌ No {class_name} samples found. Skipping this class.")
                            continue
                
                img_tensor, topo_tensor, found_class_name = result
            
            # Process this sample
            sample_prefix = process_single_sample(img_tensor, topo_tensor, class_name, model, device, output_dir, prefix)
            processed_prefixes.append(sample_prefix)
    
    # Create combined visualizations for each processed sample
    run_name = os.path.basename(results_dir)
    
    for sample_prefix in processed_prefixes:
        print(f"\nCreating combined visualization for {sample_prefix}...")
        combine_visualizations(output_dir, sample_prefix, run_name)
    
    # Create a master summary combining key visualizations from all classes
    if len(processed_prefixes) > 1:
        print(f"\nCreating master summary with all {len(processed_prefixes)} classes...")
        create_master_summary(output_dir, processed_prefixes, run_name, prefix)
    
    print("\n=== Visualization complete for all samples! ===")
    print(f"Results saved in: {output_dir}")
    if len(processed_prefixes) > 1:
        print(f"Check {prefix}master_summary.png for overview of all classes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize model attention for TopoGNN models.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to experiment results directory')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Optional path to a specific input image')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to results_dir)')
    parser.add_argument('--prefix', type=str, default='',
                        help='Prefix for saved visualization files')
    parser.add_argument('--allow_incorrect', action='store_true',
                        help='Allow visualizing incorrectly predicted samples (not recommended)')
    
    args = parser.parse_args()
    main(args) 