#!/usr/bin/env python3
"""
Simplified attention visualization script that works around NumPy 2.x compatibility issues.
"""

import os
import torch
import numpy as np
import yaml
import argparse
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import importlib
import sys
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from pathlib import Path
import random
import matplotlib.cm as cm

# Ensure the script can find other modules in the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from utils.read_data_topo import read_mimic, Dataset
except ImportError as e:
    print(f"Error importing data utilities: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

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

def analyze_test_performance(model, data_dir, device, max_samples=200):
    """Analyze model performance on test set to understand class-wise accuracy."""
    print("=== Analyzing Model Performance on Test Set ===")
    
    try:
        _, test_loader = read_mimic(batchsize=1, data_dir=data_dir)
        
        class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
        class_counts = {0: 0, 1: 0, 2: 0}
        correct_predictions = {0: 0, 1: 0, 2: 0}
        confusion_matrix = {(i, j): 0 for i in range(3) for j in range(3)}
        
        model.eval()
        
        samples_processed = 0
        print(f"Analyzing up to {max_samples} test samples...")
        
        with torch.no_grad():
            for i, (images, labels, topo_features) in enumerate(test_loader):
                if samples_processed >= max_samples:
                    break
                    
                label = labels[0].item()
                class_counts[label] += 1
                
                img_tensor = images[0].unsqueeze(0).to(device)
                topo_tensor = topo_features[0].unsqueeze(0).to(device)
                
                outputs = model(img_tensor, topo_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item()
                
                confusion_matrix[(label, predicted_label)] += 1
                
                if predicted_label == label:
                    correct_predictions[label] += 1
                
                samples_processed += 1
                
                if samples_processed % 50 == 0:
                    print(f"Processed {samples_processed} samples...")
        
        print(f"\n=== Performance Analysis Results ===")
        print(f"Total samples analyzed: {samples_processed}")
        
        overall_correct = sum(correct_predictions.values())
        overall_accuracy = overall_correct / samples_processed * 100
        print(f"Overall accuracy: {overall_accuracy:.1f}%")
        
        print(f"\nClass-wise Performance:")
        for class_idx, class_name in class_names.items():
            count = class_counts[class_idx]
            correct = correct_predictions[class_idx]
            accuracy = (correct / count * 100) if count > 0 else 0
            
            print(f"{class_name}: {correct}/{count} correct ({accuracy:.1f}% accuracy)")
            
            if count == 0:
                print(f"  ⚠️  No {class_name} samples found in analyzed subset!")
            elif correct == 0:
                print(f"  ❌ No correctly predicted {class_name} samples found!")
            elif accuracy < 20:
                print(f"  ⚠️  Very poor performance on {class_name}")
        
        # Print confusion matrix
        print(f"\nConfusion Matrix:")
        print("True\\Pred\tNormal\tCHF\tpneumonia")
        for true_idx, true_name in class_names.items():
            row = f"{true_name}\t"
            for pred_idx in range(3):
                row += f"{confusion_matrix[(true_idx, pred_idx)]}\t"
            print(row)
        
        return class_counts, correct_predictions
        
    except Exception as e:
        print(f"Error analyzing performance: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def get_sample_for_class(model, data_dir, device, target_class, allow_incorrect=True):
    """Get a sample for the specified class, preferring correct predictions."""
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    target_name = class_names.get(target_class, f'Class {target_class}')
    
    print(f"\nLooking for {target_name} sample...")
    
    try:
        _, test_loader = read_mimic(batchsize=1, data_dir=data_dir)
        
        correct_sample = None
        any_sample = None
        attempts = 0
        
        model.eval()
        
        with torch.no_grad():
            for images, labels, topo_features in test_loader:
                label = labels[0].item()
                
                if label != target_class:
                    continue
                
                attempts += 1
                img_tensor = images[0].unsqueeze(0).to(device)
                topo_tensor = topo_features[0].unsqueeze(0).to(device)
                
                # Store as potential sample
                if any_sample is None:
                    any_sample = (img_tensor, topo_tensor, target_name)
                
                # Check if prediction is correct
                outputs = model(img_tensor, topo_tensor)
                _, predicted = torch.max(outputs, 1)
                predicted_label = predicted.item()
                
                if predicted_label == label:
                    print(f"✓ Found correctly predicted {target_name} sample")
                    return img_tensor, topo_tensor, target_name, True
                
                # Only try a reasonable number of samples
                if attempts >= 20:
                    break
        
        # If no correct prediction found
        if allow_incorrect and any_sample is not None:
            print(f"⚠️  No correctly predicted {target_name} found. Using incorrectly predicted sample.")
            return any_sample[0], any_sample[1], any_sample[2], False
        else:
            print(f"❌ No {target_name} samples found")
            return None, None, None, False
            
    except Exception as e:
        print(f"Error getting sample for {target_name}: {e}")
        return None, None, None, False

def visualize_gate_values(gate_values, output_dir, prefix=''):
    """Visualize gate values that weight dim0 vs dim1 features."""
    if gate_values is None:
        print("No gate values available to visualize.")
        return None
    
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gate visualization saved to: {save_path}")
    return save_path

def visualize_topo_features(topo_features, output_dir, prefix=''):
    """Visualize raw topological features."""
    if topo_features is None:
        print("No topological features available to visualize.")
        return None
    
    # topo_features shape: [1, 2, H, W]
    topo = topo_features.squeeze().cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Visualize dimension 0
    plt.subplot(1, 2, 1)
    plt.imshow(topo[0], cmap='viridis')
    plt.colorbar()
    plt.title('Dimension 0 (Birth-Death)')
    plt.axis('off')
    
    # Visualize dimension 1
    plt.subplot(1, 2, 2)
    plt.imshow(topo[1], cmap='plasma')
    plt.colorbar()
    plt.title('Dimension 1 (Loops)')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}topo_features.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Topological features visualization saved to: {save_path}")
    return save_path

def visualize_input_and_prediction(image, model, device, output_dir, class_name, prefix='', is_correct=True):
    """Visualize input image and prediction probabilities."""
    if image is None:
        print("No image available to visualize.")
        return None
    
    # Get model prediction
    with torch.no_grad():
        # We need topo features for prediction, so we'll pass None and handle it
        # For display purposes, we'll assume the prediction was already computed
        pass
    
    # Convert image for visualization
    img = image.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    
    plt.figure(figsize=(12, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Input Image\nTrue Class: {class_name}')
    plt.axis('off')
    
    # Status indicator
    plt.subplot(1, 2, 2)
    status_text = "✓ CORRECT PREDICTION" if is_correct else "✗ INCORRECT PREDICTION"
    color = 'green' if is_correct else 'red'
    plt.text(0.5, 0.5, status_text, ha='center', va='center', 
             transform=plt.gca().transAxes, fontsize=16, color=color, weight='bold')
    plt.text(0.5, 0.3, f'True Class: {class_name}', ha='center', va='center',
             transform=plt.gca().transAxes, fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}input_and_prediction.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Input and prediction visualization saved to: {save_path}")
    return save_path

def comprehensive_visualize_sample(img_tensor, topo_tensor, class_name, model, device, output_dir, prefix, is_correct):
    """Create comprehensive visualizations for a sample."""
    print(f"Creating comprehensive visualizations for {class_name} sample...")
    
    visualization_paths = []
    
    # Setup feature extractor
    extractor = FeatureExtractor(model)
    
    # Run inference to capture activations
    with torch.no_grad():
        outputs = model(img_tensor, topo_tensor)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    pred_idx = predicted.item()
    pred_prob = probs[0, pred_idx].item()
    
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    pred_name = class_names.get(pred_idx, f"Class {pred_idx}")
    
    print(f"  Model predicted: {pred_name} ({pred_prob:.1%})")
    print(f"  True class: {class_name}")
    print(f"  Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
    
    # Get activations
    activations = extractor.activations
    print(f"  Captured activations: {list(activations.keys())}")
    
    # 1. Input and prediction
    path = visualize_input_and_prediction(img_tensor, model, device, output_dir, class_name, prefix, is_correct)
    if path: visualization_paths.append(path)
    
    # 2. Topological features
    path = visualize_topo_features(topo_tensor, output_dir, prefix)
    if path: visualization_paths.append(path)
    
    # 3. Gate values (if available)
    if 'gate' in activations:
        path = visualize_gate_values(activations['gate'], output_dir, prefix)
        if path: visualization_paths.append(path)
    
    # 4. Fused topological features (if available)
    if 'topo_fused' in activations:
        path = visualize_fused_topo(activations['topo_fused'], output_dir, prefix)
        if path: visualization_paths.append(path)
    
    # 5. Pyramid level activations
    path = visualize_pyramid_activations(activations, output_dir, prefix)
    if path: visualization_paths.append(path)
    
    # 6. Fusion attention heatmaps (one for each pyramid level)
    paths = visualize_fusion_attention_heatmaps(img_tensor, activations, output_dir, prefix)
    visualization_paths.extend(paths)
    
    # 7. Grapher/GCN activations
    path = visualize_grapher_activations(activations, output_dir, prefix)
    if path: visualization_paths.append(path)
    
    # 8. Main comprehensive visualization
    path = create_enhanced_visualization(img_tensor, topo_tensor, class_name, model, device, output_dir, prefix, is_correct, pred_name, pred_prob, probs)
    if path: visualization_paths.append(path)
    
    # Clean up hooks
    extractor.remove_hooks()
    
    print(f"  Created {len(visualization_paths)} visualizations")
    return visualization_paths

def create_enhanced_visualization(img_tensor, topo_tensor, class_name, model, device, output_dir, prefix, is_correct, pred_name, pred_prob, probs):
    """Create the main enhanced visualization combining multiple elements."""
    
    # Convert image for visualization
    img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    
    # Create visualization with more subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Input image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title(f'Input Image\nTrue: {class_name}')
    axes[0, 0].axis('off')
    
    # Prediction probabilities
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    classes = list(class_names.values())
    class_probs = [probs[0, i].item() for i in range(len(classes))]
    
    colors = ['green' if classes[i] == pred_name else 'lightgray' for i in range(len(classes))]
    axes[0, 1].bar(classes, class_probs, color=colors)
    axes[0, 1].set_ylim(0, 1)
    prediction_status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
    axes[0, 1].set_title(f'Prediction: {pred_name} ({pred_prob:.1%})\n{prediction_status}')
    axes[0, 1].set_ylabel('Probability')
    
    # Confidence visualization
    axes[0, 2].pie(class_probs, labels=classes, autopct='%1.1f%%', startangle=90)
    axes[0, 2].set_title('Prediction Confidence')
    
    # Topological features if available
    if topo_tensor is not None:
        topo = topo_tensor.squeeze().cpu().numpy()
        if len(topo.shape) == 3 and topo.shape[0] >= 2:
            # Dimension 0
            im1 = axes[1, 0].imshow(topo[0], cmap='viridis')
            axes[1, 0].set_title('Topo Dim 0 (Birth-Death)')
            axes[1, 0].axis('off')
            plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
            
            # Dimension 1
            im2 = axes[1, 1].imshow(topo[1], cmap='plasma')
            axes[1, 1].set_title('Topo Dim 1 (Loops)')
            axes[1, 1].axis('off')
            plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            # Combined/difference view
            diff = np.abs(topo[0] - topo[1])
            im3 = axes[1, 2].imshow(diff, cmap='coolwarm')
            axes[1, 2].set_title('Dimension Difference\n|Dim0 - Dim1|')
            axes[1, 2].axis('off')
            plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        else:
            for ax in axes[1, :]:
                ax.text(0.5, 0.5, 'Topo features\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
    else:
        for ax in axes[1, :]:
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save
    save_path = os.path.join(output_dir, f'{prefix}comprehensive_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualization saved to: {save_path}")
    return save_path

class FeatureExtractor:
    """Extract intermediate activations from the model."""
    def __init__(self, model):
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
        self._register_hooks()
    
    def _register_hooks(self):
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
            print(f"Found pyramid levels: {pyramid_levels}")
            
            # Register hooks for each pyramid level
            for i, module in enumerate(self.model.backbone):
                def hook_fn(module, input, output, level_idx=i):
                    if hasattr(self.model, 'pyramid_levels') and level_idx in self.model.pyramid_levels:
                        level = self.model.pyramid_levels.index(level_idx)
                        self.activations[f'pyramid_level_{level}'] = output.detach()
                        
                self.hooks.append(module.register_forward_hook(hook_fn))
        
        # Additional hooks for grapher modules
        idx = 0
        for name, module in self.model.named_modules():
            if 'grapher.grapher' in name:
                def gcn_hook(module, input, output, idx=idx):
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

def create_master_summary(output_dir, all_visualization_data, run_name, prefix=''):
    """Create a master summary combining key visualizations from all classes."""
    print("Creating master summary...")
    
    # Files to combine for each class (prioritized list)
    key_files = ['input_and_prediction.png', 'topo_features.png', 'pyramid_activations.png', 'comprehensive_visualization.png']
    
    class_images = {}
    
    # Load images for each class
    for class_name, viz_paths, is_correct in all_visualization_data:
        class_images[class_name] = {}
        class_prefix = f"{class_name.lower()}_"
        
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
        return None
    
    # Create a grid layout: rows for classes, columns for visualization types
    class_names = list(class_images.keys())
    available_types = set()
    for images in class_images.values():
        available_types.update(images.keys())
    available_types = sorted(list(available_types))
    
    if not available_types:
        print("No visualization types available for master summary.")
        return None
    
    # Create title mapping for cleaner display
    title_map = {
        'input_and_prediction.png': 'Input & Prediction',
        'topo_features.png': 'Topological Features',
        'pyramid_activations.png': 'Pyramid Activations',
        'comprehensive_visualization.png': 'Comprehensive View'
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
    try:
        from PIL import ImageDraw, ImageFont
        
        for col, vis_type in enumerate(available_types):
            title_img = Image.fromarray(title_row[:, col*target_width:(col+1)*target_width])
            draw = ImageDraw.Draw(title_img)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            title_text = title_map.get(vis_type, vis_type.replace('_', ' ').title())
            if font:
                bbox = draw.textbbox((0, 0), title_text, font=font)
                text_width = bbox[2] - bbox[0]
                x_pos = max(5, (target_width - text_width) // 2)
                draw.text((x_pos, 10), title_text, fill=(0, 0, 0), font=font)
            
            title_row[:, col*target_width:(col+1)*target_width] = np.array(title_img)
    except ImportError:
        print("PIL not available for text rendering, continuing without titles...")
    
    master_grid.append(title_row)
    
    # Add rows for each class
    for row, class_name in enumerate(class_names):
        row_images = []
        
        for col, vis_type in enumerate(available_types):
            if vis_type in class_images[class_name]:
                # Resize image to target size
                img = class_images[class_name][vis_type]
                from skimage import transform
                img_resized = transform.resize(img, (target_height, target_width, 3), 
                                             preserve_range=True, anti_aliasing=True).astype(np.uint8)
            else:
                # Create placeholder image
                img_resized = np.ones((target_height, target_width, 3), dtype=np.uint8) * 200
                
                # Add "Not Available" text using matplotlib
                fig, ax = plt.subplots(figsize=(target_width/100, target_height/100))
                ax.text(0.5, 0.5, 'Not Available', ha='center', va='center',
                       transform=ax.transAxes, fontsize=14, color='gray')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
                
                # Convert matplotlib figure to numpy array
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                
                # Resize to target size
                from skimage import transform
                img_resized = transform.resize(buf, (target_height, target_width, 3), 
                                             preserve_range=True, anti_aliasing=True).astype(np.uint8)
            
            row_images.append(img_resized)
        
        # Combine row images horizontally
        row_combined = np.hstack(row_images)
        
        # Add class label on the left
        label_width = 120
        label_img = np.ones((target_height, label_width, 3), dtype=np.uint8) * 220
        
        # Add class label using matplotlib
        fig, ax = plt.subplots(figsize=(label_width/100, target_height/100))
        ax.text(0.5, 0.5, class_name, ha='center', va='center',
               transform=ax.transAxes, fontsize=16, weight='bold', rotation=90)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Convert to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        # Resize and use as label
        from skimage import transform
        label_resized = transform.resize(buf, (target_height, label_width, 3), 
                                       preserve_range=True, anti_aliasing=True).astype(np.uint8)
        
        # Combine label with row
        row_with_label = np.hstack([label_resized, row_combined])
        master_grid.append(row_with_label)
    
    # Combine all rows vertically
    try:
        master_image = np.vstack(master_grid)
        
        # Add main title
        main_title_height = 60
        main_title_img = np.ones((main_title_height, master_image.shape[1], 3), dtype=np.uint8) * 250
        
        # Add main title using matplotlib
        fig, ax = plt.subplots(figsize=(master_image.shape[1]/100, main_title_height/100))
        main_title = f"TopoGNN Attention Visualization Summary - {run_name}"
        ax.text(0.5, 0.5, main_title, ha='center', va='center',
               transform=ax.transAxes, fontsize=20, weight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        from skimage import transform
        main_title_resized = transform.resize(buf, (main_title_height, master_image.shape[1], 3), 
                                            preserve_range=True, anti_aliasing=True).astype(np.uint8)
        
        # Final combined image
        final_image = np.vstack([main_title_resized, master_image])
        
        # Save the master summary
        save_path = os.path.join(output_dir, f'{prefix}master_summary.png')
        Image.fromarray(final_image).save(save_path)
        print(f"Master summary saved to: {save_path}")
        return save_path
        
    except Exception as e:
        print(f"Error creating master summary: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_pyramid_activations(activations, output_dir, prefix=''):
    """Visualize pyramid level activations from the model backbone."""
    pyramid_keys = [k for k in activations.keys() if 'pyramid_level' in k]
    if not pyramid_keys:
        print("No pyramid level activations found.")
        return None
    
    print(f"Visualizing {len(pyramid_keys)} pyramid levels...")
    
    plt.figure(figsize=(15, 5 * len(pyramid_keys)))
    
    for i, key in enumerate(sorted(pyramid_keys)):
        # Get activation and compute average across channels
        act = activations[key].squeeze().cpu().numpy()
        if act.ndim == 3:
            avg_act = np.mean(act, axis=0)  # Average across channels
        else:
            avg_act = act
        
        plt.subplot(len(pyramid_keys), 1, i + 1)
        plt.imshow(avg_act, cmap='inferno')
        plt.colorbar()
        plt.title(f'Pyramid Level {key.split("_")[-1]} Activation')
        plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}pyramid_activations.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pyramid activations saved to: {save_path}")
    return save_path

def visualize_fusion_attention_heatmaps(img_tensor, activations, output_dir, prefix=''):
    """Create fusion attention heatmaps for each pyramid level activation, overlaying on the input image."""
    # Convert input image to numpy (H, W, 3), normalized to [0, 1]
    img = img_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    img_h, img_w = img.shape[:2]

    pyramid_keys = [k for k in activations.keys() if 'pyramid_level' in k]
    if not pyramid_keys:
        print("No pyramid level activations found for fusion attention heatmap.")
        return []
    
    print(f"Creating fusion attention heatmaps for {len(pyramid_keys)} pyramid levels...")
    
    saved_paths = []
    
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
        
        # Resize to input image size using skimage instead of cv2
        from skimage import transform
        heatmap = transform.resize(avg_act, (img_h, img_w), preserve_range=True, anti_aliasing=True)
        heatmap = np.clip(heatmap, 0, 1)
        
        # Apply 'jet' colormap (blue-green-red)
        heatmap_color = cm.get_cmap('jet')(heatmap)[:, :, :3]  # Drop alpha
        
        # Blend heatmap with input image
        overlay = (0.5 * img + 0.5 * heatmap_color)
        overlay = np.clip(overlay, 0, 1)
        
        # Create side-by-side visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Heatmap
        im1 = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title(f'Attention Map - {key.replace("_", " ").title()}')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title('Attention Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{prefix}fusion_attention_{key}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        saved_paths.append(save_path)
        print(f"Fusion attention heatmap saved to: {save_path}")
    
    return saved_paths

def visualize_grapher_activations(activations, output_dir, prefix=''):
    """Visualize GCN/Grapher module activations."""
    grapher_keys = [k for k in activations.keys() if 'grapher' in k]
    if not grapher_keys or len(grapher_keys) == 0:
        print("No grapher activations found.")
        return None
    
    # Show a sample of grapher outputs (the first few and last few)
    samples = []
    if len(grapher_keys) > 6:
        samples = grapher_keys[:3] + grapher_keys[-3:]
    else:
        samples = grapher_keys
    
    print(f"Visualizing {len(samples)} grapher modules...")
    
    plt.figure(figsize=(15, 5 * len(samples)))
    
    for i, key in enumerate(sorted(samples)):
        # Get activation and compute average across channels
        act = activations[key].squeeze().cpu().numpy()
        if act.ndim == 3:
            avg_act = np.mean(act, axis=0)  # Average across channels
        else:
            avg_act = act
        
        plt.subplot(len(samples), 1, i + 1)
        plt.imshow(avg_act, cmap='viridis')
        plt.colorbar()
        plt.title(f'GCN Module {key} Output')
        plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}gcn_activations.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"GCN module activations saved to: {save_path}")
    return save_path

def visualize_fused_topo(fused_topo, output_dir, prefix=''):
    """Visualize the fused topological features after gating/projection."""
    if fused_topo is None:
        print("No fused topological features available to visualize.")
        return None
    
    # fused_topo shape might be [1, C, H, W] where C is the channel dimension
    fused = fused_topo.squeeze().cpu().numpy()
    
    # If there are multiple channels, compute the mean
    if len(fused.shape) == 3:
        fused = np.mean(fused, axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(fused, cmap='inferno')
    plt.colorbar()
    plt.title('Fused Topological Features (After Projection)')
    plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{prefix}fused_topo.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Fused topological features saved to: {save_path}")
    return save_path

def main(args):
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = args.output_dir if args.output_dir else results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(results_dir)
    
    # Setup device
    device_req = config.get('device', 'auto')
    if device_req == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_req)
    print(f'Using device: {device}')
    
    # Load model
    model, epoch = load_model(config, results_dir, device)
    
    # Analyze model performance
    class_counts, correct_predictions = analyze_test_performance(model, config['data_dir'], device)
    
    # Process samples for each class
    class_names = {0: "Normal", 1: "CHF", 2: "pneumonia"}
    all_visualization_data = []
    
    print(f"\n=== Creating Comprehensive Visualizations ===")
    
    for class_idx, class_name in class_names.items():
        if class_counts and class_counts.get(class_idx, 0) == 0:
            print(f"\n❌ Skipping {class_name} - no samples in analyzed subset")
            continue
        
        # Get sample
        img_tensor, topo_tensor, found_class_name, is_correct = get_sample_for_class(
            model, config['data_dir'], device, class_idx, allow_incorrect=True
        )
        
        if img_tensor is not None:
            prefix = f"{class_name.lower()}_"
            viz_paths = comprehensive_visualize_sample(
                img_tensor, topo_tensor, class_name, model, device, 
                output_dir, prefix, is_correct
            )
            all_visualization_data.append((class_name, viz_paths, is_correct))
        else:
            print(f"❌ Could not create visualization for {class_name}")
    
    # Create master summary
    run_name = os.path.basename(results_dir)
    if len(all_visualization_data) > 1:
        master_path = create_master_summary(output_dir, all_visualization_data, run_name)
    
    # Summary
    print(f"\n=== Summary ===")
    print(f"Results saved in: {output_dir}")
    
    if all_visualization_data:
        print(f"\nCreated comprehensive visualizations:")
        for class_name, viz_paths, is_correct in all_visualization_data:
            status = "✓ Correct" if is_correct else "✗ Incorrect"
            print(f"  {class_name}: {len(viz_paths)} visualizations ({status})")
            for path in viz_paths:
                print(f"    - {os.path.basename(path)}")
    else:
        print("No visualizations could be created.")
    
    if len(all_visualization_data) > 1:
        print(f"\n🎯 Master Summary: master_summary.png")
    
    if class_counts and correct_predictions:
        print(f"\nModel Performance Summary:")
        for class_idx, class_name in class_names.items():
            count = class_counts.get(class_idx, 0)
            correct = correct_predictions.get(class_idx, 0)
            accuracy = (correct / count * 100) if count > 0 else 0
            print(f"  {class_name}: {accuracy:.1f}% accuracy ({correct}/{count})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Simplified attention visualization for TopoGNN models.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to experiment results directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations (defaults to results_dir)')
    
    args = parser.parse_args()
    main(args) 