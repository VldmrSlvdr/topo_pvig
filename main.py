import os
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import pdb
import json
import csv
import yaml
import argparse
import time
import sys
import importlib
from models.transformer import create_model as create_transformer_model

# --- Utility Functions (Adapted from engine/train.py) ---

def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)), desc="Training")
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    
    # Debug variables for first epoch
    all_labels_debug = []
    all_predictions_debug = []
    all_outputs_debug = []
    
    for batch, sample in enumerate(pbar):
        x_image, labels, topo_features = sample  # ALIGNED with train_transformer.py
        x_image, labels = x_image.to(device), labels.to(device)
        if topo_features is not None:
            topo_features = topo_features.to(device)
        
        # Shape validation
        if batch == 0:
            print(f"Input shapes - Images: {x_image.shape}, Labels: {labels.shape}, Topo: {topo_features.shape if topo_features is not None else None}")
        
        # ALIGNED with train_transformer.py structure
        actual_model = model.module if isinstance(model, nn.DataParallel) else model
        use_topo = hasattr(actual_model, 'use_topo_features') and actual_model.use_topo_features
        
        optimizer.zero_grad()
        
        try:
            # EXACT same interface as train_transformer.py
            outputs = model(x_image, topo_features=topo_features if use_topo else None)
            
            # Validate output shape
            if batch == 0:
                print(f"Model output shape: {outputs.shape}")
                expected_shape = (x_image.shape[0], actual_model.num_classes if hasattr(actual_model, 'num_classes') else 3)
                print(f"Expected shape: {expected_shape}")
                if outputs.shape != expected_shape:
                    raise ValueError(f"Model output shape {outputs.shape} doesn't match expected {expected_shape}")
            
        except Exception as e:
            print(f"Error during model forward pass:")
            print(f"  Input image shape: {x_image.shape}")
            print(f"  Input topo shape: {topo_features.shape if topo_features is not None else None}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Error: {e}")
            raise
        
        # Ensure labels are in the correct format for CrossEntropyLoss
        if len(labels.shape) > 1:
            labels = labels.squeeze()
        if labels.dtype != torch.long:
            labels = labels.long()
        
        # Validate shapes before loss calculation
        if batch == 0:
            print(f"Loss calculation - Outputs: {outputs.shape}, Labels: {labels.shape}")
            print(f"Label range: min={labels.min().item()}, max={labels.max().item()}")
            print(f"Output logits range: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}")
            
        try:
            loss = loss_fn(outputs, labels)
        except Exception as e:
            print(f"Error during loss calculation:")
            print(f"  Outputs shape: {outputs.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Labels content: {labels[:5] if len(labels) > 0 else 'empty'}")
            print(f"  Error: {e}")
            raise
        
        loss.backward()
        optimizer.step()
        
        # ALIGNED with train_transformer.py - use .max(1) not torch.max
        _, predicted = outputs.max(1)
        num_correct = (predicted == labels).sum()
        
        # Collect debug info for first few batches
        if batch < 3:  # First 3 batches for analysis
            all_labels_debug.extend(labels.cpu().numpy().tolist())
            all_predictions_debug.extend(predicted.cpu().numpy().tolist())
            all_outputs_debug.extend(outputs.detach().cpu().numpy())

        loss_item = loss.item()
        acc_item = num_correct.item() / len(labels)
        count += len(labels)
        train_loss += loss_item * len(labels)
        train_acc += num_correct.item()
        pbar.set_description(f"loss: {loss_item:>f}, acc: {acc_item:>f} [{count:>d}/{size:>d}]")

    # Print debug analysis after first epoch
    if len(all_labels_debug) > 0:
        import numpy as np
        labels_array = np.array(all_labels_debug)
        predictions_array = np.array(all_predictions_debug)
        outputs_array = np.array(all_outputs_debug)
        
        print(f"\n=== TRAINING DEBUG ANALYSIS (first {len(all_labels_debug)} samples) ===")
        print(f"Label distribution: {np.bincount(labels_array, minlength=3)}")
        print(f"Prediction distribution: {np.bincount(predictions_array, minlength=3)}")
        print(f"Labels: {labels_array[:20]}")  # First 20 labels
        print(f"Predictions: {predictions_array[:20]}")  # First 20 predictions
        print(f"Sample output logits (first 5 samples):")
        for i in range(min(5, len(outputs_array))):
            print(f"  Sample {i}: logits={outputs_array[i]}, softmax={torch.softmax(torch.tensor(outputs_array[i]), dim=0).numpy()}")
        
        # Check if model is predicting same class for everything
        unique_preds = np.unique(predictions_array)
        if len(unique_preds) == 1:
            print(f"WARNING: Model is predicting only class {unique_preds[0]} for all samples!")
        
        print("=== END DEBUG ANALYSIS ===\n")

    return train_loss/count, train_acc/count

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)), desc="Testing")
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    all_labels_list = []
    all_outputs_list = []

    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x_image, labels, topo_features = sample  # ALIGNED with train_transformer.py
            x_image, labels = x_image.to(device), labels.to(device)
            if topo_features is not None:
                topo_features = topo_features.to(device)

            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            use_topo = hasattr(actual_model, 'use_topo_features') and actual_model.use_topo_features

            try:
                # EXACT same interface as train_transformer.py
                outputs = model(x_image, topo_features=topo_features if use_topo else None)
            except Exception as e:
                print(f"Error during model forward pass in test:")
                print(f"  Input image shape: {x_image.shape}")
                print(f"  Input topo shape: {topo_features.shape if topo_features is not None else None}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Error: {e}")
                raise
            
            # Ensure labels are in the correct format for CrossEntropyLoss
            if len(labels.shape) > 1:
                labels = labels.squeeze()
            if labels.dtype != torch.long:
                labels = labels.long()
                
            try:
                loss = loss_fn(outputs, labels)
            except Exception as e:
                print(f"Error during loss calculation in test:")
                print(f"  Outputs shape: {outputs.shape}")
                print(f"  Labels shape: {labels.shape}")
                print(f"  Labels dtype: {labels.dtype}")
                print(f"  Error: {e}")
                raise
                
            # ALIGNED with train_transformer.py - use .max(1) not torch.max
            _, predicted = outputs.max(1)
            num_correct = (predicted == labels).sum()
            
            loss_item = loss.item()
            acc_item = num_correct.item() / len(labels)
            count += len(labels)
            test_loss += loss_item * len(labels)
            test_acc += num_correct.item()
            
            all_labels_list.extend(labels.cpu().numpy())
            all_outputs_list.extend(torch.softmax(outputs, dim=1).cpu().numpy())

            pbar.set_description(f"loss: {loss_item:>f}, acc: {acc_item:>f} [{count:>d}/{size:>d}]")

    avg_test_loss = test_loss / count
    avg_test_acc = (test_acc / count) * 100.0 # Accuracy as percentage

    gt_np = np.array(all_labels_list)
    pd_np = np.array(all_outputs_list)
    
    # Calculate AUC (OvR for multi-class)
    if config.get('num_classes', 3) > 2 :
        gt_binarized = label_binarize(gt_np, classes=list(range(config.get('num_classes', 3))))
        auc_score = roc_auc_score(gt_binarized, pd_np, multi_class='ovr', average='weighted')
    else: # Binary classification case
        auc_score = roc_auc_score(gt_np, pd_np[:, 1]) # Assuming positive class is at index 1

    # Calculate Precision, Recall, F1 (weighted average)
    predicted_classes = np.argmax(pd_np, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_np, predicted_classes, average='weighted', zero_division=0
    )
    
    print(f"  Raw Test Metrics - Loss: {avg_test_loss:.4f}, Acc: {avg_test_acc:.2f}%, AUC: {auc_score:.4f}")
    return avg_test_loss, avg_test_acc, auc_score, precision, recall, f1

def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False, best_type=None):
    """Save checkpoint with metrics and training state"""
    # Handle DataParallel wrapper if present
    model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    state = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(state, save_path)
    if is_best:
        print(f"---> Saved {best_type}-best model checkpoint (epoch {epoch}) to {save_path}")

# --- Main Execution --- 

def main(config):
    
    # --- Reproducibility ---
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Keep benchmark True unless seed needs full guarantee
    # torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = True 
    print(f"Set seed: {seed}")

    # --- Device Setup ---
    device_req = config.get('device', 'auto')
    if device_req == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_req)
    print(f'Using device: {device}')

    # --- Output Dir Setup ---
    output_dir = os.path.join(config.get('output_base_dir', 'results'), config.get('experiment_name', 'default_run'))
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Save config for reproducibility
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved configuration to {config_save_path}")

    # --- Logging Setup ---
    writer = SummaryWriter(output_dir)
    csv_file = os.path.join(output_dir, 'training_log.csv')
    with open(csv_file, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1'])
    print(f"Logging CSV to {csv_file}")

    # --- Data Loading ---
    print("Loading data...")
    try:
        # Use the working data loader structure
        from utils.read_data_topo import read_mimic
        train_loader, test_loader = read_mimic(
            batchsize=config['batch_size'], 
            data_dir=config['data_dir']
        )
        
        # Alternative: Try the original working data loader if available
        # from utils.read_data import read_mimic_topo
        # train_loader, test_loader = read_mimic_topo(
        #     batchsize=config['batch_size'], 
        #     data_dir=config['data_dir']
        # )
        
    except ImportError:
        print("Error: Could not import read_mimic from utils.read_data_topo.")
        print("Ensure 'utils' directory is in your PYTHONPATH or accessible.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data loading: {e}")
        sys.exit(1)
    print("Data loaded successfully.")

    # --- Data Distribution Analysis ---
    print("\n=== DATASET ANALYSIS ===")
    try:
        # Check training data distribution - FIXED: Sample randomly instead of first N
        train_labels = []
        train_dataset = train_loader.dataset
        total_train_samples = len(train_dataset)
        
        # Sample 200 random indices instead of first 200
        from random import sample  # Import specific function to avoid conflict
        sample_indices = sample(range(total_train_samples), min(200, total_train_samples))
        
        for i in sample_indices:
            _, label, _ = train_dataset[i]
            train_labels.append(label)
        
        train_label_counts = np.bincount(train_labels, minlength=3)
        print(f"Training data distribution (random {len(train_labels)} samples):")
        print(f"  CHF (0): {train_label_counts[0]} ({train_label_counts[0]/len(train_labels)*100:.1f}%)")
        print(f"  Normal (1): {train_label_counts[1]} ({train_label_counts[1]/len(train_labels)*100:.1f}%)")
        print(f"  Pneumonia (2): {train_label_counts[2]} ({train_label_counts[2]/len(train_labels)*100:.1f}%)")
        
        # Check test data distribution - FIXED: Sample randomly 
        test_labels = []
        test_dataset = test_loader.dataset
        total_test_samples = len(test_dataset)
        
        test_sample_indices = sample(range(total_test_samples), min(50, total_test_samples))
        
        for i in test_sample_indices:
            _, label, _ = test_dataset[i]
            test_labels.append(label)
        
        test_label_counts = np.bincount(test_labels, minlength=3)
        print(f"Test data distribution (random {len(test_labels)} samples):")
        print(f"  CHF (0): {test_label_counts[0]} ({test_label_counts[0]/len(test_labels)*100:.1f}%)")
        print(f"  Normal (1): {test_label_counts[1]} ({test_label_counts[1]/len(test_labels)*100:.1f}%)")
        print(f"  Pneumonia (2): {test_label_counts[2]} ({test_label_counts[2]/len(test_labels)*100:.1f}%)")
        
        # Check for severe class imbalance
        min_class_ratio = min(train_label_counts) / sum(train_label_counts)
        if min_class_ratio < 0.1:  # Less than 10% for any class
            print(f"WARNING: Severe class imbalance detected! Minimum class ratio: {min_class_ratio:.3f}")
            print("Consider using weighted loss or class balancing techniques.")
        
    except Exception as e:
        print(f"Error during dataset analysis: {e}")
    print("=== END DATASET ANALYSIS ===\n")

    # --- Model Selection & Loading ---
    model_type = config.get('model_type', 'pvig_ti')
    num_classes = config['num_classes']
    drop_path_rate = config.get('drop_path_rate', 0.0)
    pretrained = config.get('pretrained', False)

    # --- Set TORCH_HOME for pretrained models (also relevant for timm) ---
    project_root = os.getcwd() 
    pretrain_dir = os.path.join(project_root, 'pretrain')
    os.makedirs(pretrain_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = pretrain_dir
    print(f"Pretrained models will be downloaded to/loaded from: {pretrain_dir}")
    # --- End TORCH_HOME setup ---

    if model_type.startswith(('vit_', 'swin_', 'deit_', 'efficientnet_', 'convnext_', 'resnet', 'resnext', 'densenet')): # Add other timm model prefixes if needed
        print(f"Attempting to load TIMM-based transformer model: {model_type} using models.transformer.create_model")
        # The create_transformer_model function expects the whole config
        # It handles 'model_type', 'pretrained', 'num_classes', 'drop_path_rate', 
        # 'img_size', 'use_topo_features', and 'topo_features_config' internally.
        try:
            model = create_transformer_model(config)
            model_func_name = model_type # For logging purposes
            print(f"TIMM-based model {model_type} loaded successfully via create_transformer_model.")
        except Exception as e:
            print(f"Error loading TIMM-based model {model_type} with create_transformer_model: {e}")
            sys.exit(1)
    else:
        # Existing PVIG and other custom model loading logic
        model_mode = config.get('model_mode', 'proj') # Default to proj if not specified
        use_custom_pretrained = config.get('use_custom_pretrained', False)
        custom_pretrained_path = config.get('custom_pretrained_path', None)
        
        model_func_name = f"{model_type}_224_gelu_{model_mode}" # Construct function name
        module_name = f"models.pvig_topo_{model_mode}" # Construct module name
        
        print(f"Attempting to load model function '{model_func_name}' from module '{module_name}'...")
        
        try:
            model_module = importlib.import_module(module_name)
            model_func = getattr(model_module, model_func_name)
        except (ImportError, AttributeError) as e:
            print(f"Error: Could not load model function '{model_func_name}' from {module_name}. {e}")
            print("Make sure the model file and function exist and names match the config.")
            sys.exit(1)
            
        pretrained_file = None
        if pretrained and not use_custom_pretrained:
            if model_type == 'pvig_ti': base_name = "pvig_ti_78.5.pth.tar"
            elif model_type == 'pvig_s': base_name = "pvig_s_82.1.pth.tar"
            elif model_type == 'pvig_m': base_name = "pvig_m_83.1.pth.tar"
            elif model_type == 'pvig_b': base_name = "pvig_b_83.66.pth.tar"
            else: base_name = None
            
            if base_name:
                # pretrained_path_base is now effectively controlled by TORCH_HOME for timm, 
                # but PVIG might have its own pretrain dir.
                # For PVIG, let's assume 'pretrained_path_base' from config is still relevant if specified
                # otherwise use the TORCH_HOME based 'pretrain' dir.
                pvig_pretrain_base = config.get('pretrained_path_base', pretrain_dir)
                pretrained_file = os.path.join(pvig_pretrain_base, base_name)
                print(f"Looking for PVIG pretrained weights at: {pretrained_file}")
                if not os.path.exists(pretrained_file):
                    print(f"Warning: PVIG Pretrained file not found. Model will be initialized randomly.")
                    # Do not set pretrained to False here, model_func might handle it or timm's own pretrained might kick in
                    # pretrained_file = None # Keep it None if not found
            else:
                print(f"Warning: Pretrained set to true, but no known base weight file for PVIG model_type '{model_type}'.")
        
        if use_custom_pretrained and custom_pretrained_path:
            if os.path.exists(custom_pretrained_path):
                print(f"Using custom pretrained weights from: {custom_pretrained_path}")
            else:
                print(f"Warning: Custom pretrained file not found at {custom_pretrained_path}")
                print("Falling back to standard pretrained weights or random initialization.")
                use_custom_pretrained = False
                custom_pretrained_path = None
        
        try:
            model = model_func(
                num_classes=num_classes, 
                drop_path_rate=drop_path_rate, 
                pretrained=pretrained, # Pass the original pretrained flag
                pretrained_path=pretrained_file, # Pass the constructed path
                custom_pretrained_path=custom_pretrained_path,
                use_custom_pretrained=use_custom_pretrained
            )
            print(f"PVIG model {model_func_name} loaded.")
        except TypeError as e:
            print(f"Warning: Model function {model_func_name} doesn't accept all parameters. Trying with basic parameters. Error: {e}")
            try:
                model = model_func(num_classes=num_classes, drop_path_rate=drop_path_rate, pretrained=pretrained)
                print(f"PVIG model {model_func_name} loaded with basic parameters.")
            except Exception as e2:
                print(f"Error loading model {model_func_name} with basic parameters: {e2}")
                sys.exit(1)
    
    model = model.to(device)

    print(f"Model '{model_func_name}' processed successfully.")
    print("There are", sum(p.numel() for p in model.parameters()), "parameters.")
    print("There are", sum(p.numel() for p in model.parameters() if p.requires_grad), "trainable parameters.")
    
    # Handle DataParallel for multi-GPU
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
      print(f"Using {torch.cuda.device_count()} GPUs (DataParallel).")
      model = torch.nn.DataParallel(model)

    # --- Loss & Optimizer ---
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer_name = config.get('optimizer', 'AdamW').lower()
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"Using Adam optimizer with LR: {lr}, weight_decay: {weight_decay}")
    elif optimizer_name == 'sgd':
         # Example SGD setup (matches commented out section in train.py)
         weight_p, bias_p = [],[]
         for name, p in model.named_parameters():
             if 'bias' in name:
                 bias_p += [p]
             else:
                 weight_p += [p]
         optimizer = torch.optim.SGD([{'params': weight_p, 'weight_decay': weight_decay if weight_decay > 0 else 1e-4}, # Add default decay for SGD weights if 0
                                      {'params': bias_p, 'weight_decay': 0}], lr=lr, momentum=0.9)
    else:
        print(f"Error: Optimizer '{optimizer_name}' not supported. Using AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    print(f"Using optimizer: {type(optimizer).__name__} with LR: {lr}")
    
    # --- Model Sanity Check ---
    print("\n=== MODEL SANITY CHECK ===")
    try:
        model.eval()
        with torch.no_grad():
            # Get a small batch for testing
            sample_batch = next(iter(train_loader))
            x_image, labels, topo_features = sample_batch  # ALIGNED with train_transformer.py
            x_image = x_image[:4].to(device)  # Use only 4 samples
            labels = labels[:4].to(device)
            
            if topo_features is not None:
                topo_features = topo_features[:4].to(device)
            
            # Forward pass - EXACT same as train_transformer.py
            actual_model = model.module if isinstance(model, nn.DataParallel) else model
            use_topo = hasattr(actual_model, 'use_topo_features') and actual_model.use_topo_features
            
            outputs = model(x_image, topo_features=topo_features if use_topo else None)
            
            print(f"Sample model outputs shape: {outputs.shape}")
            print(f"Sample labels: {labels.cpu().numpy()}")
            print(f"Sample predictions: {torch.argmax(outputs, dim=1).cpu().numpy()}")
            print(f"Sample output logits:")
            for i in range(outputs.shape[0]):
                logits = outputs[i].cpu().numpy()
                probs = torch.softmax(outputs[i], dim=0).cpu().numpy()
                print(f"  Sample {i}: logits={logits}, probs={probs}")
            
            # Check classifier weights
            if hasattr(actual_model, 'classifier'):
                classifier_weights = actual_model.classifier.weight.data
                classifier_bias = actual_model.classifier.bias.data if actual_model.classifier.bias is not None else None
                print(f"Classifier weight shape: {classifier_weights.shape}")
                print(f"Classifier weight range: [{classifier_weights.min().item():.4f}, {classifier_weights.max().item():.4f}]")
                if classifier_bias is not None:
                    print(f"Classifier bias: {classifier_bias.cpu().numpy()}")
                
                # Check if weights are very small (might indicate initialization issues)
                weight_magnitude = torch.norm(classifier_weights).item()
                print(f"Classifier weight magnitude: {weight_magnitude:.4f}")
                if weight_magnitude < 0.1:
                    print("WARNING: Classifier weights are very small, might indicate initialization issues!")
            
        model.train()  # Set back to training mode
    except Exception as e:
        print(f"Error during model sanity check: {e}")
        model.train()  # Ensure we're back in training mode
    print("=== END MODEL SANITY CHECK ===\n")
    
    # --- Training Loop ---
    start_time = time.time()
    best_loss = float('inf')
    best_acc = 0.0
    best_auc = 0.0
    best_epoch_loss = 0
    best_epoch_acc = 0
    best_epoch_auc = 0
    
    # Store metrics for best models
    best_acc_metrics = {}
    best_auc_metrics = {}
    final_metrics = {}
    
    epochs = config['epochs']
    checkpoint_interval = config.get('checkpoint_interval_epochs', 10)
    
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # print(f"===> Epoch {epoch}/{epochs}, learning rate: {optimizer.param_groups[0]['lr']:.1e}")
        print(f"===> Epoch {epoch}/{epochs}")
        
        train_loss, train_acc = train_loop(train_loader, model, criterion, optimizer, device)
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1 = test_loop(test_loader, model, criterion, device)
        
        epoch_duration = time.time() - epoch_start_time
        
        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.4f}, Test AUC: {test_auc:.4f}")
        print(f"  Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
        print(f"  Duration:   {epoch_duration:.2f}s")
        
        # --- Logging ---
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)
        writer.add_scalar("AUC/Test", test_auc, epoch)
        writer.add_scalar("Precision/Test", test_precision, epoch)
        writer.add_scalar("Recall/Test", test_recall, epoch)
        writer.add_scalar("F1/Test", test_f1, epoch)
        # writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
        
        with open(csv_file, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch, train_loss, train_acc, test_loss, test_acc, test_auc, test_precision, test_recall, test_f1])
            
        # Optional: Scheduler step
        # if scheduler: scheduler.step()
        
        # --- Checkpointing ---
        metrics = {
            'train_loss': train_loss, 'train_acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_auc': test_auc,
            'test_precision': test_precision, 'test_recall': test_recall, 'test_f1': test_f1
        }
        
        # Periodic checkpoint
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)
            print(f"Saved periodic checkpoint to {checkpoint_path}")
            
        # Best model checkpoints
        is_best_loss = test_loss < best_loss
        is_best_acc = test_acc > best_acc
        is_best_auc = test_auc > best_auc
        
        if is_best_loss:
            best_loss = test_loss
            best_epoch_loss = epoch
            save_checkpoint(model, optimizer, epoch, metrics, os.path.join(output_dir, 'best_loss_model.pth'), True, 'loss')
        
        if is_best_acc:
            best_acc = test_acc
            best_epoch_acc = epoch
            best_acc_metrics = {
                'epoch': epoch,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }
            save_checkpoint(model, optimizer, epoch, metrics, os.path.join(output_dir, 'best_acc_model.pth'), True, 'acc')
        
        if is_best_auc:
            best_auc = test_auc
            best_epoch_auc = epoch
            best_auc_metrics = {
                'epoch': epoch,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }
            save_checkpoint(model, optimizer, epoch, metrics, os.path.join(output_dir, 'best_auc_model.pth'), True, 'auc')
            
        # Store final epoch metrics
        if epoch == epochs:
            final_metrics = {
                'epoch': epoch,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            }
            
        print("-" * 50) # Separator

    # --- Final Summary --- 
    total_training_time = time.time() - start_time
    print("========================= TRAINING COMPLETE ========================")
    print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/3600:.2f} hours)")
    print(f"Best Loss Model (Epoch {best_epoch_loss}): Test Loss: {best_loss:.4f}")
    print(f"Best Accuracy Model (Epoch {best_epoch_acc}): Test Accuracy: {best_acc:.4f}")
    print(f"Best AUC Model (Epoch {best_epoch_auc}): Test AUC: {best_auc:.4f}")
    
    # Enhanced final metrics with detailed information for each best model
    enhanced_final_metrics = {
        'best_epoch_loss': best_epoch_loss,
        'best_loss': best_loss,
        'best_epoch_acc': best_epoch_acc,
        'best_acc': best_acc,
        'best_epoch_auc': best_epoch_auc,
        'best_auc': best_auc,
        'final_epoch': epochs,
        'final_test_loss': final_metrics['test_loss'],
        'final_test_acc': final_metrics['test_acc'],
        'final_test_auc': final_metrics['test_auc'],
        'final_test_precision': final_metrics['test_precision'],
        'final_test_recall': final_metrics['test_recall'],
        'final_test_f1': final_metrics['test_f1'],
        'total_training_time_sec': total_training_time,
        
        # Best accuracy model detailed metrics
        'best_acc_model': {
            'epoch': best_acc_metrics.get('epoch', best_epoch_acc),
            'test_loss': best_acc_metrics.get('test_loss', 0.0),
            'test_acc': best_acc_metrics.get('test_acc', best_acc),
            'test_auc': best_acc_metrics.get('test_auc', 0.0),
            'test_precision': best_acc_metrics.get('test_precision', 0.0),
            'test_recall': best_acc_metrics.get('test_recall', 0.0),
            'test_f1': best_acc_metrics.get('test_f1', 0.0)
        },
        
        # Best AUC model detailed metrics
        'best_auc_model': {
            'epoch': best_auc_metrics.get('epoch', best_epoch_auc),
            'test_loss': best_auc_metrics.get('test_loss', 0.0),
            'test_acc': best_auc_metrics.get('test_acc', 0.0),
            'test_auc': best_auc_metrics.get('test_auc', best_auc),
            'test_precision': best_auc_metrics.get('test_precision', 0.0),
            'test_recall': best_auc_metrics.get('test_recall', 0.0),
            'test_f1': best_auc_metrics.get('test_f1', 0.0)
        },
        
        # Final model detailed metrics
        'final_model': final_metrics
    }
    
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(enhanced_final_metrics, f, indent=4)
    print(f"Saved enhanced final summary to {os.path.join(output_dir, 'summary.json')}")
        
    # Save final model state
    save_checkpoint(model, optimizer, epochs, metrics, os.path.join(output_dir, 'final_model.pth'))
    print(f"Saved final model checkpoint to {os.path.join(output_dir, 'final_model.pth')}")
    writer.close()
    print("==================================================================")

# --- Argument Parser --- 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train PVIG model with Topological Features.')
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the YAML configuration file (e.g., configs/train_config.yaml)')
    
    args = parser.parse_args()
    
    # Load config from YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
        
    # Run main training function
    main(config)
