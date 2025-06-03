import os
import torch
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import label_binarize
import importlib
from tqdm import tqdm
import sys
from itertools import cycle
import pandas as pd
import json
from torch import nn

# Ensure the script can find other modules in the project (e.g., models, utils)
# Assuming this script is in 'utils' and main.py is in the parent directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.read_data_topo import read_mimic # Corrected import

def load_config(results_dir, custom_config_path=None):
    """Load the configuration file from the results directory or a custom path."""
    if custom_config_path:
        config_path = custom_config_path
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Custom configuration file not found: {config_path}")
        print(f"Loading custom configuration from {config_path}")
    else:
        config_path = os.path.join(results_dir, 'config.yaml')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        print(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_loop(dataloader, model, loss_fn, device):
    """Exact same test_loop function as used during training"""
    # CRITICAL: Ensure model is in evaluation mode
    model.eval()
    print(f"Test loop - Model training mode: {model.training}")  # Should be False
    
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)), desc="Testing ")
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    gt = []
    pd_scores = [] # Renamed from pd to avoid conflict with pandas alias
    n_classes = -1 # Will be updated later

    with torch.no_grad():  # Disable gradient computation
        for batch, sample in enumerate(pbar):
            x, labels, topo_features = sample # Use topo_features
            x, labels, topo_features = x.to(device), labels.to(device), topo_features.to(device)

            outputs = model(x, topo_features) # Pass topo_features
            if n_classes == -1: # Get number of classes from first output
                n_classes = outputs.shape[1]

            loss = loss_fn(outputs, labels)
            _, pred = torch.max(outputs, 1)
            num_correct = (pred == labels).sum()

            loss_item = loss.item()
            acc_item = num_correct.item() / len(labels)
            count += len(labels)
            test_loss += loss_item * len(labels)
            test_acc += num_correct.item()
            gt.extend(labels.cpu().numpy())
            pd_scores.extend(outputs.cpu().numpy())

            pbar.set_description(f"loss: {loss_item:>f}, acc: {acc_item:>f} [{count:>d}/{size:>d}]")

    gt = np.array(gt)
    pd_scores = np.array(pd_scores)
    
    # --- AUC Calculation (EXACT same as training) ---
    fpr = dict()
    tpr = dict()
    roc_auc = []
    aucavg = 0.0
    
    if n_classes > 0:
        try:
            gt_bin = label_binarize(gt, classes=list(range(n_classes)))
            # Ensure gt_bin has n_classes columns even if some classes are missing in batch
            if gt_bin.shape[1] < n_classes:
                temp_bin = np.zeros((gt_bin.shape[0], n_classes))
                temp_bin[:, :gt_bin.shape[1]] = gt_bin
                gt_bin = temp_bin
                
            for i in range(n_classes):
                 # Check if class i is present in ground truth for AUC calculation
                if i < gt_bin.shape[1] and np.any(gt_bin[:, i]):
                    fpr[i], tpr[i], _ = roc_curve(gt_bin[:, i], pd_scores[:, i])
                    roc_auc.append(auc(fpr[i], tpr[i]))
                else:
                    print(f"Warning: Class {i} not present in test ground truth or binarization failed. AUC for this class set to NaN.")
                    roc_auc.append(float('nan')) # Assign NaN if class has no samples
            
            # Calculate average ignoring potential NaNs
            valid_aucs = [a for a in roc_auc if not np.isnan(a)]
            if valid_aucs:
                aucavg = np.mean(valid_aucs)
            else:
                 aucavg = 0.0 # Or handle as error if no valid classes
                 print("Warning: No valid AUC values found for any class.")
            print("AUC per class: {}".format([f'{a:.3f}' if not np.isnan(a) else 'NaN' for a in roc_auc]))
        except Exception as e:
            print(f"Error during AUC calculation: {e}")
            # Assign default values if AUC calculation fails
            roc_auc = [0.0] * n_classes
            aucavg = 0.0
    else:
        print("Warning: Could not determine number of classes from model output.")
        roc_auc = []
        aucavg = 0.0

    return test_loss/count, test_acc/count, aucavg, gt, pd_scores, roc_auc

def load_model(config, results_dir, device, model_type='best_acc'):
    """Load the model definition and the specified checkpoint state.
    
    Args:
        config: Configuration dictionary
        results_dir: Directory containing model checkpoints
        device: Device to load model to
        model_type: Type of model to load ('best_acc', 'best_auc', 'final')
    """
    model_type_name = config.get('model_type', 'pvig_ti')
    model_mode = config.get('model_mode', 'proj')
    num_classes = config['num_classes']
    drop_path_rate = config.get('drop_path_rate', 0.0)
    # Pretrained flag shouldn't matter here as we load specific weights
    pretrained_flag = False 

    model_func_name = f"{model_type_name}_224_gelu_{model_mode}"
    module_name = f"models.pvig_topo_{model_mode}"
    
    print(f"Loading model function '{model_func_name}' from module '{module_name}'...")
    try:
        model_module = importlib.import_module(module_name)
        model_func = getattr(model_module, model_func_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not load model function '{model_func_name}' from {module_name}. {e}")
        sys.exit(1)
        
    model = model_func(num_classes=num_classes, drop_path_rate=drop_path_rate, pretrained=pretrained_flag)
    
    # Try to load the summary.json to find best model checkpoint
    summary_path = os.path.join(results_dir, 'summary.json')
    best_model_info = None
    
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                
            # Extract best model info from summary
            if model_type == 'best_acc':
                best_epoch = summary.get('best_epoch_acc')
                best_metric = summary.get('best_acc')
                best_model_file = 'best_acc_model.pth'
                print(f"Using best accuracy model from epoch {best_epoch} (Acc: {best_metric:.4f})")
            elif model_type == 'best_auc':
                best_epoch = summary.get('best_epoch_auc')
                best_metric = summary.get('best_auc')
                best_model_file = 'best_auc_model.pth'
                print(f"Using best AUC model from epoch {best_epoch} (AUC: {best_metric:.4f})")
            else:  # final
                best_epoch = summary.get('final_epoch')
                best_model_file = 'final_model.pth'
                print(f"Using final model from epoch {best_epoch}")
                
            best_model_info = {
                'epoch': best_epoch,
                'file': best_model_file
            }
        except Exception as e:
            print(f"Warning: Could not extract best model info from summary.json: {e}")
    
    # Determine which checkpoint to load
    if best_model_info:
        checkpoint_path = os.path.join(results_dir, best_model_info['file'])
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Best model file {checkpoint_path} not found.")
            best_model_info = None
    
    # Fallback logic if no best model info or file not found
    if not best_model_info:
        checkpoint_candidates = [
            ('best_acc_model.pth', 'best accuracy'),
            ('best_auc_model.pth', 'best AUC'),
            ('final_model.pth', 'final')
        ]
        
        for checkpoint_file, desc in checkpoint_candidates:
            checkpoint_path = os.path.join(results_dir, checkpoint_file)
            if os.path.exists(checkpoint_path):
                print(f"Using {desc} model checkpoint: {checkpoint_file}")
                break
        else:
            raise FileNotFoundError(f"No model checkpoint found in {results_dir}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first
    
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
    
    # CRITICAL: Explicitly set model to evaluation mode
    model.eval()
    print("Model loaded successfully and set to EVALUATION mode.")
    
    # Debug: Check if model is actually in eval mode
    print(f"Model training mode: {model.training}")  # Should be False
    
    # Debug: Check dropout and batchnorm layers
    dropout_layers = []
    batchnorm_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_layers.append(name)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            batchnorm_layers.append(name)
    
    if dropout_layers:
        print(f"Found {len(dropout_layers)} Dropout layers: {dropout_layers[:3]}{'...' if len(dropout_layers) > 3 else ''}")
    if batchnorm_layers:
        print(f"Found {len(batchnorm_layers)} BatchNorm layers: {batchnorm_layers[:3]}{'...' if len(batchnorm_layers) > 3 else ''}")
    
    return model

def plot_roc_curves(labels, raw_scores, n_classes, class_names, save_path):
    """Calculate ROC/AUC and generate the plot using raw logits (to match training)."""
    # Binarize the labels
    labels_bin = label_binarize(labels, classes=list(range(n_classes)))
    if labels_bin.shape[1] < n_classes:
        print(f"Warning: Binarized labels have shape {labels_bin.shape}, expected ({len(labels)}, {n_classes}). Padding with zeros.")
        temp_bin = np.zeros((labels_bin.shape[0], n_classes))
        present_classes = labels_bin.shape[1]
        temp_bin[:,:present_classes] = labels_bin # Pad only the existing classes
        labels_bin = temp_bin
        
    # Compute ROC curve and ROC area for each class using RAW SCORES (like training)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    valid_classes = 0
    for i in range(n_classes):
        # Check if class i has positive samples in the binarized labels
        if i < labels_bin.shape[1] and np.any(labels_bin[:, i]):
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], raw_scores[:, i])  # Use raw scores
            roc_auc[i] = auc(fpr[i], tpr[i])
            valid_classes += 1
        else:
            print(f"Warning: No positive samples found for class {i} in test set. Skipping ROC calculation.")
            fpr[i], tpr[i] = None, None # Indicate missing curve
            roc_auc[i] = float('nan')
            
    if valid_classes == 0:
        print("Error: No valid classes found to compute ROC curves. Exiting plot generation.")
        return {}

    # Compute macro-average ROC curve and ROC area (same as training)
    valid_aucs = [roc_auc[i] for i in range(n_classes) if not np.isnan(roc_auc[i])]
    if valid_aucs:
        # Calculate macro average like in training
        macro_auc = np.mean(valid_aucs)
        roc_auc["macro"] = macro_auc
        
        # Also compute interpolated macro curve for plotting
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if fpr[i] is not None]))
        mean_tpr = np.zeros_like(all_fpr)
        count = 0
        for i in range(n_classes):
            if fpr[i] is not None:
                 mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                 count += 1
                 
        if count > 0:
            mean_tpr /= count
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
    else:
        roc_auc["macro"] = 0.0
        fpr["macro"] = None
        tpr["macro"] = None

    # Plot all ROC curves
    plt.figure(figsize=(8, 6))
    lw = 2 # line width

    # Plot Macro Average First
    if fpr["macro"] is not None:
        plt.plot(fpr["macro"], tpr["macro"],
                 label='Macro-average ROC (AUC = {0:0.3f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

    # Plot Individual Classes
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        class_name = class_names.get(i, f'Class {i}')
        if fpr[i] is not None:
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label=f'ROC {class_name} (AUC = {roc_auc[i]:0.3f})')
        else:
             # Optionally add a note if a class is missing
             plt.plot([], [], linestyle='', label=f'ROC {class_name} (No samples)')
             
    plt.plot([0, 1], [0, 1], 'k--', lw=lw) # Diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"ROC curve plot saved to {save_path}")
    # plt.show() # Uncomment to display plot directly
    plt.close()
    
    # Return AUC values for reporting
    return roc_auc

def calculate_metrics(true_labels, prob_scores, n_classes, class_name_map, threshold=0.5, use_custom_threshold=False):
    """Calculate precision, recall, and F1-score for each class using probability scores for classification."""
    if use_custom_threshold:
        # Apply custom threshold to determine class predictions
        # For multi-class: if no class exceeds threshold, pick the highest scoring class
        pred_labels = []
        for scores in prob_scores:
            if np.any(scores >= threshold):
                # If any class exceeds threshold, choose the highest among those
                classes_above_threshold = np.where(scores >= threshold)[0]
                pred_labels.append(classes_above_threshold[np.argmax(scores[classes_above_threshold])])
            else:
                # If no class exceeds threshold, pick the highest scoring class
                pred_labels.append(np.argmax(scores))
        pred_labels = np.array(pred_labels)
        print(f"Using custom threshold: {threshold}")
    else:
        # Standard approach: just take the highest scoring class
        pred_labels = np.argmax(prob_scores, axis=1)
        print("Using standard argmax classification (no threshold)")
    
    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, labels=range(n_classes), average=None
    )
    
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # Calculate metrics with macro average
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    # Prepare metrics dictionary
    metrics = {
        'Class': [class_name_map.get(i, f"Class {i}") for i in range(n_classes)] + ['Macro Average'],
        'Precision': list(precision) + [macro_precision],
        'Recall': list(recall) + [macro_recall],
        'F1-Score': list(f1) + [macro_f1],
        'Support': list(support) + [sum(support)]
    }
    
    # Add accuracy as a separate metric
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    # Create confusion matrix data
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=range(n_classes))
    
    return metrics, accuracy, cm

def save_metrics_to_csv(metrics, save_path):
    """Save metrics to a CSV file."""
    df = pd.DataFrame(metrics)
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to {save_path}")

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def main(args):
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
        
    # Load configuration
    config = load_config(results_dir, args.config_file)
    
    # --- Device Setup --- (EXACT same as training)
    device_req = config.get('device', 'auto')
    if device_req == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_req)
    print(f'Using device: {device}')
    
    # --- Load Data --- (EXACT same as training)
    print("Loading test data...")
    try:
        train_loader, test_loader = read_mimic(
            batchsize=config['batch_size'], 
            data_dir=config['data_dir']
            # EXACT same call as in training - no num_workers parameter modification
        )
    except ImportError as e:
        print(f"Error importing data loader: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)
    print("Test data loaded successfully.")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Test batch size: {test_loader.batch_size}")
    
    # --- Load Model ---
    model = load_model(config, results_dir, device, model_type=args.model_type)
    
    # EXTRA SAFETY: Ensure model is definitely in eval mode before inference
    model.eval()
    print(f"Final check - Model training mode: {model.training}")  # Should be False
    
    # --- Create Loss Function (same as training) ---
    criterion = nn.CrossEntropyLoss().to(device)
    
    # --- Run Exact Same Test Loop as Training ---
    print("\nRunning inference using EXACT same test_loop as training...")
    test_loss, test_acc, test_auc, gt_labels, raw_scores, individual_aucs = test_loop(test_loader, model, criterion, device)
    
    print(f"\nTest Results (using training's test_loop):")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")
    print(f"Test AUC:  {test_auc:.4f}")
    
    n_classes = config.get('num_classes', raw_scores.shape[1])
    
    # --- Define Class Names ---
    class_name_map = {
        0: "CHF",      # Make sure this matches your actual dataset!
        1: "Normal", 
        2: "pneumonia" 
    }
    # Ensure the map covers the expected number of classes
    if len(class_name_map) != n_classes:
        print(f"Warning: Defined class_name_map has {len(class_name_map)} entries, but expected {n_classes} classes. Using default names.")
        class_name_map = {i: f"Class {i}" for i in range(n_classes)}
    
    class_names = [class_name_map.get(i, f"Class {i}") for i in range(n_classes)]

    # --- Plot ROC (using raw scores) --- 
    plot_save_path = os.path.join(results_dir, f'{args.model_type}_roc_auc_plot.png')
    roc_auc_values = plot_roc_curves(gt_labels, raw_scores, n_classes, class_name_map, plot_save_path)
    
    # --- Calculate Classification Metrics (using probabilities) ---
    prob_scores = torch.softmax(torch.from_numpy(raw_scores), dim=1).numpy()
    metrics, accuracy, confusion_mat = calculate_metrics(
        gt_labels, prob_scores, n_classes, class_name_map, 
        threshold=args.threshold, 
        use_custom_threshold=args.use_threshold
    )
    
    # Save metrics
    metrics_prefix = f"{args.model_type}_"
    if args.use_threshold:
        metrics_prefix += f"thresh_{args.threshold}_"
        
    metrics_save_path = os.path.join(results_dir, f'{metrics_prefix}classification_metrics.csv')
    save_metrics_to_csv(metrics, metrics_save_path)
    
    # Save results to text file
    with open(os.path.join(results_dir, f'{metrics_prefix}results.txt'), 'w') as f:
        f.write(f"Model Type: {args.model_type}\n")
        f.write(f"Model Training Mode: {model.training}\n")  # Debug info
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test AUC (from test_loop): {test_auc:.4f}\n")
        f.write(f"Test AUC (from plot): {roc_auc_values.get('macro', 0.0):.4f}\n")
        f.write(f"Macro Precision: {metrics['Precision'][-1]:.4f}\n")
        f.write(f"Macro Recall: {metrics['Recall'][-1]:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['F1-Score'][-1]:.4f}\n")
        f.write(f"Threshold: {args.threshold if args.use_threshold else 'N/A (using argmax)'}\n")
        
        # Add per-class AUC values
        f.write("\nPer-Class AUC (from individual_aucs):\n")
        for i in range(n_classes):
            class_name = class_name_map.get(i, f"Class {i}")
            auc_value = individual_aucs[i] if i < len(individual_aucs) else float('nan')
            f.write(f"{class_name}: {auc_value:.4f}\n")
    
    # Plot and save confusion matrix
    cm_save_path = os.path.join(results_dir, f'{metrics_prefix}confusion_matrix.png')
    plot_confusion_matrix(confusion_mat, class_names, cm_save_path)
    
    # Print comparison with training summary
    print("\n" + "="*50)
    print(f"FINAL COMPARISON WITH TRAINING SUMMARY")
    print("="*50)
    
    summary_path = os.path.join(results_dir, 'summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                training_summary = json.load(f)
            
            if args.model_type == 'best_acc':
                training_acc = training_summary.get('best_acc', 0.0)
                training_auc = training_summary.get('best_auc', 0.0)  # Note: might be from different epoch
                print(f"Training Best Acc:     {training_acc:.4f}")
                print(f"Inference Accuracy:    {test_acc:.4f}")
                print(f"Accuracy Difference:   {abs(training_acc - test_acc):.4f}")
                print(f"Training Best AUC:     {training_auc:.4f}")
                print(f"Inference AUC:         {test_auc:.4f}")
                print(f"AUC Difference:        {abs(training_auc - test_auc):.4f}")
            elif args.model_type == 'best_auc':
                training_auc = training_summary.get('best_auc', 0.0)
                training_acc = training_summary.get('best_acc', 0.0)  # Note: might be from different epoch
                print(f"Training Best AUC:     {training_auc:.4f}")
                print(f"Inference AUC:         {test_auc:.4f}")
                print(f"AUC Difference:        {abs(training_auc - test_auc):.4f}")
                print(f"Training Best Acc:     {training_acc:.4f}")
                print(f"Inference Accuracy:    {test_acc:.4f}")
                print(f"Accuracy Difference:   {abs(training_acc - test_acc):.4f}")
            elif args.model_type == 'final':
                training_acc = training_summary.get('final_test_acc', 0.0)
                training_auc = training_summary.get('final_test_auc', 0.0)
                print(f"Training Final Acc:    {training_acc:.4f}")
                print(f"Inference Accuracy:    {test_acc:.4f}")
                print(f"Accuracy Difference:   {abs(training_acc - test_acc):.4f}")
                print(f"Training Final AUC:    {training_auc:.4f}")
                print(f"Inference AUC:         {test_auc:.4f}")
                print(f"AUC Difference:        {abs(training_auc - test_auc):.4f}")
            
        except Exception as e:
            print(f"Could not load training summary for comparison: {e}")
    else:
        print("No training summary found for comparison")
    
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ROC curves and calculate metrics from training results.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the experiment results directory (e.g., results/pvig_topo_dim0_ti_run1)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold (default: 0.5)')
    parser.add_argument('--use_threshold', action='store_true',
                        help='Use custom threshold for classification instead of argmax')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Path to a custom config.yaml file to use instead of the one in results_dir')
    parser.add_argument('--model_type', type=str, default='best_acc', choices=['best_acc', 'best_auc', 'final'],
                        help='Which model checkpoint to use for evaluation (default: best_acc)')
    
    args = parser.parse_args()
    main(args) 