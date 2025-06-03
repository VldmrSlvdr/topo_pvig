import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import timm
from tqdm import tqdm
import yaml
import json
import csv
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import random
import matplotlib.pyplot as plt
from utils.read_data_topo import read_mimic
from models.transformer import create_model

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def train_loop(dataloader, model, loss_fn, optimizer, device):
    """Train for one epoch"""
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)), desc="Training")
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    
    for batch, sample in enumerate(pbar):
        x_image, labels, topo_features = sample
        x_image, labels = x_image.to(device), labels.to(device)
        if topo_features is not None:
            topo_features = topo_features.to(device)
        
        optimizer.zero_grad()
        outputs = model(x_image, topo_features=topo_features if model.module.use_topo_features else None)
        loss = loss_fn(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, predicted = outputs.max(1)
        num_correct = (predicted == labels).sum()
        
        loss_item = loss.item()
        acc_item = num_correct.item() / len(labels)
        count += len(labels)
        train_loss += loss_item * len(labels)
        train_acc += num_correct.item()
        
        pbar.set_description(f"loss: {loss_item:>f}, acc: {acc_item:>f} [{count:>d}/{size:>d}]")
    
    return train_loss/count, train_acc/count

def test_loop(dataloader, model, loss_fn, device):
    """Validate the model"""
    model.eval()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)), desc="Validation")
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x_image, labels, topo_features = sample
            x_image, labels = x_image.to(device), labels.to(device)
            if topo_features is not None:
                topo_features = topo_features.to(device)
            
            outputs = model(x_image, topo_features=topo_features if model.module.use_topo_features else None)
            loss = loss_fn(outputs, labels)
            
            _, predicted = outputs.max(1)
            num_correct = (predicted == labels).sum()
            
            loss_item = loss.item()
            acc_item = num_correct.item() / len(labels)
            count += len(labels)
            test_loss += loss_item * len(labels)
            test_acc += num_correct.item()
            
            all_preds.extend(F.softmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_description(f"loss: {loss_item:>f}, acc: {acc_item:>f} [{count:>d}/{size:>d}]")
    
    # Calculate metrics
    accuracy = 100.*test_acc/count
    avg_loss = test_loss/count
    
    # Calculate AUC
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        np.argmax(all_preds, axis=1), 
        average='weighted'
    )
    
    return avg_loss, accuracy, auc, precision, recall, f1

def save_checkpoint(model, optimizer, epoch, metrics, save_path, is_best=False, metric_name='acc'):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    # Save regular checkpoint
    torch.save(checkpoint, save_path)
    
    # Save best model if needed
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), f'best_{metric_name}_model.pth')
        torch.save(checkpoint, best_path)

def plot_metrics(log_train_loss, log_train_acc, log_test_loss, log_test_acc, log_auc, n_epochs, save_dir):
    """Plot training metrics"""
    plt.figure(figsize=(15, 5))
    
    # Loss plot
    plt.subplot(131)
    plt.plot(np.arange(1, n_epochs + 1), log_train_loss, label='Train')
    plt.plot(np.arange(1, n_epochs + 1), log_test_loss, label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Accuracy plot
    plt.subplot(132)
    plt.plot(np.arange(1, n_epochs + 1), log_train_acc, label='Train')
    plt.plot(np.arange(1, n_epochs + 1), log_test_acc, label='Test')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    # AUC plot
    plt.subplot(133)
    plt.plot(np.arange(1, n_epochs + 1), log_auc)
    plt.title('AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def main(config):
    # Set random seed
    set_seed(config['seed'])
    
    # --- Set TORCH_HOME for pretrained models ---
    project_root = os.getcwd() # Or specify your project root if this script is run from a subfolder
    pretrain_dir = os.path.join(project_root, 'pretrain')
    os.makedirs(pretrain_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = pretrain_dir
    print(f"Pretrained models will be downloaded to/loaded from: {pretrain_dir}")
    # --------------------------------------------

    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create output directory
    output_dir = os.path.join(config['output_base_dir'], config['experiment_name'])
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Setup logging
    writer = SummaryWriter(output_dir)
    csv_file = os.path.join(output_dir, 'training_log.csv')
    with open(csv_file, 'w', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1'])
    
    # Load data
    train_loader, test_loader = read_mimic(
        batchsize=config['batch_size'],
        data_dir=config['data_dir'],
        num_workers=config.get('num_workers', 4)
    )
    
    # Create model
    model = create_model(config)
    
    # Move model to device
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    
    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    
    # Setup scheduler if specified
    scheduler = None
    if config.get('scheduler'):
        if config['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['scheduler_params']['step_size'],
                gamma=config['scheduler_params']['gamma']
            )
        elif config['scheduler'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['epochs'],
                eta_min=config['scheduler_params'].get('eta_min', 0)
            )
    
    # Training loop
    best_acc = 0
    best_auc = 0
    best_epoch_acc = 0
    best_epoch_auc = 0
    
    # Logging lists
    log_train_loss = []
    log_train_acc = []
    log_test_loss = []
    log_test_acc = []
    log_auc = []
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_loop(train_loader, model, criterion, optimizer, device)
        
        # Validate
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1 = test_loop(
            test_loader, model, criterion, device
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/test', test_acc, epoch)
        writer.add_scalar('AUC/test', test_auc, epoch)
        writer.add_scalar('Precision/test', test_precision, epoch)
        writer.add_scalar('Recall/test', test_recall, epoch)
        writer.add_scalar('F1/test', test_f1, epoch)
        
        # Store metrics for plotting
        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)
        log_test_loss.append(test_loss)
        log_test_acc.append(test_acc)
        log_auc.append(test_auc)
        
        # Save checkpoint
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        # Save regular checkpoint
        if (epoch + 1) % config['checkpoint_interval_epochs'] == 0:
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            )
        
        # Save best model based on accuracy
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch_acc = epoch
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(checkpoint_dir, 'best_acc_model.pth'),
                is_best=True, metric_name='acc'
            )
        
        # Save best model based on AUC
        if test_auc > best_auc:
            best_auc = test_auc
            best_epoch_auc = epoch
            save_checkpoint(
                model, optimizer, epoch, metrics,
                os.path.join(checkpoint_dir, 'best_auc_model.pth'),
                is_best=True, metric_name='auc'
            )
        
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, Test AUC: {test_auc:.4f}')
        print(f'Best Acc: {best_acc:.2f}% (Epoch {best_epoch_acc+1})')
        print(f'Best AUC: {best_auc:.4f} (Epoch {best_epoch_auc+1})')
        
        # Log to CSV
        with open(csv_file, 'a', newline='') as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([epoch+1, train_loss, train_acc, test_loss, test_acc, test_auc, test_precision, test_recall, test_f1])
    
    # Plot final metrics
    plot_metrics(log_train_loss, log_train_acc, log_test_loss, log_test_acc, log_auc, config['epochs'], output_dir)
    
    # Save final summary
    summary = {
        'best_epoch_acc': best_epoch_acc + 1,
        'best_acc': best_acc,
        'best_epoch_auc': best_epoch_auc + 1,
        'best_auc': best_auc,
        'final_metrics': metrics
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    writer.close()
    print("\nTraining completed!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Vision Transformer or Swin Transformer models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config) 