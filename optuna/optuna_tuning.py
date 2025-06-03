import os
import torch
import random
import numpy as np
import optuna
from optuna.trial import TrialState
import yaml
import argparse
import importlib
import time
from datetime import datetime
import json
import sys
from pathlib import Path

# Add parent directory to Python path to import from main project
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Import existing functionality from parent directory
from main import train_loop, test_loop, save_checkpoint

def define_search_space(trial, base_config):
    """Define the hyperparameter search space for Optuna"""
    params = {}
    
    # Learning rate - log scale
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # Weight decay - log scale
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    
    # Optimizer
    params['optimizer'] = trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'SGD'])
    
    # LR Scheduler
    params['scheduler'] = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'OneCycleLR', None])
    
    # Scheduler-specific parameters
    if params['scheduler'] == 'StepLR':
        params['scheduler_params'] = {
            'step_size': trial.suggest_int('step_size', 5, 30),
            'gamma': trial.suggest_float('gamma', 0.1, 0.9)
        }
    elif params['scheduler'] == 'CosineAnnealingLR':
        params['scheduler_params'] = {
            'T_max': base_config['epochs'],  # Use full training cycle
            'eta_min': trial.suggest_float('eta_min', 1e-7, 1e-5, log=True)
        }
    elif params['scheduler'] == 'OneCycleLR':
        params['scheduler_params'] = {
            'max_lr': params['learning_rate'] * trial.suggest_float('max_lr_factor', 5, 20),
            'pct_start': trial.suggest_float('pct_start', 0.1, 0.4),
            'anneal_strategy': 'cos',
            'total_steps': None  # Will be set in training code
        }
    else:
        params['scheduler_params'] = None
    
    # Batch size
    params['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    
    # Dropout and regularization
    params['drop_path_rate'] = trial.suggest_float('drop_path_rate', 0.0, 0.3)
    
    return params

def objective(trial, base_config):
    """Optuna objective function for hyperparameter optimization"""
    # Create a copy of the base config
    config = base_config.copy()
    
    # Update with trial-suggested parameters
    params = define_search_space(trial, base_config)
    config.update(params)
    
    # Create a unique experiment name for this trial
    config['experiment_name'] = f"{base_config['experiment_name']}_optuna_trial_{trial.number}"
    
    # Set up paths
    output_dir = os.path.join(config.get('output_base_dir', 'results'), config['experiment_name'])
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save trial config
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    
    # Set device
    device_req = config.get('device', 'auto')
    if device_req == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else 
                             "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(device_req)
    
    # Load data
    try:
        from utils.read_data_topo import read_mimic
        train_loader, test_loader = read_mimic(
            batchsize=config['batch_size'],
            data_dir=config['data_dir'],
            num_workers=config.get('num_workers', 0)
        )
    except ImportError as e:
        print(f"Error importing data loader: {e}")
        print("Make sure the parent directory is accessible for imports")
        raise
    
    # Set OneCycleLR total_steps if used
    if config['scheduler'] == 'OneCycleLR' and config['scheduler_params']:
        config['scheduler_params']['total_steps'] = len(train_loader) * config['epochs']
    
    # Load model
    model_type = config.get('model_type', 'pvig_ti')
    num_classes = config['num_classes']
    drop_path_rate = config.get('drop_path_rate', 0.0)
    pretrained = config.get('pretrained', False)
    
    # Handle TIMM-based transformer models (updated from main.py logic)
    if model_type.startswith(('vit_', 'swin_', 'deit_', 'efficientnet_', 'convnext_', 'resnet', 'resnext', 'densenet')):
        print(f"Loading TIMM-based transformer model: {model_type}")
        try:
            from models.transformer import create_model as create_transformer_model
            model = create_transformer_model(config)
        except Exception as e:
            print(f"Error loading TIMM-based model {model_type}: {e}")
            return 0.0
    else:
        # Handle PVIG and other custom models
        model_mode = config.get('model_mode', 'proj')
        model_func_name = f"{model_type}_224_gelu_{model_mode}"
        module_name = f"models.pvig_topo_{model_mode}"
        
        try:
            model_module = importlib.import_module(module_name)
            model_func = getattr(model_module, model_func_name)
        except (ImportError, AttributeError) as e:
            print(f"Error loading model: {e}")
            return 0.0
        
        # Handle pretrained weights for PVIG models
        pretrained_file = None
        if pretrained:
            if model_type == 'pvig_ti': base_name = "pvig_ti_78.5.pth.tar"
            elif model_type == 'pvig_s': base_name = "pvig_s_82.1.pth.tar"
            elif model_type == 'pvig_m': base_name = "pvig_m_83.1.pth.tar"
            elif model_type == 'pvig_b': base_name = "pvig_b_83.66.pth.tar"
            else: base_name = None
            
            if base_name:
                # Use parent directory for pretrain path
                pretrain_base = config.get('pretrained_path_base', str(parent_dir / 'pretrain'))
                pretrained_file = os.path.join(pretrain_base, base_name)
                if not os.path.exists(pretrained_file):
                    print(f"Warning: Pretrained file not found at {pretrained_file}")
                    pretrained = False
        
        # Initialize PVIG model
        try:
            model = model_func(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                pretrained=pretrained,
                pretrained_path=pretrained_file
            )
        except TypeError:
            # Fallback for models that don't accept all parameters
            model = model_func(
                num_classes=num_classes,
                drop_path_rate=drop_path_rate,
                pretrained=pretrained
            )
    
    model = model.to(device)
    
    # Multi-GPU support if available
    if (device.type == 'cuda' and torch.cuda.device_count() > 1) or \
       (device.type == 'mps' and torch.mps.device_count() > 1):
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count() if device.type == 'cuda' else torch.mps.device_count()} {device.type.upper()} devices")
    
    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer_name = config.get('optimizer', 'AdamW')
    lr = config.get('learning_rate', 1e-4)
    weight_decay = config.get('weight_decay', 1e-2)
    
    if optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    scheduler_name = config.get('scheduler', None)
    scheduler_params = config.get('scheduler_params', {})
    scheduler = None
    
    if scheduler_name == 'StepLR' and scheduler_params:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_params['step_size'],
            gamma=scheduler_params['gamma']
        )
    elif scheduler_name == 'CosineAnnealingLR' and scheduler_params:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_params['T_max'],
            eta_min=scheduler_params['eta_min']
        )
    elif scheduler_name == 'OneCycleLR' and scheduler_params:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=scheduler_params['max_lr'],
            total_steps=scheduler_params['total_steps'],
            pct_start=scheduler_params['pct_start'],
            anneal_strategy=scheduler_params['anneal_strategy']
        )
    
    # Early stopping variables
    best_acc = 0.0
    best_auc = 0.0
    best_epoch = 0
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"Epoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_loss, train_acc = train_loop(train_loader, model, loss_fn, optimizer, device)
        
        # Test
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1 = test_loop(test_loader, model, loss_fn, device)
        
        # Log metrics to file
        metrics_path = os.path.join(output_dir, 'metrics.json')
        metrics = {
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        # Append to metrics file
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        all_metrics.append(metrics)
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Update scheduler if enabled
        if scheduler:
            if scheduler_name in ['StepLR', 'CosineAnnealingLR']:
                scheduler.step()
        
        # Report to Optuna
        trial.report(test_auc, epoch)
        
        # Save checkpoints
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, metrics, checkpoint_path)
        
        # Save best model
        is_best_acc = test_acc > best_acc
        is_best_auc = test_auc > best_auc
        
        if is_best_acc:
            best_acc = test_acc
            best_checkpoint_acc = os.path.join(checkpoint_dir, "best_acc_checkpoint.pth")
            save_checkpoint(model, optimizer, epoch, metrics, best_checkpoint_acc, is_best=True, best_type='accuracy')
        
        if is_best_auc:
            best_auc = test_auc
            best_epoch = epoch
            patience_counter = 0
            best_checkpoint_auc = os.path.join(checkpoint_dir, "best_auc_checkpoint.pth")
            save_checkpoint(model, optimizer, epoch, metrics, best_checkpoint_auc, is_best=True, best_type='AUC')
        else:
            patience_counter += 1
        
        # Handle pruning (early stopping for this trial)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
    
    # Return the best AUC value for optimization
    return best_auc

def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning for TopoGNN")
    parser.add_argument('--config', type=str, required=True, help='Path to base YAML config file')
    parser.add_argument('--study_name', type=str, default='topognn_study', help='Name for the Optuna study')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of trials to run')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds for the study')
    parser.add_argument('--db_path', type=str, default=None, 
                        help='Path to SQLite database for storing study results (default: in-memory)')
    
    args = parser.parse_args()
    
    # Load base configuration
    with open(args.config, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Get timestamp for unique study name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"{args.study_name}_{timestamp}"
    
    # Set up storage
    if args.db_path:
        storage = f"sqlite:///{args.db_path}"
    else:
        storage = None
    
    # Create pruner and sampler
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10, interval_steps=1
    )
    sampler = optuna.samplers.TPESampler(seed=base_config.get('seed', 42))
    
    # Create optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize AUC
        storage=storage,
        pruner=pruner,
        sampler=sampler,
        load_if_exists=True
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, base_config),
        n_trials=args.n_trials,
        timeout=args.timeout
    )
    
    # Print results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value (AUC): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best parameters to file
    output_dir = Path(base_config.get('output_base_dir', 'results')) / "optuna_results"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save best parameters
    best_params = trial.params
    best_params_path = output_dir / f"best_params_{timestamp}.yaml"
    
    # Create full config with best params
    best_config = base_config.copy()
    best_config.update(best_params)
    best_config['experiment_name'] = f"{base_config['experiment_name']}_best_optuna_{timestamp}"
    
    # Save as YAML
    with open(best_params_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    print(f"Best parameters saved to {best_params_path}")
    
    # Save study statistics
    study_stats = {
        'best_trial': {
            'number': trial.number,
            'value': trial.value,
            'params': trial.params
        },
        'n_trials': len(study.trials),
        'n_completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
        'n_pruned_trials': len([t for t in study.trials if t.state == TrialState.PRUNED]),
        'datetime': timestamp
    }
    
    stats_path = output_dir / f"study_statistics_{timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(study_stats, f, indent=2)
    
    print(f"Study statistics saved to {stats_path}")
    
    # Generate and save optimization history plots
    try:
        import matplotlib.pyplot as plt
        
        # Optimization history plot
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.tight_layout()
        plt.savefig(output_dir / f"optimization_history_{timestamp}.png")
        
        # Parameter importance plot
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.tight_layout()
        plt.savefig(output_dir / f"param_importances_{timestamp}.png")
        
        # Parallel coordinate plot
        plt.figure(figsize=(15, 8))
        optuna.visualization.matplotlib.plot_parallel_coordinate(study)
        plt.tight_layout()
        plt.savefig(output_dir / f"parallel_coordinate_{timestamp}.png")
        
        print(f"Plots saved to {output_dir}")
    except Exception as e:
        print(f"Could not generate plots: {e}")

if __name__ == "__main__":
    main() 