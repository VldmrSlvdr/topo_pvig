# Hyperparameter Tuning with Optuna

This guide explains how to use Optuna for hyperparameter optimization with the TopoGNN model.

## Directory Structure

The Optuna optimization files are located in the `optuna/` directory:
- `optuna/optuna_tuning.py` - Main optimization script
- `optuna/run_optuna_tuning.sh` - Shell script to run optimization
- `optuna/OPTUNA_README.md` - This documentation
- `optuna/optuna_studies.db` - SQLite database for storing results

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
pip install optuna plotly matplotlib --upgrade
```

## Hyperparameter Search Space

The current implementation optimizes the following hyperparameters:

- **Learning Rate**: Range 1e-5 to 1e-3 (log scale)
- **Weight Decay**: Range 1e-6 to 1e-2 (log scale)
- **Optimizer**: Choice of AdamW, Adam, or SGD
- **Learning Rate Scheduler**: Choice of StepLR, CosineAnnealingLR, OneCycleLR, or None
- **Batch Size**: Choice of 16, 32, 64, or 128
- **Drop Path Rate**: Range 0.0 to 0.3 (controls stochastic depth)

Scheduler-specific parameters are also tuned based on the selected scheduler.

## Running the Optimization

### Option 1: Using the Shell Script (Recommended)

Run from the **project root directory**:

```bash
./optuna/run_optuna_tuning.sh
```

### Option 2: Direct Python Execution

Run from the **project root directory**:

```bash
python optuna/optuna_tuning.py --config configs/train_config_proj.yaml
```

### Command Line Arguments

Both execution methods support the following arguments:

- `--config`: Path to the base configuration YAML file (default: configs/train_config_proj.yaml)
- `--study_name`: Name for the Optuna study (default: topognn_optimization_proj)
- `--n_trials`: Number of trials to run (default: 20)
- `--db_path`: Path to SQLite database for storing results (default: optuna/optuna_studies.db)
- `--timeout`: Timeout in seconds (default: 172800, which is 48 hours)

Example with custom arguments:

```bash
./optuna/run_optuna_tuning.sh --config configs/swin_train_config.yaml --n_trials 50 --study_name swin_optimization
```

## Supported Model Types

The optimization script supports both TIMM-based and custom models:

### TIMM-based Models
- ViT models: `vit_base_patch16_224`, `vit_small_patch16_224`, etc.
- Swin Transformer: `swin_base_patch4_window7_224`, `swin_tiny_patch4_window7_224`, etc.
- Other TIMM models: DeiT, EfficientNet, ConvNeXt, ResNet, etc.

### Custom Models
- PVIG models: `pvig_ti`, `pvig_s`, `pvig_m`, `pvig_b` with different modes (`proj`, `gated`, `baseline`)

## Results and Visualization

The optimization process will create:

1. A directory with results for each trial in your configured output base directory
2. A special directory for the best parameters and visualization plots
3. Visualization plots showing:
   - Optimization history
   - Parameter importance
   - Parallel coordinate plots of parameters

Results are saved in:
- `[output_base_dir]/optuna_results/best_params_[timestamp].yaml`: Best hyperparameters configuration
- `[output_base_dir]/optuna_results/study_statistics_[timestamp].json`: Study statistics
- `[output_base_dir]/optuna_results/*.png`: Visualization plots

## Using the Optimized Parameters

After optimization is complete, you can use the best parameters by:

1. Using the generated best parameters YAML file directly with `main.py`:
   ```bash
   python main.py --config [output_base_dir]/optuna_results/best_params_[timestamp].yaml
   ```

2. Or by manually updating your configuration file with the best parameters from the optimization.

## Customizing the Search Space

To modify the hyperparameter search space, edit the `define_search_space` function in `optuna/optuna_tuning.py`.

## Resuming a Study

To resume a previous study, simply use the same `--db_path` and `--study_name` as the original study:

```bash
./optuna/run_optuna_tuning.sh --db_path optuna/my_previous_study.db --study_name my_previous_study
```

## Important Notes

- **Working Directory**: Always run the optimization from the **project root directory**, not from inside the `optuna/` directory
- **Database Location**: The default database is stored at `optuna/optuna_studies.db` relative to the project root
- **Early Stopping**: The optimization includes early stopping with a patience of 10 epochs to prevent overfitting
- **Model Support**: Both early fusion transformer models and PVIG models are supported 