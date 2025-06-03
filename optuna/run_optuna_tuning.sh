#!/bin/bash

# Change to parent directory (project root) to run the script
cd "$(dirname "$0")/.."

# Set default values
CONFIG_PATH="configs/train_config_proj.yaml"
STUDY_NAME="topognn_optimization_proj"
N_TRIALS=20
DB_PATH="optuna/optuna_studies.db"  # Updated path relative to project root
TIMEOUT=172800  # 48 hours in seconds

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --study_name)
      STUDY_NAME="$2"
      shift 2
      ;;
    --n_trials)
      N_TRIALS="$2"
      shift 2
      ;;
    --db_path)
      DB_PATH="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--config CONFIG_PATH] [--study_name STUDY_NAME] [--n_trials N_TRIALS] [--db_path DB_PATH] [--timeout TIMEOUT]"
      exit 1
      ;;
  esac
done

echo "Running Optuna optimization from project root directory: $(pwd)"
echo "Config path: $CONFIG_PATH"
echo "Study name: $STUDY_NAME"
echo "Number of trials: $N_TRIALS"
echo "Database path: $DB_PATH"
echo "Timeout: $TIMEOUT seconds"

# Install Optuna if not already installed
pip install optuna plotly matplotlib --upgrade

# Run the optimization from the optuna subdirectory script
python optuna/optuna_tuning.py \
  --config "$CONFIG_PATH" \
  --study_name "$STUDY_NAME" \
  --n_trials "$N_TRIALS" \
  --db_path "$DB_PATH" \
  --timeout "$TIMEOUT"

echo "Optimization completed!"
echo "Results saved in your configured output directory under 'optuna_results/'" 