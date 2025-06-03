# Topological Feature Generator

This tool generates aligned topological features for the TopoGNN model. It ensures that the preprocessing steps match exactly with the data loading pipeline used during training.

## Features

- Uses the same preprocessing (resize to 256, center crop to 224) as the data loader
- Generates topological features in the correct 56x56 resolution
- Creates properly structured output directories
- Preserves original dataset organization (train/test and class labels)

## Installation Requirements

- Python 3.6+
- PyTorch
- torchvision
- HomCloud (`pip install homcloud`)
- NumPy
- PIL/Pillow
- tqdm
- PyYAML

## Usage

1. Edit the configuration file `configs/topo_features.yaml` to set your input/output directories
2. Run the feature generator:

```bash
python topo_feature_generator.py --config configs/topo_features.yaml
```

For more detailed logs:

```bash
python topo_feature_generator.py --config configs/topo_features.yaml --debug
```

## Configuration Options

The configuration file (`configs/topo_features.yaml`) contains the following options:

### Input/Output Settings

- `input_dir`: Directory containing source images (with train/test and class subdirectories)
- `output_dir`: Base directory where features will be saved
- `file_pattern`: File extensions to process (e.g. "*.jpg,*.png")
- `recursive`: Whether to search subdirectories recursively
- `force_reprocess`: Whether to regenerate features for images that already have them

### Topological Settings

- `filtration`: The type of filtration to use ('superlevel' or 'sublevel')
- `topo_features.lifetime_threshold`: Threshold for significant topological features

## Output Structure

The script creates the following directory structure:

```
output_dir/
├── topo_features/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── image1.npy
│   │   │   ├── image2.npy
│   │   │   └── ...
│   │   ├── class2/
│   │   └── ...
│   └── test/
│       ├── class1/
│       ├── class2/
│       └── ...
└── persistence/
    ├── train/
    └── test/
```

- `topo_features/`: Contains .npy files with the topological features (shape: 2×56×56)
- `persistence/`: Contains persistence diagram files for debugging or further analysis

## Understanding the Features

Each `.npy` file contains a numpy array with shape (2, 56, 56):
- Channel 0: Dimension 0 topological features (connected components)
- Channel 1: Dimension 1 topological features (loops/holes)

These features can be directly loaded by the data loader in `read_data_topo.py`. 