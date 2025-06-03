# TopoGNN Model Architectures

This repository contains implementations of various deep learning architectures enhanced with topological feature integration for medical image classification. All models use **early fusion** for optimal topological feature integration.

## 🏗️ Architecture Overview

### Early Fusion Strategy
All transformer models now use **early fusion** where topological features are projected and added to patch embeddings early in the pipeline, ensuring topological information influences the entire model processing.

```python
# Early fusion workflow:
# 1. Image → Patch Embeddings (B, N, D)
# 2. Topo Features (B, 2, 56, 56) → Projection → (B, D, H', W') → (B, N, D)  
# 3. Combined Features: patch_embeddings + projected_topo → (B, N, D)
# 4. Process through transformer layers
```

## 🚀 Available Models

### 1. PVIG (Pyramid Vision Graph) - **Reference Implementation** 
**Location**: `models/pvig_topo_proj.py`

- **PVIG-Tiny**: `pvig_ti_224_gelu_proj()`
- **PVIG-Small**: `pvig_s_224_gelu_proj()`  
- **PVIG-Medium**: `pvig_m_224_gelu_proj()`
- **PVIG-Base**: `pvig_b_224_gelu_proj()`

**Key Features**:
- ✅ **Early fusion**: Projects 2-channel topo → backbone embedding dimension, then adds
- ✅ **Graph-based processing**: Vision Graph Neural Network architecture
- ✅ **Multi-scale pyramid**: Hierarchical feature processing
- ✅ **Pretrained weights**: Available for all variants
- ✅ **Production ready**: Most stable and tested implementation

### 2. Vision Transformers (ViT) - **Updated with Early Fusion**
**Location**: `models/vit_topo_proj.py`

- **ViT-Tiny**: `vit_tiny_224_gelu_proj()` (192-dim)
- **ViT-Small**: `vit_small_224_gelu_proj()` (384-dim)  
- **ViT-Base**: `vit_base_224_gelu_proj()` (768-dim)
- **ViT-Large**: `vit_large_224_gelu_proj()` (1024-dim)

**Key Features**:
- ✅ **Early fusion**: Topo features added to patch embeddings via hooks
- ✅ **TIMM integration**: Uses `timm.VisionTransformer` as backbone
- ✅ **Dimension detection**: Automatically detects embedding dimensions
- ✅ **Hook-based fusion**: Intercepts patch embeddings for seamless integration

### 3. Swin Transformers - **Updated with Early Fusion**  
**Location**: `models/swin_topo_proj.py`

- **Swin-Tiny**: `swin_tiny_224_gelu_proj()` (96-dim)
- **Swin-Small**: `swin_small_224_gelu_proj()` (96-dim, deeper)
- **Swin-Base**: `swin_base_224_gelu_proj()` (128-dim)

**Key Features**:
- ✅ **Early fusion**: Same hook-based approach as ViT
- ✅ **Hierarchical windows**: Shifted window attention
- ✅ **TIMM integration**: Uses `timm.SwinTransformer` backbone
- ✅ **Efficient computation**: Better than ViT for high-resolution features

### 4. Unified Transformer Interface - **New**
**Location**: `models/transformer.py` 

**Key Features**:
- ✅ **Universal wrapper**: `TransformerTopoFusion` class handles all TIMM models
- ✅ **Auto-detection**: Determines embedding dimensions automatically  
- ✅ **Hook registration**: Sets up early fusion for any transformer architecture
- ✅ **TIMM compatibility**: Works with any TIMM vision transformer

**Supported TIMM Models**:
```python
# These prefixes are automatically detected:
'vit_', 'swin_', 'deit_', 'efficientnet_', 'convnext_', 'resnet', 'resnext', 'densenet'

# Examples:
- vit_base_patch16_224
- swin_base_patch4_window7_224  
- deit_base_patch16_224
- convnext_base
```

### 5. CNN Models - **Classic Implementations**
**Locations**: `models/resnet_topo_proj.py`, `models/densenet_topo_proj.py`

- **ResNet**: 18, 50, 101 variants with late fusion
- **DenseNet121**: Efficient dense connections with topo integration
- **Note**: These use late fusion and are maintained for comparison

## 🔧 Unified Training Pipeline

### Single Training Script
All models now use the unified `main.py` script:

```bash
# PVIG models
python main.py --config configs/train_config_proj.yaml

# Swin Transformers  
python main.py --config configs/swin_train_config.yaml

# ViT models
python main.py --config configs/train_vit_topo.yaml

# Any TIMM model via transformer.py
python main.py --config configs/custom_timm_config.yaml
```

### Model Selection Logic
```python
# In main.py:
if model_type.startswith(('vit_', 'swin_', 'deit_', ...)):
    # Use unified TransformerTopoFusion wrapper
    model = create_transformer_model(config)
else:
    # Use custom model constructors (PVIG, etc.)
    model = load_custom_model(config)
```

## ⚙️ Configuration Structure

### Standard Config Format
```yaml
# Model selection
model_type: "swin_base_patch4_window7_224"  # TIMM name or custom name
num_classes: 3
pretrained: true
drop_path_rate: 0.1

# Early fusion settings
use_topo_features: true
topo_features_config:
  in_chans: 2           # Input topo channels
  img_size: 56          # Topo feature spatial size

# Training settings  
batch_size: 16
learning_rate: 0.0001
epochs: 50
optimizer: "AdamW"
```

## 📊 Performance Characteristics

### Computational Requirements (Updated)

| Model | Parameters | Memory (GB) | Batch Size | Early Fusion |
|-------|------------|-------------|------------|--------------|
| PVIG-Tiny | ~15M | ~2 | 32 | ✅ Native |
| ViT-Tiny | ~6M | ~2 | 32 | ✅ Hook-based |
| PVIG-Small | ~25M | ~3 | 16 | ✅ Native |
| ViT-Small | ~22M | ~3 | 16 | ✅ Hook-based |
| Swin-Tiny | ~28M | ~4 | 16 | ✅ Hook-based |
| PVIG-Medium | ~40M | ~5 | 12 | ✅ Native |
| PVIG-Base | ~80M | ~6 | 8 | ✅ Native |
| ViT-Base | ~86M | ~6 | 8 | ✅ Hook-based |
| Swin-Base | ~87M | ~8 | 4 | ✅ Hook-based |

### Recommended Use Cases

- **Quick Experiments**: PVIG-Tiny, ViT-Tiny
- **Balanced Performance**: PVIG-Small, Swin-Tiny  
- **High Accuracy**: PVIG-Base, ViT-Base, Swin-Base
- **Research/SOTA**: ViT-Large, custom TIMM models

## 🔬 Topological Feature Integration

### Early Fusion Architecture
```python
class TransformerTopoFusion(nn.Module):
    def __init__(self, config):
        # 1. Create TIMM backbone
        self.backbone = timm.create_model(model_type, ...)
        
        # 2. Detect embedding dimension
        embed_dim = self._detect_embed_dim()  # e.g., 768 for ViT-Base
        
        # 3. Create projection layer
        self.topo_proj = nn.Sequential(
            nn.Conv2d(2, embed_dim, kernel_size=1),  # 2 → 768 channels
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )
        
        # 4. Register hook for early fusion
        self.backbone.patch_embed.register_forward_hook(patch_embed_hook)
```

### Fusion Process
1. **Image Processing**: `(B, 3, 224, 224)` → `patch_embed` → `(B, N, D)`
2. **Topo Processing**: `(B, 2, 56, 56)` → `topo_proj` → `(B, D, 56, 56)` → resize & flatten → `(B, N, D)`
3. **Early Fusion**: `patch_embeddings + projected_topo` → `(B, N, D)`
4. **Transformer Processing**: Continue through transformer blocks

## 🛠️ Installation & Setup

### Core Requirements
```bash
# Basic dependencies
pip install torch torchvision tqdm pyyaml

# For TIMM models (recommended)
pip install timm

# For visualization
pip install matplotlib seaborn scikit-learn

# For hyperparameter optimization  
pip install optuna plotly
```

### Quick Start
```bash
# 1. Clone and setup
git clone <repository>
cd TopoGNN

# 2. Install dependencies
pip install -r requirements.txt

# 3. Prepare data (ensure topo features exist)
python utils/check_topo_features.py

# 4. Train with early fusion
python main.py --config configs/swin_train_config.yaml
```

## 🧪 Hyperparameter Optimization

### Optuna Integration
```bash
# Run automated hyperparameter search
./optuna/run_optuna_tuning.sh --config configs/swin_train_config.yaml --n_trials 50

# Resume previous study
./optuna/run_optuna_tuning.sh --db_path optuna/my_study.db --study_name my_study
```

## 🔍 Model Comparison & Analysis

### Interface Consistency
All models follow the same forward interface:
```python
def forward(self, x_image, topo_features=None):
    """
    Args:
        x_image: Input images (B, 3, 224, 224)
        topo_features: Topological features (B, 2, 56, 56) or None
    
    Returns:
        logits: Class predictions (B, num_classes)
    """
```

### Evaluation Tools
```bash
# Compare multiple experiments
python utils/visualize_results.py --results_dir results/

# Analyze attention patterns
python utils/visualize_attention.py --model_path results/best_model.pth

# Check feature importance
python utils/topo_importance_processor.py
```

## 🐛 Troubleshooting

### Common Issues

1. **Shape Mismatches**: All resolved with current early fusion implementation
2. **TIMM Import Errors**: Install timm or use fallback implementations
3. **Memory Issues**: Reduce batch size or use gradient accumulation
4. **Hook Conflicts**: Only one hook per model instance

### Performance Tips

1. **Use Early Fusion**: Current implementation is optimal
2. **Mixed Precision**: Enable for memory efficiency
3. **Gradient Checkpointing**: For very large models
4. **Pretrained Weights**: Always recommended for faster convergence

## 📈 Results & Benchmarks

The early fusion approach has shown improved performance over late fusion:
- **Better feature integration**: Topo features influence entire pipeline
- **Consistent training**: No shape mismatches or interface issues  
- **Model agnostic**: Works with any TIMM transformer architecture
- **Production ready**: Stable and well-tested implementation

For detailed results and comparisons, see the experiments in the `results/` directory. 