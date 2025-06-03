import torch
import torch.nn as nn
import timm
import torch.nn.functional as F
from models.swin_topo_proj import swin_tiny_proj, swin_small_proj, swin_base_proj

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size // patch_size, img_size // patch_size) if isinstance(img_size, int) else (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if H < self.patch_size or W < self.patch_size:
            raise ValueError(
                f"Input image size ({H}x{W}) is smaller than patch size ({self.patch_size}x{self.patch_size}). "
                f"Ensure input image size is adequate for the patch size."
            )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class TransformerTopoFusion(nn.Module):
    """
    Wrapper class for Vision Transformer and Swin Transformer models 
    with topological feature integration using EARLY FUSION.
    Aligns with pvig_topo_proj.py approach: project topo features and add to initial image features.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_classes = config.get('num_classes', 3)
        self.img_size = config.get('img_size', 224)
        self.use_topo_features = config.get('use_topo_features', True)
        
        # Create base transformer model (ViT or Swin)
        self.backbone = timm.create_model(
            config['model_type'],
            pretrained=config.get('pretrained', True),
            num_classes=self.num_classes,  # Keep original head
            drop_path_rate=config.get('drop_path_rate', 0.1),
            **({'img_size': self.img_size} if 'vit' in config['model_type'] else {})
        )
        
        if self.use_topo_features:
            # Get the embedding dimension from the backbone
            # For transformers, this is typically the patch embedding dimension
            print(f"Determining actual embedding dimension for {config['model_type']}...")
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, self.img_size, self.img_size)
                if hasattr(self.backbone, 'patch_embed'):
                    # Get actual patch embedding output to determine correct dimension
                    dummy_patches = self.backbone.patch_embed(dummy_input)
                    print(f"Patch embedding output shape: {dummy_patches.shape}")
                    
                    if dummy_patches.ndim == 3:  # (B, N, D)
                        embed_dim = dummy_patches.shape[-1]
                    elif dummy_patches.ndim == 4:  # (B, D, H, W)
                        embed_dim = dummy_patches.shape[1]
                    else:
                        raise ValueError(f"Unexpected patch embedding output shape: {dummy_patches.shape}")
                else:
                    # Fallback: For models without patch_embed, assume 768
                    embed_dim = 768
            
            print(f"Backbone embedding dimension: {embed_dim}")
            
            # Early fusion: project topo features to match backbone embedding dimension
            # This aligns with pvig_topo_proj.py approach
            topo_config = config.get('topo_features_config', {})
            topo_in_chans = topo_config.get('in_chans', 2)
            
            # Project 2-channel topo features to backbone embedding dimension
            # Following pvig_topo_proj.py pattern
            self.topo_proj = nn.Sequential(
                nn.Conv2d(topo_in_chans, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()  # Use GELU to match transformer activation
            )
            
            print(f"Topo projection: {topo_in_chans} -> {embed_dim} channels")
            
            # Hook to intercept and modify patch embeddings
            self._setup_early_fusion_hook()
        else:
            print("Using backbone features only (no topological features)")

    def _setup_early_fusion_hook(self):
        """Setup hook to intercept patch embeddings and add projected topo features"""
        def patch_embed_hook(module, input, output):
            if hasattr(self, '_current_topo_features') and self._current_topo_features is not None:
                # Project topo features
                projected_topo = self.topo_proj(self._current_topo_features)
                
                # Handle different output formats
                if output.ndim == 3:  # (B, N, D) - sequence format
                    B, N, D = output.shape
                    # Convert projected topo to sequence format
                    # Resize to match patch grid size
                    patch_grid_size = int(N ** 0.5)  # Assume square grid
                    projected_topo_resized = F.interpolate(
                        projected_topo, 
                        size=(patch_grid_size, patch_grid_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    # Flatten to sequence format
                    projected_topo_seq = projected_topo_resized.flatten(2).transpose(1, 2)  # (B, N, D)
                    # Add to patch embeddings
                    output = output + projected_topo_seq
                elif output.ndim == 4:  # (B, D, H, W) - spatial format
                    # Resize topo features to match output spatial dimensions
                    projected_topo_resized = F.interpolate(
                        projected_topo,
                        size=(output.shape[2], output.shape[3]),
                        mode='bilinear',
                        align_corners=False
                    )
                    # Add to output
                    output = output + projected_topo_resized
                else:
                    raise ValueError(f"Unexpected patch embedding output shape: {output.shape}")
                
                # Clear the stored topo features
                self._current_topo_features = None
            
            return output
        
        # Register hook on patch embedding layer
        if hasattr(self.backbone, 'patch_embed'):
            self.backbone.patch_embed.register_forward_hook(patch_embed_hook)
        else:
            print("Warning: No patch_embed found in backbone, early fusion may not work correctly")

    def forward(self, x_image, topo_features=None):
        if self.use_topo_features and topo_features is not None:
            # Store topo features for the hook to use
            self._current_topo_features = topo_features
        
        # Forward pass through backbone (hook will handle early fusion)
        output = self.backbone(x_image)
        
        return output

def create_model(config):
    """Factory function to create transformer models with topological features"""
    model_type = config['model_type']
    
    # Check if it's a transformer model that should use TransformerTopoFusion
    timm_transformer_types = ['vit_', 'swin_', 'deit_', 'efficientnet_', 'convnext_', 'resnet', 'resnext', 'densenet']
    
    # Convert specific custom model names to timm equivalents
    if model_type == 'swin_tiny_proj':
        config = config.copy()
        config['model_type'] = 'swin_tiny_patch4_window7_224'
        return TransformerTopoFusion(config)
    elif model_type == 'swin_small_proj':
        config = config.copy()
        config['model_type'] = 'swin_small_patch4_window7_224'
        return TransformerTopoFusion(config)
    elif model_type == 'swin_base_proj':
        config = config.copy()
        config['model_type'] = 'swin_base_patch4_window7_224'
        return TransformerTopoFusion(config)
    elif any(model_type.startswith(prefix) for prefix in timm_transformer_types):
        # Use TransformerTopoFusion for all timm-based models
        return TransformerTopoFusion(config)
    else:
        # Fallback to old direct constructor approach for non-transformer models
        if model_type == 'swin_tiny_proj':
            return swin_tiny_proj(**config)
        elif model_type == 'swin_small_proj':
            return swin_small_proj(**config)
        elif model_type == 'swin_base_proj':
            return swin_base_proj(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}") 