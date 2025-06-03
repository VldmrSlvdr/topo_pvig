import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

try:
    from timm.models.vision_transformer import VisionTransformer
    from timm.models.registry import register_model
    from timm.models.helpers import build_model_with_cfg
    TIMM_AVAILABLE = True
except ImportError:
    print("Warning: timm not available. ViT models will use simplified implementation.")
    TIMM_AVAILABLE = False

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

class SimpleViT(nn.Module):
    """Simplified Vision Transformer for when timm is not available"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., **kwargs):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1  # class token
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
            embed_dim=embed_dim, norm_layer=nn.LayerNorm)
        
        num_patches = self.patch_embed.num_patches
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=drop_rate,
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return self.head(x[:, 0])  # Use class token for classification

class ViT_TopoProj(nn.Module):
    """Vision Transformer with topological feature integration via early fusion projection"""
    
    def __init__(self, vit_type='vit_base', num_classes=3, pretrained=False, drop_path_rate=0.0, **kwargs):
        super(ViT_TopoProj, self).__init__()
        
        if TIMM_AVAILABLE:
            # Use timm ViT models if available
            if vit_type == 'vit_tiny':
                self.backbone = VisionTransformer(
                    img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
                    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4., qkv_bias=True,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=drop_path_rate
                )
            elif vit_type == 'vit_small':
                self.backbone = VisionTransformer(
                    img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
                    embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=drop_path_rate
                )
            elif vit_type == 'vit_base':
                self.backbone = VisionTransformer(
                    img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=drop_path_rate
                )
            elif vit_type == 'vit_large':
                self.backbone = VisionTransformer(
                    img_size=224, patch_size=16, in_chans=3, num_classes=num_classes,
                    embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4., qkv_bias=True,
                    drop_rate=0., attn_drop_rate=0., drop_path_rate=drop_path_rate
                )
            else:
                raise ValueError(f"Unsupported ViT type: {vit_type}")
            
        else:
            # Use simplified implementation
            if vit_type == 'vit_tiny':
                self.backbone = SimpleViT(
                    embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.,
                    num_classes=num_classes, drop_path_rate=drop_path_rate
                )
            elif vit_type == 'vit_small':
                self.backbone = SimpleViT(
                    embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.,
                    num_classes=num_classes, drop_path_rate=drop_path_rate
                )
            elif vit_type == 'vit_base':
                self.backbone = SimpleViT(
                    embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                    num_classes=num_classes, drop_path_rate=drop_path_rate
                )
            elif vit_type == 'vit_large':
                self.backbone = SimpleViT(
                    embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.,
                    num_classes=num_classes, drop_path_rate=drop_path_rate
                )
            else:
                raise ValueError(f"Unsupported ViT type: {vit_type}")
        
        # Get the embedding dimension from the actual patch embedding output
        print(f"Determining actual embedding dimension for {vit_type}...")
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
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
                # Fallback based on vit_type
                if vit_type == 'vit_tiny':
                    embed_dim = 192
                elif vit_type == 'vit_small':
                    embed_dim = 384
                elif vit_type == 'vit_base':
                    embed_dim = 768
                elif vit_type == 'vit_large':
                    embed_dim = 1024
                else:
                    embed_dim = 768  # Default
        
        print(f"Backbone embedding dimension: {embed_dim}")
        
        # Early fusion: project topo features to match backbone embedding dimension
        # This aligns with pvig_topo_proj.py approach
        self.topo_proj = nn.Sequential(
            nn.Conv2d(2, embed_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()  # Use GELU to match transformer activation
        )
        
        print(f"Topo projection: 2 -> {embed_dim} channels")
        
        # Hook to intercept and modify patch embeddings
        self._setup_early_fusion_hook()
        
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
    
    def forward(self, x, topo_features=None):
        """
        Forward pass with topological feature integration via early fusion
        
        Args:
            x: Input images (B, 3, 224, 224)  
            topo_features: Topological features (B, 2, 56, 56), can be None during inference
        """
        if topo_features is not None:
            # Store topo features for the hook to use
            self._current_topo_features = topo_features
        
        # Forward pass through backbone (hook will handle early fusion)
        output = self.backbone(x)
        
        return output

def vit_tiny_224_gelu_proj(pretrained=False, pretrained_path=None, custom_pretrained_path=None, use_custom_pretrained=False, **kwargs):
    """
    Create ViT-Tiny model with topological feature projection
    """
    num_classes = kwargs.get('num_classes', 3)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)
    
    model = ViT_TopoProj(
        vit_type='vit_tiny',
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    # Load custom pretrained weights if specified
    if use_custom_pretrained and custom_pretrained_path:
        try:
            print(f"Loading custom pretrained weights from {custom_pretrained_path}")
            checkpoint = torch.load(custom_pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print("Custom pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"Error loading custom pretrained weights: {e}")
            print("Continuing with standard initialization")
    
    return model

def vit_small_224_gelu_proj(pretrained=False, pretrained_path=None, custom_pretrained_path=None, use_custom_pretrained=False, **kwargs):
    """
    Create ViT-Small model with topological feature projection
    """
    num_classes = kwargs.get('num_classes', 3)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)
    
    model = ViT_TopoProj(
        vit_type='vit_small',
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    # Load custom pretrained weights if specified
    if use_custom_pretrained and custom_pretrained_path:
        try:
            print(f"Loading custom pretrained weights from {custom_pretrained_path}")
            checkpoint = torch.load(custom_pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print("Custom pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"Error loading custom pretrained weights: {e}")
            print("Continuing with standard initialization")
    
    return model

def vit_base_224_gelu_proj(pretrained=False, pretrained_path=None, custom_pretrained_path=None, use_custom_pretrained=False, **kwargs):
    """
    Create ViT-Base model with topological feature projection
    """
    num_classes = kwargs.get('num_classes', 3)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)
    
    model = ViT_TopoProj(
        vit_type='vit_base',
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    # Load custom pretrained weights if specified
    if use_custom_pretrained and custom_pretrained_path:
        try:
            print(f"Loading custom pretrained weights from {custom_pretrained_path}")
            checkpoint = torch.load(custom_pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print("Custom pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"Error loading custom pretrained weights: {e}")
            print("Continuing with standard initialization")
    
    return model

def vit_large_224_gelu_proj(pretrained=False, pretrained_path=None, custom_pretrained_path=None, use_custom_pretrained=False, **kwargs):
    """
    Create ViT-Large model with topological feature projection
    """
    num_classes = kwargs.get('num_classes', 3)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)
    
    model = ViT_TopoProj(
        vit_type='vit_large',
        num_classes=num_classes,
        pretrained=pretrained,
        drop_path_rate=drop_path_rate,
        **kwargs
    )
    
    # Load custom pretrained weights if specified
    if use_custom_pretrained and custom_pretrained_path:
        try:
            print(f"Loading custom pretrained weights from {custom_pretrained_path}")
            checkpoint = torch.load(custom_pretrained_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            model.load_state_dict(state_dict, strict=False)
            print("Custom pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"Error loading custom pretrained weights: {e}")
            print("Continuing with standard initialization")
    
    return model

# Aliases for convenience
vit_tiny_proj = vit_tiny_224_gelu_proj
vit_small_proj = vit_small_224_gelu_proj
vit_base_proj = vit_base_224_gelu_proj
vit_large_proj = vit_large_224_gelu_proj 