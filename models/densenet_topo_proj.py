import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
import math

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .875, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': 'features.conv0', 'classifier': 'classifier',
        **kwargs
    }

class DenseNet_TopoProj(nn.Module):
    """DenseNet121 with topological feature integration via projection"""
    
    def __init__(self, num_classes=3, pretrained=False, drop_path_rate=0.0, **kwargs):
        super(DenseNet_TopoProj, self).__init__()
        
        # Load base DenseNet121
        self.backbone = densenet121(pretrained=pretrained)
        
        # Remove the final classifier
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Topological feature projection
        # DenseNet features start at 64 channels after first conv+pool
        self.topo_proj = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer to combine image and topo features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0, bias=False),  # 64 + 64 -> 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        
        # Initialize projection layers
        self._init_topo_projection()
        
    def _init_topo_projection(self):
        """Initialize topological projection layers"""
        for m in self.topo_proj.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for m in self.fusion_conv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, topo_features):
        """
        Forward pass with topological feature integration
        
        Args:
            x: Input images (B, 3, 224, 224)
            topo_features: Topological features (B, 2, 56, 56)
        """
        batch_size = x.size(0)
        
        # Process image through initial layers
        x = self.backbone.features.conv0(x)  # (B, 64, 112, 112)
        x = self.backbone.features.norm0(x)
        x = self.backbone.features.relu0(x)
        x = self.backbone.features.pool0(x)  # (B, 64, 56, 56)
        
        # Process topological features
        topo_proj = self.topo_proj(topo_features)  # (B, 64, 56, 56)
        
        # Fuse image and topological features
        fused = torch.cat([x, topo_proj], dim=1)  # (B, 128, 56, 56)
        fused = self.fusion_conv(fused)  # (B, 64, 56, 56)
        
        # Continue through DenseNet blocks
        x = self.backbone.features.denseblock1(fused)
        x = self.backbone.features.transition1(x)
        
        x = self.backbone.features.denseblock2(x)
        x = self.backbone.features.transition2(x)
        
        x = self.backbone.features.denseblock3(x)
        x = self.backbone.features.transition3(x)
        
        x = self.backbone.features.denseblock4(x)
        x = self.backbone.features.norm5(x)
        
        # Global average pooling and classification
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

def densenet121_224_gelu_proj(pretrained=False, pretrained_path=None, custom_pretrained_path=None, use_custom_pretrained=False, **kwargs):
    """
    Create DenseNet121 model with topological feature projection
    
    Args:
        pretrained: Use ImageNet pretrained weights
        pretrained_path: Path to pretrained weights file
        custom_pretrained_path: Path to custom pretrained weights
        use_custom_pretrained: Whether to use custom pretrained weights
        **kwargs: Additional arguments (num_classes, drop_path_rate, etc.)
    """
    # Extract model parameters
    num_classes = kwargs.get('num_classes', 3)
    drop_path_rate = kwargs.get('drop_path_rate', 0.0)
    
    # Create model
    model = DenseNet_TopoProj(
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
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict with strict=False to allow for missing keys
            model.load_state_dict(state_dict, strict=False)
            print("Custom pretrained weights loaded successfully")
            
        except Exception as e:
            print(f"Error loading custom pretrained weights: {e}")
            print("Continuing with standard initialization")
    
    return model

# Alias for convenience
densenet121_proj = densenet121_224_gelu_proj 