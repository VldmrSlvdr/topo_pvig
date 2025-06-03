"""
Code is referenced from
https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/vig_pytorch/pyramid_vig.py

MODIFIED: To use only Dimension 1 topological features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath
import pdb
import os

from .gcn_lib import Grapher, act_layer

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x


class Stem(nn.Module):
    """ Image to Visual Embedding """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepGCN_TopoDim1(torch.nn.Module): # Renamed class
    def __init__(self, opt):
        super(DeepGCN_TopoDim1, self).__init__() # Renamed class

        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        blocks = opt.blocks
        imagesize = opt.imagesize
        self.pyramid_levels = opt.pyramid_levels
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]
        max_dilation = 49 // max(num_knn)

        self.stem = Stem(in_dim=3, out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], imagesize//4, imagesize//4))
        HW = imagesize // 4 * imagesize // 4

        self.topo_bn = nn.BatchNorm2d(1) # Still BatchNorm for 1 channel

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx]))
                         ]
                idx += 1
        self.backbone = Seq(*self.backbone)
        self.out_channels = channels[i]

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
            elif isinstance(m, nn.BatchNorm2d) and hasattr(self, 'topo_bn') and m == self.topo_bn:
                 nn.init.constant_(m.weight, 1)
                 nn.init.constant_(m.bias, 0)

    def forward(self, inputs, topo_features):
        # topo_features expected shape: (B, 2, 56, 56)
        # Select only Dimension 1
        topo_dim1 = topo_features[:, 1:2, :, :] # Shape: (B, 1, 56, 56)

        # Process image
        img_feat = self.stem(inputs) + self.pos_embed

        # Process topo dim 1 and add using broadcasting
        topo_proc = self.topo_bn(topo_dim1)
        x = img_feat + topo_proc # Addition works via broadcasting (C0 vs 1 channel)

        # Pass combined features through backbone
        B, C, H, W = x.shape
        feature = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if hasattr(self, 'pyramid_levels') and self.pyramid_levels is not None and i in self.pyramid_levels:
                feature.append(x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


# --- Model Creation Functions --- 

def pvig_ti_224_gelu_dim1(pretrained=False, pretrained_path=None, **kwargs): # Renamed function
    class OptInit:
        def __init__(self, num_classes=3, drop_path_rate=0.0, **kwargs):
            self.k = 9
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2]
            self.pyramid_levels = [4,11,14]
            self.imagesize = 224
            self.channels = [48, 96, 240, 384]
            self.n_classes = num_classes
            self.emb_dims = 1024

    opt = OptInit(**kwargs)
    model = DeepGCN_TopoDim1(opt) # Instantiate renamed class
    model.default_cfg = default_cfgs['vig_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_ti_78.5.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_ti_224_gelu_dim1 from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
                
            model_dict = model.state_dict()
            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.'
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and not k.startswith(prediction_prefix) and not k.startswith('topo_bn.')}
            
            if not pretrained_dict:
                print("Warning: No matching keys found...")
            else:
                print(f"Loading {len(pretrained_dict)} matching parameters.")
                
            model_dict.update(pretrained_dict)
            load_result = model.load_state_dict(model_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            if load_result.missing_keys:
                print(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys: {load_result.unexpected_keys}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")
    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Adjust final layer
    num_final_features = model.prediction[-1].in_channels
    model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
    m = model.prediction[-1]
    torch.nn.init.kaiming_normal_(m.weight)
    m.weight.requires_grad = True
    if m.bias is not None:
        m.bias.data.zero_()
        m.bias.requires_grad = True

    return model

def pvig_s_224_gelu_dim1(pretrained=False, pretrained_path=None, **kwargs):
    """ViG Small with Dimension 1 topological features."""
    class OptInit:
        def __init__(self, num_classes=3, drop_path_rate=0.0, **kwargs):
            self.k = 9
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,6,2]
            self.pyramid_levels = [4,11,14]
            self.imagesize = 224
            # Small model has wider channels
            self.channels = [80, 160, 400, 640]
            self.n_classes = num_classes
            self.emb_dims = 1024

    opt = OptInit(**kwargs)
    model = DeepGCN_TopoDim1(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_s_82.1.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_s_224_gelu_dim1 from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
                
            model_dict = model.state_dict()
            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.'
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and not k.startswith(prediction_prefix) and not k.startswith('topo_bn.')}
            
            if not pretrained_dict:
                print("Warning: No matching keys found...")
            else:
                print(f"Loading {len(pretrained_dict)} matching parameters.")
                
            model_dict.update(pretrained_dict)
            load_result = model.load_state_dict(model_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            if load_result.missing_keys:
                print(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys: {load_result.unexpected_keys}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")
    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Adjust final layer
    num_final_features = model.prediction[-1].in_channels
    model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
    m = model.prediction[-1]
    torch.nn.init.kaiming_normal_(m.weight)
    m.weight.requires_grad = True
    if m.bias is not None:
        m.bias.data.zero_()
        m.bias.requires_grad = True

    return model

def pvig_m_224_gelu_dim1(pretrained=False, pretrained_path=None, **kwargs):
    """ViG Medium with Dimension 1 topological features."""
    class OptInit:
        def __init__(self, num_classes=3, drop_path_rate=0.0, **kwargs):
            self.k = 9
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,16,2] # Medium model has more transformer blocks
            self.pyramid_levels = [4,11,24] # Adjust pyramid levels for changed block structure
            self.imagesize = 224
            # Medium model has wider channels
            self.channels = [96, 192, 384, 768]
            self.n_classes = num_classes
            self.emb_dims = 1024

    opt = OptInit(**kwargs)
    model = DeepGCN_TopoDim1(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_m_83.1.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_m_224_gelu_dim1 from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
                
            model_dict = model.state_dict()
            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.'
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and not k.startswith(prediction_prefix) and not k.startswith('topo_bn.')}
            
            if not pretrained_dict:
                print("Warning: No matching keys found...")
            else:
                print(f"Loading {len(pretrained_dict)} matching parameters.")
                
            model_dict.update(pretrained_dict)
            load_result = model.load_state_dict(model_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            if load_result.missing_keys:
                print(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys: {load_result.unexpected_keys}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")
    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Adjust final layer
    num_final_features = model.prediction[-1].in_channels
    model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
    m = model.prediction[-1]
    torch.nn.init.kaiming_normal_(m.weight)
    m.weight.requires_grad = True
    if m.bias is not None:
        m.bias.data.zero_()
        m.bias.requires_grad = True

    return model

def pvig_b_224_gelu_dim1(pretrained=False, pretrained_path=None, **kwargs):
    """ViG Base with Dimension 1 topological features."""
    class OptInit:
        def __init__(self, num_classes=3, drop_path_rate=0.0, **kwargs):
            self.k = 9
            self.conv = 'mr'
            self.act = 'gelu'
            self.norm = 'batch'
            self.bias = True
            self.dropout = 0.0
            self.use_dilation = True
            self.epsilon = 0.2
            self.use_stochastic = False
            self.drop_path = drop_path_rate
            self.blocks = [2,2,18,2] # Base model has even more transformer blocks
            self.pyramid_levels = [4,11,26] # Adjust pyramid levels for changed block structure
            self.imagesize = 224
            # Base model has the widest channels
            self.channels = [128, 256, 512, 1024]
            self.n_classes = num_classes
            self.emb_dims = 1024

    opt = OptInit(**kwargs)
    model = DeepGCN_TopoDim1(opt)
    model.default_cfg = default_cfgs['vig_b_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_b_83.66.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_b_224_gelu_dim1 from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
                
            model_dict = model.state_dict()
            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.'
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and not k.startswith(prediction_prefix) and not k.startswith('topo_bn.')}
            
            if not pretrained_dict:
                print("Warning: No matching keys found...")
            else:
                print(f"Loading {len(pretrained_dict)} matching parameters.")
                
            model_dict.update(pretrained_dict)
            load_result = model.load_state_dict(model_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            if load_result.missing_keys:
                print(f"Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys: {load_result.unexpected_keys}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")
    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Adjust final layer
    num_final_features = model.prediction[-1].in_channels
    model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
    m = model.prediction[-1]
    torch.nn.init.kaiming_normal_(m.weight)
    m.weight.requires_grad = True
    if m.bias is not None:
        m.bias.data.zero_()
        m.bias.requires_grad = True

    return model

if __name__ == "__main__":
    model = pvig_ti_224_gelu_dim1(num_classes=3)
    print(model)
    batch_size = 4
    input_img = torch.rand(batch_size, 3, 224, 224)
    input_topo = torch.rand(batch_size, 2, 56, 56)
    output = model(input_img, input_topo)
    print("Output shape:", output.shape) 