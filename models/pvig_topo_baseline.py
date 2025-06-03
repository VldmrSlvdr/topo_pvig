"""
Code is referenced from
https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/vig_pytorch/pyramid_vig.py

MODIFIED: Pure baseline model that doesn't use topological features.
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


class DeepGCN_Baseline(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN_Baseline, self).__init__()

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

        # No topological feature processing in this baseline model

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

    def forward(self, inputs, topo_features=None):
        # topo_features is accepted for API compatibility but not used
        
        # Process image only
        x = self.stem(inputs) + self.pos_embed

        # Pass features through backbone
        B, C, H, W = x.shape
        feature = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if hasattr(self, 'pyramid_levels') and self.pyramid_levels is not None and i in self.pyramid_levels:
                feature.append(x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)


# --- Model Creation Functions --- 

def pvig_ti_224_gelu_baseline(pretrained=False, pretrained_path=None, **kwargs):
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
    model = DeepGCN_Baseline(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_ti_78.5.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_ti_224_gelu_baseline from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
            
            # Remove weights for any layers from topo models that don't exist in baseline
            state_dict.pop('b1.weight', None)
            state_dict.pop('b1.bias', None) 
            state_dict.pop('b1.running_mean', None)
            state_dict.pop('b1.running_var', None)
            state_dict.pop('b1.num_batches_tracked', None)
            # Remove any topo_proj or topo_gate_combine weights
            keys_to_remove = [k for k in state_dict.keys() if 'topo_proj' in k or 'topo_gate_combine' in k]
            for k in keys_to_remove:
                state_dict.pop(k, None)

            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.' # Usually prediction.4.*
            final_pred_weight = prediction_prefix + 'weight'
            final_pred_bias = prediction_prefix + 'bias'

            if opt.n_classes != 1000: # Assuming pretrained was on 1000 classes
                state_dict.pop(final_pred_weight, None)
                state_dict.pop(final_pred_bias, None)
                print(f"Removed final prediction layer weights due to class mismatch.")

            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            
            # Report missing keys - these should only be the topo_* ones that we expect to be missing
            if load_result.missing_keys:
                print(f"Missing keys (expected for a baseline model): {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys (should be none): {load_result.unexpected_keys}")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")

    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Initialize or re-initialize the final prediction layer if necessary
    num_final_features = model.prediction[-1].in_channels
    current_final_pred = model.prediction[-1]
    if not (isinstance(current_final_pred, nn.Conv2d) and current_final_pred.out_channels == opt.n_classes):
        print(f"Re-initializing final prediction layer for {opt.n_classes} classes.")
        model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
        m = model.prediction[-1]
        torch.nn.init.kaiming_normal_(m.weight)
        m.weight.requires_grad = True
        if m.bias is not None:
            m.bias.data.zero_()
            m.bias.requires_grad = True
    else:
        print("Final prediction layer already matches target classes.")

    return model

def pvig_s_224_gelu_baseline(pretrained=False, pretrained_path=None, **kwargs):
    """ViG Small baseline model without topological features."""
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
    model = DeepGCN_Baseline(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_s_82.1.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_s_224_gelu_baseline from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
            
            # Remove weights for any layers from topo models that don't exist in baseline
            state_dict.pop('b1.weight', None)
            state_dict.pop('b1.bias', None) 
            state_dict.pop('b1.running_mean', None)
            state_dict.pop('b1.running_var', None)
            state_dict.pop('b1.num_batches_tracked', None)
            # Remove any topo_proj or topo_gate_combine weights
            keys_to_remove = [k for k in state_dict.keys() if 'topo_proj' in k or 'topo_gate_combine' in k]
            for k in keys_to_remove:
                state_dict.pop(k, None)

            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.' # Usually prediction.4.*
            final_pred_weight = prediction_prefix + 'weight'
            final_pred_bias = prediction_prefix + 'bias'

            if opt.n_classes != 1000: # Assuming pretrained was on 1000 classes
                state_dict.pop(final_pred_weight, None)
                state_dict.pop(final_pred_bias, None)
                print(f"Removed final prediction layer weights due to class mismatch.")

            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            
            # Report missing keys - these should only be the topo_* ones that we expect to be missing
            if load_result.missing_keys:
                print(f"Missing keys (expected for a baseline model): {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys (should be none): {load_result.unexpected_keys}")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")

    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Initialize or re-initialize the final prediction layer if necessary
    num_final_features = model.prediction[-1].in_channels
    current_final_pred = model.prediction[-1]
    if not (isinstance(current_final_pred, nn.Conv2d) and current_final_pred.out_channels == opt.n_classes):
        print(f"Re-initializing final prediction layer for {opt.n_classes} classes.")
        model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
        m = model.prediction[-1]
        torch.nn.init.kaiming_normal_(m.weight)
        m.weight.requires_grad = True
        if m.bias is not None:
            m.bias.data.zero_()
            m.bias.requires_grad = True
    else:
        print("Final prediction layer already matches target classes.")

    return model

def pvig_m_224_gelu_baseline(pretrained=False, pretrained_path=None, **kwargs):
    """ViG Medium baseline model without topological features."""
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
    model = DeepGCN_Baseline(opt)
    model.default_cfg = default_cfgs['vig_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_m_83.1.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_m_224_gelu_baseline from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
            
            # Remove weights for any layers from topo models that don't exist in baseline
            state_dict.pop('b1.weight', None)
            state_dict.pop('b1.bias', None) 
            state_dict.pop('b1.running_mean', None)
            state_dict.pop('b1.running_var', None)
            state_dict.pop('b1.num_batches_tracked', None)
            # Remove any topo_proj or topo_gate_combine weights
            keys_to_remove = [k for k in state_dict.keys() if 'topo_proj' in k or 'topo_gate_combine' in k]
            for k in keys_to_remove:
                state_dict.pop(k, None)

            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.' # Usually prediction.4.*
            final_pred_weight = prediction_prefix + 'weight'
            final_pred_bias = prediction_prefix + 'bias'

            if opt.n_classes != 1000: # Assuming pretrained was on 1000 classes
                state_dict.pop(final_pred_weight, None)
                state_dict.pop(final_pred_bias, None)
                print(f"Removed final prediction layer weights due to class mismatch.")

            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            
            # Report missing keys - these should only be the topo_* ones that we expect to be missing
            if load_result.missing_keys:
                print(f"Missing keys (expected for a baseline model): {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys (should be none): {load_result.unexpected_keys}")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")

    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Initialize or re-initialize the final prediction layer if necessary
    num_final_features = model.prediction[-1].in_channels
    current_final_pred = model.prediction[-1]
    if not (isinstance(current_final_pred, nn.Conv2d) and current_final_pred.out_channels == opt.n_classes):
        print(f"Re-initializing final prediction layer for {opt.n_classes} classes.")
        model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
        m = model.prediction[-1]
        torch.nn.init.kaiming_normal_(m.weight)
        m.weight.requires_grad = True
        if m.bias is not None:
            m.bias.data.zero_()
            m.bias.requires_grad = True
    else:
        print("Final prediction layer already matches target classes.")

    return model

def pvig_b_224_gelu_baseline(pretrained=False, pretrained_path=None, **kwargs):
    """ViG Base/Large baseline model without topological features."""
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
    model = DeepGCN_Baseline(opt)
    model.default_cfg = default_cfgs['vig_b_224_gelu']

    # Default path as fallback
    default_pretrained_path = "./pretrain/pvig_b_83.66.pth.tar"
    # Use provided path if given, otherwise use default
    effective_path = pretrained_path if pretrained_path else default_pretrained_path
    
    if pretrained and os.path.exists(effective_path):
        print(f"Loading pretrained weights for pvig_b_224_gelu_baseline from {effective_path}...")
        try:
            checkpoint = torch.load(effective_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                state_dict = checkpoint
                print("Warning: Loaded checkpoint is not a dict. Assuming it's a raw state_dict.")
            
            # Remove weights for any layers from topo models that don't exist in baseline
            state_dict.pop('b1.weight', None)
            state_dict.pop('b1.bias', None) 
            state_dict.pop('b1.running_mean', None)
            state_dict.pop('b1.running_var', None)
            state_dict.pop('b1.num_batches_tracked', None)
            # Remove any topo_proj or topo_gate_combine weights
            keys_to_remove = [k for k in state_dict.keys() if 'topo_proj' in k or 'topo_gate_combine' in k]
            for k in keys_to_remove:
                state_dict.pop(k, None)

            prediction_prefix = 'prediction.' + str(len(model.prediction) - 1) + '.' # Usually prediction.4.*
            final_pred_weight = prediction_prefix + 'weight'
            final_pred_bias = prediction_prefix + 'bias'

            if opt.n_classes != 1000: # Assuming pretrained was on 1000 classes
                state_dict.pop(final_pred_weight, None)
                state_dict.pop(final_pred_bias, None)
                print(f"Removed final prediction layer weights due to class mismatch.")

            load_result = model.load_state_dict(state_dict, strict=False)
            print(f"Pretrained weights load result: {load_result}")
            
            # Report missing keys - these should only be the topo_* ones that we expect to be missing
            if load_result.missing_keys:
                print(f"Missing keys (expected for a baseline model): {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"Unexpected keys (should be none): {load_result.unexpected_keys}")

        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with randomly initialized model.")

    elif pretrained:
        print(f"Warning: Pretrained weights path not found: {effective_path}")
        print("Proceeding with randomly initialized model.")

    # Initialize or re-initialize the final prediction layer if necessary
    num_final_features = model.prediction[-1].in_channels
    current_final_pred = model.prediction[-1]
    if not (isinstance(current_final_pred, nn.Conv2d) and current_final_pred.out_channels == opt.n_classes):
        print(f"Re-initializing final prediction layer for {opt.n_classes} classes.")
        model.prediction[-1] = nn.Conv2d(num_final_features, opt.n_classes, 1, bias=True)
        m = model.prediction[-1]
        torch.nn.init.kaiming_normal_(m.weight)
        m.weight.requires_grad = True
        if m.bias is not None:
            m.bias.data.zero_()
            m.bias.requires_grad = True
    else:
        print("Final prediction layer already matches target classes.")

    return model

if __name__ == "__main__":
    model = pvig_ti_224_gelu_baseline(num_classes=3, pretrained=False)
    print(model)
    batch_size = 4
    input_img = torch.rand(batch_size, 3, 224, 224)
    # Dummy topo_features, but they'll be ignored
    dummy_topo = torch.rand(batch_size, 2, 56, 56)
    output = model(input_img, dummy_topo)
    print("Output shape:", output.shape) 