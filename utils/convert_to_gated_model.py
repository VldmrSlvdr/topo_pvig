#!/usr/bin/env python
"""
Convert a baseline model checkpoint to be partially compatible with the gated model architecture.
This script extracts transferable parameters from a baseline model and saves them in a format
suitable for partial loading into the gated model.

Usage:
    python convert_to_gated_model.py input_checkpoint.pth output_pretrained.pth.tar
"""

import torch
import os
import sys
import argparse
from collections import OrderedDict

def convert_baseline_to_gated(input_path, output_path, verbose=True):
    """
    Convert a baseline model checkpoint to be compatible with gated model.
    
    Args:
        input_path: Path to the input baseline checkpoint file
        output_path: Path where the converted weights will be saved
        verbose: Whether to print detailed information
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
        
    print(f"Loading baseline checkpoint from {input_path}")
    try:
        checkpoint = torch.load(input_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    # Extract model weights from checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print("Extracted model_state_dict from checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print("Extracted state_dict from checkpoint")
        else:
            # Try to use the whole checkpoint as state_dict if it looks like one
            if any(k.endswith('.weight') or k.endswith('.bias') for k in checkpoint.keys()):
                state_dict = checkpoint
                print("Using entire checkpoint as state_dict")
            else:
                print("Error: Could not find model weights in checkpoint")
                return False
    else:
        print("Error: Checkpoint is not a dictionary")
        return False
    
    # Handle DataParallel prefix if present
    clean_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # Remove 'module.' prefix
        else:
            name = k
        clean_state_dict[name] = v
    
    # Create new state dict for gated model
    new_state_dict = OrderedDict()
    
    # Keep track of mappings for logging
    retained_keys = []
    skipped_keys = []
    
    # Define parameter mappings between models
    # Format: {gated_model_key_pattern: baseline_model_key_pattern}
    # Use None for keys that shouldn't be copied
    mappings = {
        # Example: 'stem.': 'stem.',  # Copy all stem parameters
        # Example: 'backbone.': 'backbone.',  # Copy all backbone parameters
        # Example: 'prediction.': 'prediction.',  # Copy all prediction layer parameters
        # Example: 'topo_gate_combine.': None,  # Skip all topo gate parameters (needs to be initialized fresh)
    }
    
    # Check if keys match known patterns for different architectures
    is_baseline_to_gated = any('backbone.' in k for k in clean_state_dict.keys())
    
    if is_baseline_to_gated:
        print("Detected baseline to gated model conversion")
        # Define what parameters can be transferred
        for key, value in clean_state_dict.items():
            # Skip metadata keys
            if key in ['epoch', 'metrics', 'optimizer_state_dict']:
                skipped_keys.append(key)
                continue
                
            # Skip topo projection if present in baseline
            if 'topo_proj' in key:
                skipped_keys.append(key)
                continue
                
            # Keep backbone and stem parameters
            if any(segment in key for segment in ['backbone.', 'stem.', 'prediction.', 'pos_embed']):
                new_state_dict[key] = value
                retained_keys.append(key)
            else:
                skipped_keys.append(key)
    else:
        print("WARNING: Could not detect specific architecture patterns")
        print("Copying all parameters directly - this may not work for loading")
        new_state_dict = clean_state_dict
    
    if verbose:
        print(f"\nTotal parameters in original state_dict: {len(clean_state_dict)}")
        print(f"Parameters kept for gated model: {len(retained_keys)}")
        print(f"Parameters skipped: {len(skipped_keys)}")
        
        if len(retained_keys) > 0:
            print("\nKept parameters (first 10):")
            for i, key in enumerate(retained_keys[:10]):
                print(f"  {key}: {clean_state_dict[key].shape}")
            if len(retained_keys) > 10:
                print(f"  ... and {len(retained_keys) - 10} more")
        
        if len(skipped_keys) > 0:
            print("\nSkipped parameters (first 10):")
            for i, key in enumerate(skipped_keys[:10]):
                if key in clean_state_dict:
                    print(f"  {key}: {clean_state_dict[key].shape}")
                else:
                    print(f"  {key}")
            if len(skipped_keys) > 10:
                print(f"  ... and {len(skipped_keys) - 10} more")
    
    # Save as a standalone state_dict for pretrained loading
    print(f"Saving converted weights to {output_path}")
    try:
        torch.save(new_state_dict, output_path)
        print("Successfully converted checkpoint to gated-compatible format")
        print("\nNOTE: Since the architectures are very different, you may still see many")
        print("'missing keys' warnings when loading this into the gated model.")
        return True
    except Exception as e:
        print(f"Error saving converted weights: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a baseline model checkpoint to be compatible with gated model")
    parser.add_argument("input", help="Path to input baseline checkpoint file (.pth)")
    parser.add_argument("output", help="Path to output gated-compatible weights file (.pth.tar)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    success = convert_baseline_to_gated(args.input, args.output, verbose=not args.quiet)
    sys.exit(0 if success else 1) 