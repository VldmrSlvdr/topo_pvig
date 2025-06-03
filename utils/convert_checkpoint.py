#!/usr/bin/env python
"""
Convert a full model checkpoint to a pretrained weights format.
This script extracts only the model state_dict from a full checkpoint
and saves it in a format compatible with pretrained model loading.

Usage:
    python convert_checkpoint.py input_checkpoint.pth output_pretrained.pth.tar
"""

import torch
import os
import sys
import argparse
from collections import OrderedDict

def convert_checkpoint(input_path, output_path, verbose=True):
    """
    Convert a full checkpoint to pretrained weights format.
    
    Args:
        input_path: Path to the input checkpoint file
        output_path: Path where the converted weights will be saved
        verbose: Whether to print detailed information
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return False
        
    print(f"Loading checkpoint from {input_path}")
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
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]  # Remove 'module.' prefix
        else:
            name = k
        new_state_dict[name] = v
    
    if verbose:
        print(f"Total parameters in state_dict: {len(new_state_dict)}")
        
        # Print parameter shapes for debugging
        print("\nParameter shapes:")
        for i, (key, value) in enumerate(new_state_dict.items()):
            print(f"{key}: {value.shape}")
            if i >= 9:  # Only show first 10 parameters
                print("...")
                break
    
    # Save as a standalone state_dict for pretrained loading
    print(f"Saving converted weights to {output_path}")
    try:
        torch.save(new_state_dict, output_path)
        print("Successfully converted checkpoint to pretrained format")
        return True
    except Exception as e:
        print(f"Error saving converted weights: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a model checkpoint to pretrained weights format")
    parser.add_argument("input", help="Path to input checkpoint file (.pth)")
    parser.add_argument("output", help="Path to output pretrained weights file (.pth.tar)")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    success = convert_checkpoint(args.input, args.output, verbose=not args.quiet)
    sys.exit(0 if success else 1) 