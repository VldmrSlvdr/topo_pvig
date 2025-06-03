#!/usr/bin/env python3
"""
Ultra-Minimal Topological Feature Generator

Extremely simplified topological feature generator that works with
basic Python installations. No advanced dependencies required.

Just requires:
- PIL (Pillow)
- NumPy (basic version)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image

# Basic logging setup
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_image(image_path, output_dir):
    """Process a single image to generate features"""
    try:
        # Parse path components
        image_path = Path(image_path)
        base_filename = image_path.stem
        label_name = image_path.parent.name  
        mode = image_path.parent.parent.name if image_path.parent.parent.name in ['train', 'test'] else 'unknown'
        
        # Create output directory
        topo_dir = Path(output_dir) / "topo_features" / mode / label_name
        os.makedirs(topo_dir, exist_ok=True)
        
        # Output path for features
        output_path = topo_dir / f"{base_filename}.npy"
        
        # Skip if already exists
        if output_path.exists():
            return True
            
        # Open and process image
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        
        # Resize to 256
        img = img.resize((256, 256), Image.BICUBIC)
        
        # Center crop to 224x224
        width, height = img.size
        left = (width - 224) // 2
        top = (height - 224) // 2
        right = left + 224
        bottom = top + 224
        img = img.crop((left, top, right, bottom))
        
        # Convert to grayscale (average of RGB channels)
        r, g, b = img.split()
        img_array = np.array(r) * 0.299 + np.array(g) * 0.587 + np.array(b) * 0.114
        img_array = img_array.astype(np.float32) / 255.0
        
        # Resize to 56x56 for feature maps
        img_small = Image.fromarray((img_array * 255).astype(np.uint8))
        img_small = img_small.resize((56, 56), Image.BICUBIC)
        img_56 = np.array(img_small).astype(np.float32) / 255.0
        
        # Create simple topological features (bright and dark regions)
        feature_map_0 = np.zeros((56, 56), dtype=np.float32)
        feature_map_0[img_56 > 0.65] = 1.0
        
        feature_map_1 = np.zeros((56, 56), dtype=np.float32)
        feature_map_1[img_56 < 0.35] = 1.0
        
        # Stack features
        topo_features = np.stack([feature_map_0, feature_map_1], axis=0)
        
        # Save features
        np.save(output_path, topo_features)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Ultra minimal topo feature generator")
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--sample", action="store_true", help="Process only 5 images as a test")
    
    args = parser.parse_args()
    
    # Find images
    logger.info(f"Searching for images in {args.input}...")
    input_dir = Path(args.input)
    image_files = list(input_dir.glob("**/*.jpg"))
    
    logger.info(f"Found {len(image_files)} images")
    
    # Process only a sample if requested
    if args.sample:
        logger.info("Sample mode: will process only 5 images")
        image_files = image_files[:5]
    
    # Process images
    success = 0
    for i, image_path in enumerate(image_files):
        if i % 10 == 0:
            logger.info(f"Processing {i+1}/{len(image_files)}")
        
        if process_image(image_path, args.output):
            success += 1
    
    logger.info(f"Finished. Successfully processed {success}/{len(image_files)} images.")
    
if __name__ == "__main__":
    main() 