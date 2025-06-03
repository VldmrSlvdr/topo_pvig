#!/usr/bin/env python3
"""
Topological Feature Generator

This script generates topological features (.npy files) for images using the same preprocessing pipeline 
as used in the data loader (read_data_topo.py) to ensure alignment during training.

The script:
1. Applies the same resize/crop transformations as the test data loader
2. Generates persistence diagrams using HomCloud
3. Creates and saves topological feature maps as .npy files
"""

import os
import sys
import logging
import argparse
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('topo_feature_generator')

# Try to import HomCloud, but provide a fallback
try:
    import homcloud.interface as hc
    HAS_HOMCLOUD = True
    logger.info("HomCloud successfully imported")
except Exception as e:
    logger.warning(f"Failed to import HomCloud: {str(e)}")
    logger.warning("Will use fallback mode to generate simplified topological features")
    HAS_HOMCLOUD = False

def preprocess_image_like_dataloader(image_path):
    """
    Preprocess the image exactly like in the dataloader (read_data_topo.py)
    
    Args:
        image_path: Path to the input image
        
    Returns:
        numpy_img: Numpy array of the preprocessed image
        torch_img: Torch tensor of the preprocessed image (normalized)
    """
    # Create transforms matching the test transforms in read_data_topo.py
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # No normalization here as we need raw values for HomCloud
    ])
    
    # Open and convert image
    with open(image_path, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")
    
    # Apply transforms
    torch_img = test_transforms(img)
    
    # Convert to numpy for HomCloud (grayscale conversion by averaging RGB channels)
    numpy_img = torch_img.mean(dim=0).numpy()
    
    return numpy_img, torch_img

def generate_persistence_diagram(img_array, output_path, filtration='superlevel'):
    """Generate persistence diagram using HomCloud"""
    if not HAS_HOMCLOUD:
        logger.warning("HomCloud not available, skipping persistence diagram generation")
        return output_path
    
    # Convert image to format HomCloud expects (values between -2 and 1)
    # Scale from [0,1] to [-2,1] as a common range for HomCloud
    homcloud_img = img_array * 3 - 2
    
    # First create PD
    pd_list = hc.PDList.from_bitmap_levelset(homcloud_img, filtration, save_to=output_path)
    
    # Then create PHTree for extracting birth pixels and volumes
    hc.BitmapPHTrees.for_bitmap_levelset(homcloud_img, filtration, save_to=output_path)
    
    return output_path

def create_fallback_topological_features(img_array):
    """
    Create simplified topological features without using HomCloud
    This uses image processing to approximate topological features
    """
    logger.info("Using fallback method to generate approximate topological features")
    
    # Resize to 56x56 to match expected output size
    img_56 = transforms.Resize((56, 56))(torch.tensor(img_array).unsqueeze(0)).squeeze(0).numpy()
    
    # Create feature maps for dim0 (bright spots) and dim1 (dark spots)
    # We'll use simple thresholding to approximate these features
    
    # Dim0 - bright features (high intensity regions)
    bright_threshold = 0.65  # Threshold for bright regions
    feature_map_0 = np.zeros((56, 56))
    feature_map_0[img_56 > bright_threshold] = 1.0
    
    # Dim1 - dark features (low intensity regions)
    dark_threshold = 0.35  # Threshold for dark regions
    feature_map_1 = np.zeros((56, 56))
    feature_map_1[img_56 < dark_threshold] = 1.0
    
    # Stack the feature maps
    topo_features = np.stack([feature_map_0, feature_map_1], axis=0)
    
    return topo_features

def create_topological_feature_maps(pdgm_path, config):
    """Create topological feature maps for dimensions 0 and 1"""
    # If HomCloud is not available, use the fallback method
    if not HAS_HOMCLOUD:
        # Get the input image from config for fallback method
        # We would need to pass the image array to this function in fallback mode
        logger.warning("HomCloud not available, using fallback method")
        return None  # Will be handled by the calling function
    
    # Read the PD trees
    try:
        # Convert Path object to string for HomCloud compatibility
        pdgm_path_str = str(pdgm_path)
        phtrees_0 = hc.PDList(pdgm_path_str).bitmap_phtrees(0)
        phtrees_1 = hc.PDList(pdgm_path_str).bitmap_phtrees(1)
    except Exception as e:
        logger.error(f"Error reading persistence diagram: {e}")
        return None
    
    # Extract parameters from config
    topo_config = config.get('topo_features', {})
    lifetime_threshold = topo_config.get('lifetime_threshold', -0.5)
    
    # Filter significant features based on lifetime
    nodes_0 = [node for node in phtrees_0.nodes if node.lifetime() < lifetime_threshold and node.death_time() != -np.inf]
    nodes_1 = [node for node in phtrees_1.nodes if node.lifetime() < lifetime_threshold and node.death_time() != -np.inf]
    
    logger.debug(f"Dim 0 features: {len(nodes_0)}, Dim 1 features: {len(nodes_1)}")
    
    # Create empty feature maps - using 56x56 dimensions as expected by the model
    # The topo features are for 224x224 images but at 1/4 resolution
    feature_map_0 = np.zeros((56, 56))
    feature_map_1 = np.zeros((56, 56))
    
    # Scale factor from 224x224 to 56x56
    scale_factor = 56 / 224
    
    # Add features to the maps with scaling
    for node in nodes_0:
        if node.volume():
            weight = min(1.0, abs(node.lifetime()) / 2.0)  # Scale weight
            for pixel in node.volume():
                # HomCloud uses (x,y) coordinates where x is column, y is row
                x, y = pixel
                # Scale coordinates to 56x56
                x_scaled, y_scaled = int(x * scale_factor), int(y * scale_factor)
                # Make sure coordinates are within bounds
                # Use [y, x] indexing since numpy arrays are [row, col]
                if 0 <= y_scaled < 56 and 0 <= x_scaled < 56:
                    feature_map_0[y_scaled, x_scaled] += weight
    
    for node in nodes_1:
        if node.volume():
            weight = min(1.0, abs(node.lifetime()) / 2.0)  # Scale weight
            for pixel in node.volume():
                # HomCloud uses (x,y) coordinates where x is column, y is row
                x, y = pixel
                # Scale coordinates to 56x56
                x_scaled, y_scaled = int(x * scale_factor), int(y * scale_factor)
                # Use [y, x] indexing since numpy arrays are [row, col]
                if 0 <= y_scaled < 56 and 0 <= x_scaled < 56:
                    feature_map_1[y_scaled, x_scaled] += weight
    
    # Normalize the feature maps to [0, 1]
    if feature_map_0.max() > 0:
        feature_map_0 = feature_map_0 / feature_map_0.max()
    if feature_map_1.max() > 0:
        feature_map_1 = feature_map_1 / feature_map_1.max()
    
    # Transpose the feature maps to align with image orientation
    # This ensures the topological features align properly with the input image
    feature_map_0 = feature_map_0.T
    feature_map_1 = feature_map_1.T
    
    # Combine into a single array with shape (2, 56, 56)
    topo_features = np.stack([feature_map_0, feature_map_1], axis=0)
    
    return topo_features

def process_single_image(image_path, output_dir, config):
    """Process a single image to generate topological features"""
    try:
        # Extract base filename, label name, and relative path
        image_path = Path(image_path)
        base_filename = image_path.stem
        label_name = image_path.parent.name
        mode = image_path.parent.parent.name if image_path.parent.parent.name in ['train', 'test'] else 'unknown'
        
        # Log processing attempt
        logger.info(f"Processing image: {image_path}")
        
        # Create output directory structure
        topo_features_dir = Path(output_dir) / "topo_features" / mode / label_name
        topo_features_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {topo_features_dir}")
        
        # Persistence diagram directory for debugging/analysis
        pdgm_dir = Path(output_dir) / "persistence" / mode / label_name
        pdgm_dir.mkdir(parents=True, exist_ok=True)
        
        # Output paths
        topo_features_path = topo_features_dir / f"{base_filename}.npy"
        pdgm_path = pdgm_dir / f"{base_filename}.pdgm"
        logger.debug(f"Output paths: {topo_features_path}, {pdgm_path}")
        
        # Skip if output already exists and force_reprocess is False
        force_reprocess = config.get('force_reprocess', False)
        if not force_reprocess and topo_features_path.exists():
            logger.debug(f"Skipping {image_path} - output file already exists")
            return True
        
        # Preprocess the image like in the dataloader
        logger.debug(f"Preprocessing image: {image_path}")
        numpy_img, torch_img = preprocess_image_like_dataloader(image_path)
        logger.debug(f"Image shape after preprocessing: {numpy_img.shape}")
        
        # Create topological features - using either HomCloud or fallback
        if HAS_HOMCLOUD:
            # Generate persistence diagram
            logger.debug(f"Generating persistence diagram with HomCloud")
            filtration = config.get('filtration', 'superlevel')
            try:
                generate_persistence_diagram(numpy_img, pdgm_path, filtration)
                logger.debug(f"Persistence diagram generated: {pdgm_path}")
            except Exception as e:
                logger.error(f"Error generating persistence diagram: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
            
            # Create topological feature maps
            logger.debug(f"Creating topological feature maps using HomCloud")
            try:
                topo_features = create_topological_feature_maps(pdgm_path, config)
                logger.debug(f"Topological features created with shape: {topo_features.shape if topo_features is not None else None}")
            except Exception as e:
                logger.error(f"Error creating topological features: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return False
        else:
            # HomCloud not available - check if fallback is enabled
            if config.get('enable_fallback', False):
                logger.warning(f"HomCloud not available. Using fallback method for {image_path}")
                logger.warning("Note: Fallback method produces lower quality topological features.")
                try:
                    topo_features = create_fallback_topological_features(numpy_img)
                    logger.debug(f"Fallback topological features created with shape: {topo_features.shape}")
                except Exception as e:
                    logger.error(f"Error creating fallback topological features: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return False
            else:
                logger.error(f"HomCloud not available. Cannot generate topological features for {image_path}")
                logger.error("Install HomCloud to enable topological feature generation.")
                logger.error("Alternatively, use --enable-fallback to use simplified features (not recommended).")
                return False
        
        if topo_features is None:
            logger.error(f"Failed to create topological features for {image_path}")
            return False
        
        # Check dimensions
        if topo_features.shape != (2, 56, 56):
            logger.error(f"Unexpected topo features shape: {topo_features.shape}")
            return False
        
        # Save topological features
        try:
            np.save(topo_features_path, topo_features)
            logger.debug(f"Saved topological features to {topo_features_path}")
        except Exception as e:
            logger.error(f"Error saving topological features: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        
        logger.info(f"Successfully processed {image_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_directory(config):
    """Process all images in the specified directory"""
    input_dir = config.get('input_dir')
    output_dir = config.get('output_dir')
    
    # Check if fallback is enabled and warn user
    if config.get('enable_fallback', False) and not HAS_HOMCLOUD:
        logger.warning("=" * 60)
        logger.warning("WARNING: Using fallback topological feature generation!")
        logger.warning("This produces simplified, lower-quality features.")
        logger.warning("For best results, install HomCloud.")
        logger.warning("=" * 60)
    
    if not input_dir or not output_dir:
        logger.error("input_dir and output_dir must be specified in config")
        return 0
    
    # Extract parameters from config
    file_pattern = config.get('file_pattern', '*.jpg,*.png')
    recursive = config.get('recursive', True)
    
    logger.info(f"Processing directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"File pattern: {file_pattern}, Recursive: {recursive}")
    
    # Get file extensions to match
    patterns = file_pattern.split(',')
    extensions = [p.strip().replace('*', '') for p in patterns]
    
    # Find all images
    image_files = []
    input_dir_path = Path(input_dir)
    
    if recursive:
        for ext in extensions:
            image_files.extend(list(input_dir_path.glob(f"**/*{ext}")))
    else:
        for ext in extensions:
            image_files.extend(list(input_dir_path.glob(f"*{ext}")))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process a few images first for debugging
    debug_limit = min(5, len(image_files))
    logger.info(f"Processing first {debug_limit} images for debugging...")
    for image_path in image_files[:debug_limit]:
        result = process_single_image(image_path, output_dir, config)
        logger.info(f"Debug process result for {image_path}: {result}")
    
    # Process the rest of the images with progress bar
    success_count = 0
    for image_path in tqdm(image_files, desc="Generating topological features"):
        if process_single_image(image_path, output_dir, config):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count} out of {len(image_files)} images")
    logger.info(f"Features saved to: {os.path.join(output_dir, 'topo_features')}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Generate topological features for images")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--enable-fallback", action="store_true", 
                       help="Enable fallback method when HomCloud is not available (produces lower quality features)")
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Add fallback setting to config
        config['enable_fallback'] = args.enable_fallback
        logger.info(f"Loaded configuration from {args.config}")
        logger.info(f"Fallback method: {'enabled' if args.enable_fallback else 'disabled'}")
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return 1
    
    # Process the directory
    process_directory(config)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 