#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from skimage import exposure, morphology, filters, feature
from scipy import ndimage
import homcloud.interface as hc
from skimage.transform import resize
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('topo_importance_processor')

def preprocess_medical_image(image_path, config):
    """Apply the combined preprocessing method to a medical image with standardized dimensions"""
    # Extract standardization parameters
    preproc_config = config.get('preprocessing', {})
    target_width = preproc_config.get('target_width', 512)  # Default to 512x512
    target_height = preproc_config.get('target_height', 512)
    invert = preproc_config.get('invert', True)
    clahe_clip_limit = preproc_config.get('clahe_clip_limit', 0.03)
    structure_sigma = preproc_config.get('structure_sigma', 1)
    structure_weight = preproc_config.get('structure_weight', 0.3)
    
    # Load image and get original dimensions
    pil_img = Image.open(image_path).convert('L')
    orig_width, orig_height = pil_img.size
    
    # Resize to standard dimensions
    resized_img = pil_img.resize((target_width, target_height), Image.LANCZOS)
    
    # Convert to numpy array
    img = np.array(resized_img) / 255.0
    
    # Store original and new dimensions for reference
    dimensions = {
        'original': (orig_width, orig_height),
        'resized': (target_width, target_height),
        'scale_x': target_width / orig_width,
        'scale_y': target_height / orig_height
    }
    
    # 1. Invert the image (chest X-rays have dark lungs)
    if invert:
        img = 1.0 - img
    
    # 2. Apply contrast enhancement with CLAHE
    img = exposure.equalize_adapthist(img, clip_limit=clahe_clip_limit)
    
    # 3. Apply vessel/structure enhancement
    img_structure = ndimage.gaussian_gradient_magnitude(img, sigma=structure_sigma)
    # Normalize the structure enhancement
    img_structure = (img_structure - img_structure.min()) / (img_structure.max() - img_structure.min())
    
    # 4. Combine for enhanced medical image
    processed = (1 - structure_weight) * img + structure_weight * img_structure
    
    # 5. Convert to the format HomCloud expects
    homcloud_img = processed * 3 - 2
    
    return processed, homcloud_img, dimensions

def generate_persistence_diagram(img_array, output_path, filtration='superlevel'):
    """Generate persistence diagram using HomCloud"""
    # First create PD
    pd_list = hc.PDList.from_bitmap_levelset(img_array, filtration, save_to=output_path)
    
    # Then create PHTree for extracting birth pixels and volumes
    hc.BitmapPHTrees.for_bitmap_levelset(img_array, filtration, save_to=output_path)
    
    return output_path

def create_topological_importance_heatmap(img_array, pdgm_path, config):
    """Create heatmaps showing topologically important regions"""
    # Extract parameters
    topo_config = config.get('topo_importance', {})
    lifetime_threshold = topo_config.get('lifetime_threshold', -0.5)
    sigma = topo_config.get('sigma', 15)
    
    # Read the PD trees
    phtrees_0 = hc.PDList(pdgm_path).bitmap_phtrees(0)
    phtrees_1 = hc.PDList(pdgm_path).bitmap_phtrees(1)
    
    # Extract significant nodes
    nodes_0 = [node for node in phtrees_0.nodes if node.lifetime() < lifetime_threshold and node.death_time() != -np.inf]
    nodes_1 = [node for node in phtrees_1.nodes if node.lifetime() < lifetime_threshold and node.death_time() != -np.inf]
    
    logger.debug(f"Dim 0 features: {len(nodes_0)}, Dim 1 features: {len(nodes_1)}")
    
    # Get image dimensions
    height, width = img_array.shape
    
    # Create empty heatmaps with original dimensions
    heatmap_0 = np.zeros((height, width))
    heatmap_1 = np.zeros((height, width))
    
    # Add volumes to the heatmaps directly
    for node in nodes_0:
        if node.volume():
            weight = min(1.0, abs(node.lifetime()) / 2.0)  # Scale weight
            for pixel in node.volume():
                # HomCloud uses (x,y) coordinates
                x, y = pixel
                # Make sure coordinates are within bounds
                if 0 <= y < height and 0 <= x < width:
                    heatmap_0[y, x] += weight
    
    for node in nodes_1:
        if node.volume():
            weight = min(1.0, abs(node.lifetime()) / 2.0)  # Scale weight
            for pixel in node.volume():
                x, y = pixel
                if 0 <= y < height and 0 <= x < width:
                    heatmap_1[y, x] += weight
    
    # Apply Gaussian blur to create smooth heatmaps
    heatmap_0_smooth = ndimage.gaussian_filter(heatmap_0, sigma=sigma)
    heatmap_1_smooth = ndimage.gaussian_filter(heatmap_1, sigma=sigma)
    
    # Normalize the heatmaps
    if heatmap_0_smooth.max() > 0:
        heatmap_0_smooth = heatmap_0_smooth / heatmap_0_smooth.max()
    if heatmap_1_smooth.max() > 0:
        heatmap_1_smooth = heatmap_1_smooth / heatmap_1_smooth.max()
    
    # Create combined heatmap (RGB)
    combined_heatmap = np.zeros((height, width, 3))
    combined_heatmap[:, :, 0] = heatmap_0_smooth  # Red channel for dim 0
    combined_heatmap[:, :, 2] = heatmap_1_smooth  # Blue channel for dim 1
    
    # Just transpose (flip along main diagonal) without horizontal flip
    heatmap_0_fixed = heatmap_0_smooth.T
    heatmap_1_fixed = heatmap_1_smooth.T
    combined_fixed = np.transpose(combined_heatmap, (1, 0, 2))
    
    # Return correctly oriented heatmaps and nodes
    return {
        'dim0': heatmap_0_fixed,
        'dim1': heatmap_1_fixed,
        'combined': combined_fixed,
        'nodes_0': nodes_0,
        'nodes_1': nodes_1
    }

def create_importance_visualizations(original_img, heatmap_data, output_paths, config):
    """Create and save various visualizations of topological importance"""
    # Extract parameters
    topo_config = config.get('topo_importance', {})
    alpha = topo_config.get('alpha', 0.7)
    
    # Extract heatmaps
    heatmap_0 = heatmap_data['dim0']
    heatmap_1 = heatmap_data['dim1']
    combined_heatmap = heatmap_data['combined']
    nodes_0 = heatmap_data['nodes_0']
    nodes_1 = heatmap_data['nodes_1']
    
    # Create background image for overlay
    if isinstance(original_img, np.ndarray) and original_img.ndim == 2:
        # Convert grayscale to RGB
        img_rgb = np.stack([original_img] * 3, axis=2) * 255
    else:
        img_rgb = np.array(original_img)
    
    # Check dimensions and ensure they match
    h_img, w_img = img_rgb.shape[:2]
    h_heat, w_heat = combined_heatmap.shape[:2]
    
    if h_img != h_heat or w_img != w_heat:
        logger.warning(f"Dimension mismatch: img({h_img},{w_img}) vs heatmap({h_heat},{w_heat})")
        # Resize heatmap to match image dimensions
        combined_heatmap = resize(combined_heatmap, (h_img, w_img, 3), 
                               anti_aliasing=True, preserve_range=True)
        heatmap_0 = resize(heatmap_0, (h_img, w_img), 
                        anti_aliasing=True, preserve_range=True)
        heatmap_1 = resize(heatmap_1, (h_img, w_img), 
                        anti_aliasing=True, preserve_range=True)
        logger.info(f"Resized heatmap to match image dimensions: ({h_img},{w_img})")
    
    # Create overlay
    overlay = (1 - alpha) * img_rgb + alpha * 255 * combined_heatmap
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # Make a copy for critical points visualization
    overlay_points = overlay.copy()
    
    # Plot critical points directly on the overlay image copy
    for node in nodes_0:
        birth_pixel = node.birth_pixel()
        if birth_pixel:
            x, y = birth_pixel
            # Just transpose: (x,y) -> (y,x)
            nx = y
            ny = x
            # Only draw if within bounds
            if 0 <= nx < w_img and 0 <= ny < h_img:
                # Draw a small red square (3x3 pixels)
                radius = 1
                y_min, y_max = max(0, ny-radius), min(h_img, ny+radius+1)
                x_min, x_max = max(0, nx-radius), min(w_img, nx+radius+1)
                overlay_points[y_min:y_max, x_min:x_max] = [255, 0, 0]  # Red
    
    for node in nodes_1:
        birth_pixel = node.birth_pixel()
        if birth_pixel:
            x, y = birth_pixel
            # Just transpose: (x,y) -> (y,x)
            nx = y
            ny = x
            if 0 <= nx < w_img and 0 <= ny < h_img:
                # Draw a small blue square
                radius = 1
                y_min, y_max = max(0, ny-radius), min(h_img, ny+radius+1)
                x_min, x_max = max(0, nx-radius), min(w_img, nx+radius+1)
                overlay_points[y_min:y_max, x_min:x_max] = [0, 0, 255]  # Blue
    
    # Save the overlay with critical points directly
    Image.fromarray(overlay_points).save(os.path.splitext(output_paths['overlay'])[0] + "_points.jpg")
    
    # Save individual heatmaps
    plt.imsave(output_paths['dim0'], heatmap_0, cmap='hot')
    plt.imsave(output_paths['dim1'], heatmap_1, cmap='cool')
    
    # Save combined heatmap
    plt.imsave(output_paths['combined'], combined_heatmap)
    
    # Save overlay on original
    Image.fromarray(overlay).save(output_paths['overlay'])
    
    # Create comprehensive visualization with multiple views
    plt.figure(figsize=(20, 15))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Dimension 0 heatmap
    plt.subplot(2, 3, 2)
    plt.imshow(heatmap_0, cmap='hot')
    plt.title('Dimension 0 Importance\n(Connected Components)')
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    
    # Dimension 1 heatmap
    plt.subplot(2, 3, 3)
    plt.imshow(heatmap_1, cmap='cool')
    plt.title('Dimension 1 Importance\n(Loops/Holes)')
    plt.axis('off')
    plt.colorbar(shrink=0.7)
    
    # Combined heatmap only
    plt.subplot(2, 3, 4)
    plt.imshow(combined_heatmap)
    plt.title('Combined Topological Importance\n(Red=D0, Blue=D1, Purple=Both)')
    plt.axis('off')
    
    # Final overlay
    plt.subplot(2, 3, 5)
    plt.imshow(overlay)
    plt.title('Topological Importance Overlay')
    plt.axis('off')
    
    # Final overlay with critical points
    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title('Topological Critical Points')

    # Get image dimensions
    height, width = original_img.shape if isinstance(original_img, np.ndarray) else original_img.size[::-1]

    # Plot birth points for dim 0 (red)
    for node in nodes_0:
        birth_pixel = node.birth_pixel()
        if birth_pixel:
            x, y = birth_pixel
            # Just transpose: (x,y) -> (y,x)
            nx = y
            ny = x
            if 0 <= nx < width and 0 <= ny < height:
                plt.plot(nx, ny, 'ro', markersize=3, alpha=0.7)

    # Same for dim 1 features (blue)
    for node in nodes_1:
        birth_pixel = node.birth_pixel()
        if birth_pixel:
            x, y = birth_pixel
            # Just transpose: (x,y) -> (y,x)
            nx = y
            ny = x
            if 0 <= nx < width and 0 <= ny < height:
                plt.plot(nx, ny, 'bo', markersize=3, alpha=0.7)

    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_paths['summary'], dpi=300)
    plt.close()
    
    return True

def outputs_exist(image_path, output_dirs, config):
    """Check if output files already exist for this image"""
    # Get base filename and relative path
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    rel_path = os.path.dirname(os.path.relpath(image_path, config.get('input_dir', '.')))
    
    # Check if all expected output files exist
    expected_files = [
        os.path.join(output_dirs['preprocessed'], rel_path, f"{base_filename}.jpg"),
        os.path.join(output_dirs['heatmap_dim0'], rel_path, f"{base_filename}.jpg"),
        os.path.join(output_dirs['heatmap_dim1'], rel_path, f"{base_filename}.jpg"),
        os.path.join(output_dirs['heatmap_combined'], rel_path, f"{base_filename}.jpg"),
        os.path.join(output_dirs['overlay'], rel_path, f"{base_filename}.jpg"),
        os.path.join(output_dirs['summary'], rel_path, f"{base_filename}_summary.jpg")
    ]
    
    for file_path in expected_files:
        if not os.path.exists(file_path):
            return False
    
    return True

def process_single_image(image_path, output_dirs, config):
    """Process a single image to create topological importance visualizations"""
    try:
        # Check if force_reprocess flag is set in config
        force_reprocess = config.get('force_reprocess', False)
        
        # Skip if output files already exist and force_reprocess is not True
        if not force_reprocess and outputs_exist(image_path, output_dirs, config):
            logger.info(f"Skipping {image_path} - output files already exist")
            return True
            
        # Get base filename and relative path
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Get relative path from input directory to the image file
        rel_path = os.path.dirname(os.path.relpath(image_path, config.get('input_dir', '.')))
        
        # Create subdirectories for this relative path in all output dirs
        for dir_type, dir_path in output_dirs.items():
            rel_output_dir = os.path.join(dir_path, rel_path)
            os.makedirs(rel_output_dir, exist_ok=True)
        
        # Preprocess the image with standardized dimensions
        processed_img, homcloud_img, dimensions = preprocess_medical_image(image_path, config)
        
        # Get target dimensions from config for consistency
        preproc_config = config.get('preprocessing', {})
        target_width = preproc_config.get('target_width', 512)
        target_height = preproc_config.get('target_height', 512)
        
        # Load and resize original image to match standardized dimensions
        original_img = Image.open(image_path).convert('L')
        original_img = original_img.resize((target_width, target_height), Image.LANCZOS)
        original_np = np.array(original_img) / 255.0
        
        # Save preprocessed image
        preprocessed_path = os.path.join(output_dirs['preprocessed'], rel_path, f"{base_filename}.jpg")
        Image.fromarray((processed_img * 255).astype(np.uint8)).save(preprocessed_path)
        
        # Save dimensions information for reference
        dimensions_path = os.path.join(output_dirs['preprocessed'], rel_path, f"{base_filename}_dimensions.json")
        with open(dimensions_path, 'w') as f:
            json.dump(dimensions, f)
        
        # Generate persistence diagram
        pdgm_path = os.path.join(output_dirs['persistence'], rel_path, f"{base_filename}.pdgm")
        os.makedirs(os.path.dirname(pdgm_path), exist_ok=True)
        
        # Use the filtration type from config
        filtration = config.get('filtration', 'superlevel')
        generate_persistence_diagram(homcloud_img, pdgm_path, filtration)
        
        # Create topological importance heatmaps
        heatmap_data = create_topological_importance_heatmap(homcloud_img, pdgm_path, config)
        
        # Define output paths for visualizations
        visualization_paths = {
            'dim0': os.path.join(output_dirs['heatmap_dim0'], rel_path, f"{base_filename}.jpg"),
            'dim1': os.path.join(output_dirs['heatmap_dim1'], rel_path, f"{base_filename}.jpg"),
            'combined': os.path.join(output_dirs['heatmap_combined'], rel_path, f"{base_filename}.jpg"),
            'overlay': os.path.join(output_dirs['overlay'], rel_path, f"{base_filename}.jpg"),
            'summary': os.path.join(output_dirs['summary'], rel_path, f"{base_filename}_summary.jpg")
        }
        
        # Create and save visualizations using the standardized original image
        create_importance_visualizations(
            original_np,
            heatmap_data,
            visualization_paths,
            config
        )
        
        return True
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def process_directory(config):
    """Process all images in a directory based on configuration"""
    # Extract parameters from config
    input_dir = config.get('input_dir', './input')
    output_dir = config.get('output_dir', './output')
    file_pattern = config.get('file_pattern', '*.jpg,*.png')
    recursive = config.get('recursive', True)
    
    # Create output directory structure
    output_dirs = {
        'preprocessed': os.path.join(output_dir, 'preprocessed'),
        'persistence': os.path.join(output_dir, 'persistence'),
        'heatmap_dim0': os.path.join(output_dir, 'heatmap_dim0'),
        'heatmap_dim1': os.path.join(output_dir, 'heatmap_dim1'),
        'heatmap_combined': os.path.join(output_dir, 'heatmap_combined'),
        'overlay': os.path.join(output_dir, 'overlay'),
        'summary': os.path.join(output_dir, 'summary')
    }
    
    # Create base directories
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # Get file extensions to match
    patterns = file_pattern.split(',')
    extensions = [p.strip().replace('*', '') for p in patterns]
    
    # Find all images recursively using os.walk if recursive is True
    image_files = []
    
    if recursive:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in extensions):
                    image_files.append(os.path.join(root, file))
    else:
        # Non-recursive mode - just get files in the input directory
        for pattern in patterns:
            import glob
            image_files.extend(glob.glob(os.path.join(input_dir, pattern.strip())))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image with progress bar
    success_count = 0
    for image_path in tqdm(image_files, desc="Processing images"):
        if process_single_image(image_path, output_dirs, config):
            success_count += 1
    
    logger.info(f"Successfully processed {success_count} out of {len(image_files)} images")
    logger.info(f"Results saved to: {output_dir}")
    
    return success_count

def main():
    parser = argparse.ArgumentParser(description="Process a directory of medical images to find topologically important regions")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Enable debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return 1
    
    # Process the directory
    process_directory(config)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 