#%%
"""
Visual Test Time Augmentation Testing Script

This script loads the FDDB-trained model from HuggingFace and visualizes the TTA transforms
applied to test images WITHOUT showing predictions. This is useful for understanding
how each transformation affects the input images.
"""

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
import numpy as np
from ellipse_rcnn.hf import EllipseRCNN
from tta.tta_transforms import TTA_TRANSFORMS, tta_predict
from ellipse_rcnn.data.fddb import FDDB
import os
import random

#%%
def load_fddb_model():
    """Load the FDDB-trained model from HuggingFace repository."""
    try:
        model = EllipseRCNN.from_pretrained("MJGT/ellipse-rcnn-FDDB")
        model.eval()
        print("✅ Successfully loaded FDDB model from HuggingFace")
        return model
    except Exception as e:
        print(f"❌ Error loading model from HuggingFace: {e}")
        print("Make sure the model is uploaded and accessible at MJGT/ellipse-rcnn-FDDB")
        return None

#%%
def visualize_tta_transforms(image_tensor, title_prefix=""):
    """
    Visualize all TTA transformations applied to an image without predictions.
    
    Args:
        image_tensor: Input image tensor [C, H, W]
        title_prefix: Prefix for the plot title
    """
    # Create subplot grid for all transforms
    n_transforms = len(TTA_TRANSFORMS)
    cols = 3
    rows = (n_transforms + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle(f'{title_prefix}TTA Transform Visualization (Images Only)', fontsize=16)
    
    # Flatten axes for easier indexing
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, transform_config in enumerate(TTA_TRANSFORMS):
        forward_func, forward_kwargs = transform_config['forward']
        transform_name = transform_config['name']
        
        # Apply forward transformation
        transformed_img = forward_func(image_tensor, **forward_kwargs)
        
        # Convert to numpy for visualization
        if transformed_img.dim() == 3:
            # Convert from [C, H, W] to [H, W, C] for matplotlib
            img_np = transformed_img.permute(1, 2, 0).cpu().numpy()
            if img_np.shape[2] == 1:  # Grayscale
                img_np = img_np.squeeze(2)
        else:
            img_np = transformed_img.cpu().numpy()
        
        # Plot the transformed image
        ax = axes[i]
        if len(img_np.shape) == 2:  # Grayscale
            ax.imshow(img_np, cmap='gray')
        else:  # RGB
            ax.imshow(img_np)
        
        ax.set_title(f'{transform_name}', fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Add a colored border to match the transform color from TTA_TRANSFORMS
        for spine in ax.spines.values():
            spine.set_edgecolor(transform_config['color'])
            spine.set_linewidth(3)
    
    # Hide unused subplots
    for i in range(n_transforms, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

#%%
def test_with_sample_images():
    """Test TTA visualization with sample images from the docs folder."""
    
    # Load model (for completeness, though we won't use predictions)
    model = load_fddb_model()
    
    # List of test images
    test_images = [
        # ("Craters Sample", r"..\docs\example_craters.png"),
        # ("Face Sample", r"..\docs\friends.jpg"),
        # ("FDDB Sample", r"..\docs\fddb_sample.png"),
        ("Friends Sample", r"..\docs\friends.jpg")
    ]
    
    for img_name, img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n🖼️  Processing: {img_name}")
            
            # Load and preprocess image
            try:
                img = Image.open(img_path).convert("L")  # Convert to grayscale
                img_tensor = to_tensor(img)
                
                print(f"   Image shape: {img_tensor.shape}")
                print(f"   Image size: {img.size}")
                
                # Visualize TTA transforms
                visualize_tta_transforms(img_tensor, f"{img_name} - ")
                
            except Exception as e:
                print(f"   ❌ Error processing {img_name}: {e}")
        else:
            print(f"⚠️  Image not found: {img_path}")

#%%
def test_with_custom_image():
    """Test TTA visualization with a custom image path."""
    
    # You can modify this path to test with your own images
    custom_image_path = r"..\docs\friends.jpg"  # Change this to your image path
    
    if not os.path.exists(custom_image_path):
        print(f"❌ Custom image not found at: {custom_image_path}")
        print("Please update the custom_image_path variable with a valid image path")
        return
    
    print(f"🖼️  Testing with custom image: {custom_image_path}")
    
    try:
        # Load and preprocess image
        img = Image.open(custom_image_path).convert("L")
        img_tensor = to_tensor(img)
        
        print(f"Image shape: {img_tensor.shape}")
        print(f"Image size: {img.size}")
        
        # Visualize TTA transforms
        visualize_tta_transforms(img_tensor, "Custom Image - ")
        
    except Exception as e:
        print(f"❌ Error processing custom image: {e}")

#%%
def compare_transform_effects():
    """
    Create a detailed comparison showing the effects of different transform categories.
    """
    
    # Load a test image
    test_image_path = r"..\docs\friends.jpg"
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    img = Image.open(test_image_path).convert("L")
    img_tensor = to_tensor(img)
    
    # Group transforms by category
    transform_categories = {
        'Photometric': ['Original', 'Gamma 0.7', 'Gamma 1.3', 'Brightness/Contrast'],
        'Geometric': ['Original', 'Rotation +10°', 'Rotation -10°', 'Horizontal Flip']
    }
    
    for category, transform_names in transform_categories.items():
        print(f"\n📊 Showing {category} Transforms")
        
        # Filter transforms for this category
        category_transforms = [t for t in TTA_TRANSFORMS if t['name'] in transform_names]
        
        # Create subplot for category
        fig, axes = plt.subplots(1, len(category_transforms), figsize=(4*len(category_transforms), 4))
        fig.suptitle(f'{category} Transform Effects', fontsize=16)
        
        if len(category_transforms) == 1:
            axes = [axes]
        
        for i, transform_config in enumerate(category_transforms):
            forward_func, forward_kwargs = transform_config['forward']
            transform_name = transform_config['name']
            
            # Apply transformation
            transformed_img = forward_func(img_tensor, **forward_kwargs)
            img_np = transformed_img.permute(1, 2, 0).squeeze().cpu().numpy()
            
            # Plot
            axes[i].imshow(img_np, cmap='gray')
            axes[i].set_title(transform_name, fontweight='bold')
            axes[i].axis('off')
            
            # Add colored border
            for spine in axes[i].spines.values():
                spine.set_edgecolor(transform_config['color'])
                spine.set_linewidth(3)
        
        plt.tight_layout()
        plt.show()

#%%
def print_transform_info():
    """Print detailed information about available TTA transforms."""
    
    print("🔧 Available TTA Transforms:")
    print("=" * 50)
    
    for i, transform_config in enumerate(TTA_TRANSFORMS, 1):
        name = transform_config['name']
        color = transform_config['color']
        forward_func, forward_kwargs = transform_config['forward']
        
        print(f"{i}. {name}")
        print(f"   Color: {color}")
        print(f"   Function: {forward_func.__name__}")
        if forward_kwargs:
            print(f"   Parameters: {forward_kwargs}")
        print()

#%%
# Main execution
if __name__ == "__main__":
    print("🚀 Starting Visual TTA Testing")
    print("=" * 50)
    
    # Print available transforms
    print_transform_info()
    
    # Test with sample images
    test_with_sample_images()
    
    # Test with custom image
    test_with_custom_image()
    
    # Compare transform effects by category
    compare_transform_effects()
    
    print("\n✅ Visual TTA testing completed!")

#%%
# Interactive cell for testing specific images
def interactive_test():
    """
    Interactive cell for testing TTA visualization with any image.
    Modify the image_path variable below to test with different images.
    """
    
    # 🔧 MODIFY THIS PATH TO TEST WITH YOUR OWN IMAGE
    image_path = r"..\docs\example_craters.png"  # <-- Change this path
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please update the image_path variable above")
        return
    
    print(f"🖼️  Loading image: {image_path}")
    
    try:
        # Load image
        img = Image.open(image_path).convert("L")
        img_tensor = to_tensor(img)
        
        print(f"✅ Image loaded successfully!")
        print(f"   Shape: {img_tensor.shape}")
        print(f"   Size: {img.size}")
        print(f"   Data type: {img_tensor.dtype}")
        print(f"   Value range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        
        # Show original image first
        plt.figure(figsize=(8, 6))
        plt.imshow(img_tensor.permute(1, 2, 0).squeeze().cpu().numpy(), cmap='gray')
        plt.title("Original Image", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.show()
        
        # Show all TTA transforms
        visualize_tta_transforms(img_tensor, "Interactive Test - ")
        
    except Exception as e:
        print(f"❌ Error: {e}")

# Run interactive test
interactive_test()

#%%
def plot_ellipse(ax, ellipse_params, color, alpha=0.7, linewidth=2, label=None):
    """
    Plot an ellipse on the given axes.
    
    Args:
        ax: Matplotlib axes
        ellipse_params: Tensor [a, b, x, y, theta] or numpy array
        color: Color for the ellipse
        alpha: Transparency
        linewidth: Line width
        label: Label for legend
    """
    if torch.is_tensor(ellipse_params):
        ellipse_params = ellipse_params.cpu().numpy()
    
    a, b, x, y, theta = ellipse_params
    
    # Generate ellipse points
    t = np.linspace(0, 2*np.pi, 100)
    ellipse_x = a * np.cos(t)
    ellipse_y = b * np.sin(t)
    
    # Rotate ellipse
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    rotated_x = ellipse_x * cos_theta - ellipse_y * sin_theta + x
    rotated_y = ellipse_x * sin_theta + ellipse_y * cos_theta + y
    
    ax.plot(rotated_x, rotated_y, color=color, alpha=alpha, linewidth=linewidth, label=label)

#%%
def compare_base_vs_tta_predictions(data_path="../data/FDDB", min_score=0.5, num_images=4, custom_image_path=None): # "../data/FDDB/images/2003/01/14/big/img_719.jpg"
    """
    Compare base model predictions vs TTA predictions on random FDDB images or a custom image.
    
    Args:
        data_path: Path to FDDB dataset (ignored if custom_image_path is provided)
        min_score: Minimum confidence threshold
        num_images: Number of images to test (ignored if custom_image_path is provided)
        custom_image_path: Path to a custom image file. If provided, only this image will be tested
    """
    
    print("🔍 Comparing Base Model vs TTA Predictions")
    print("=" * 50)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_fddb_model()
    if model is None:
        return
    
    model.to(device)
    
    # Handle custom image path
    if custom_image_path is not None:
        print(f"🖼️ Testing custom image: {custom_image_path}")
        
        if not os.path.exists(custom_image_path):
            print(f"❌ Custom image not found: {custom_image_path}")
            return
        
        try:
            # Load and preprocess the custom image
            from PIL import Image
            from torchvision.transforms.functional import to_tensor
            
            img = Image.open(custom_image_path).convert("L")  # Convert to grayscale
            image_tensor = to_tensor(img).to(device)
            
            # Create a dummy target (no ground truth for custom images)
            target = {"ellipse_params": torch.empty((0, 5))}
            
            selected_images = [(0, image_tensor, target)]
            num_images = 1
            
            print(f"✅ Custom image loaded: {img.size} -> {image_tensor.shape}")
            
        except Exception as e:
            print(f"❌ Error loading custom image: {e}")
            return
    else:
        # Load FDDB dataset for random images
        try:
            dataset = FDDB(data_path)
            print(f"✅ FDDB dataset loaded with {len(dataset)} images")
        except Exception as e:
            print(f"❌ Error loading FDDB dataset: {e}")
            return
        
        # Select random images with ground truth ellipses
        selected_images = []
        max_attempts = 50
        
        for attempt in range(max_attempts):
            if len(selected_images) >= num_images:
                break
                
            idx = random.randint(0, len(dataset) - 1)
            image_tensor, target = dataset[idx]
            
            if target["ellipse_params"].numel() > 0:  # Has ground truth ellipses
                selected_images.append((idx, image_tensor, target))
                print(f"📸 Selected image {len(selected_images)}: index {idx} ({len(target['ellipse_params'])} GT ellipses)")
        
        if len(selected_images) < num_images:
            print(f"⚠️ Could only find {len(selected_images)} images with ground truth ellipses")
            num_images = len(selected_images)
        
        if num_images == 0:
            print("❌ Could not find any images with ground truth ellipses")
            return
    
    # Create subplot grid
    cols = 2
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 * rows))
    
    # Set appropriate title
    if custom_image_path is not None:
        title = f'Base vs TTA Predictions - Custom Image'
    else:
        title = 'Base vs TTA Predictions Comparison - FDDB Dataset'
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    else:
        axes = axes.flatten()
    
    # Process each selected image
    for img_idx, (dataset_idx, image_tensor, target) in enumerate(selected_images):
        ax = axes[img_idx]
        
        print(f"\n🔄 Processing image {img_idx + 1}/{num_images} (dataset index: {dataset_idx})")
        
        image_tensor = image_tensor.to(device)
        target_ellipses = target["ellipse_params"]
        
        # Get base model prediction
        with torch.no_grad():
            base_pred = model([image_tensor])
            base_prediction = base_pred[0]
        
        # Filter base predictions by score
        if base_prediction["ellipse_params"].numel() > 0:
            base_mask = base_prediction["scores"] > min_score
            base_ellipses = base_prediction["ellipse_params"][base_mask]
            base_scores = base_prediction["scores"][base_mask]
        else:
            base_ellipses = torch.empty((0, 5))
            base_scores = torch.empty((0,))
        
        # Get individual TTA predictions (without consensuation)
        individual_predictions = []
        original_H, original_W = image_tensor.shape[1:]
        
        for transform_config in TTA_TRANSFORMS:
            forward_func, forward_kwargs = transform_config['forward']
            reverse_func, reverse_kwargs = transform_config['reverse']
            
            # Apply forward transform
            transformed_img = forward_func(image_tensor, **forward_kwargs)
            
            # Get prediction on transformed image
            with torch.no_grad():
                pred = model([transformed_img])
                prediction = pred[0]
            
            # Reverse transform predictions back to original space
            if prediction["ellipse_params"].numel() > 0:
                reversed_pred = reverse_func(prediction, original_H, original_W, **reverse_kwargs)
                
                # Filter by score
                score_mask = reversed_pred["scores"] > min_score
                filtered_ellipses = reversed_pred["ellipse_params"][score_mask]
                filtered_scores = reversed_pred["scores"][score_mask]
            else:
                filtered_ellipses = torch.empty((0, 5))
                filtered_scores = torch.empty((0,))
            
            individual_predictions.append({
                'name': transform_config['name'],
                'ellipses': filtered_ellipses,
                'scores': filtered_scores,
                'color': transform_config['color']
            })
        
        # Get final TTA prediction (with consensuation)
        tta_predictions = tta_predict(
            model=model,
            image_tensor=image_tensor,
            device=device,
            min_score=min_score,
            consensuate=True,
            visualize=False
        )
        tta_prediction = tta_predictions[0]
        
        # Filter TTA predictions by score
        if tta_prediction["ellipse_params"].numel() > 0:
            tta_mask = tta_prediction["scores"] > min_score
            tta_ellipses = tta_prediction["ellipse_params"][tta_mask]
            tta_scores = tta_prediction["scores"][tta_mask]
        else:
            tta_ellipses = torch.empty((0, 5))
            tta_scores = torch.empty((0,))
        
        # Convert image for display - handle both grayscale and RGB
        img_display = image_tensor.cpu().squeeze()
        if img_display.dim() == 3:  # RGB image [C, H, W]
            if img_display.shape[0] == 3:  # RGB
                img_display = img_display.permute(1, 2, 0).numpy()
                ax.imshow(img_display)
            elif img_display.shape[0] == 1:  # Single channel
                img_display = img_display.squeeze(0).numpy()
                ax.imshow(img_display, cmap='gray')
        else:  # Already 2D grayscale
            img_display = img_display.numpy()
            ax.imshow(img_display, cmap='gray')
        
        # Set appropriate title based on image source
        if custom_image_path is not None:
            image_name = os.path.basename(custom_image_path)
            ax.set_title(f'Custom Image: {image_name}\n'
                        f'GT: {len(target_ellipses)}, Base: {len(base_ellipses)}, '
                        f'Indiv: {sum(len(p["ellipses"]) for p in individual_predictions)}, '
                        f'TTA: {len(tta_ellipses)}', 
                        fontsize=11, fontweight='bold')
        else:
            ax.set_title(f'Image {img_idx + 1} (idx: {dataset_idx})\n'
                        f'GT: {len(target_ellipses)}, Base: {len(base_ellipses)}, '
                        f'Indiv: {sum(len(p["ellipses"]) for p in individual_predictions)}, '
                        f'TTA: {len(tta_ellipses)}', 
                        fontsize=11, fontweight='bold')
        
        # Plot ground truth ellipses (thick green)
        for i, gt_ellipse in enumerate(target_ellipses):
            plot_ellipse(ax, gt_ellipse, 'green', alpha=0.9, linewidth=3, 
                        label='Ground Truth' if img_idx == 0 and i == 0 else None)
        
        # Plot base model predictions (blue)
        for i, ellipse in enumerate(base_ellipses):
            plot_ellipse(ax, ellipse, 'blue', alpha=0.8, linewidth=2,
                        label='Base Model' if img_idx == 0 and i == 0 else None)
        
        # Plot individual TTA predictions (faded colors by transform)
        legend_added = set()
        for pred_data in individual_predictions:
            for i, ellipse in enumerate(pred_data['ellipses']):
                label = f"{pred_data['name']} (Indiv)" if (img_idx == 0 and 
                        pred_data['name'] not in legend_added) else None
                if label:
                    legend_added.add(pred_data['name'])
                plot_ellipse(ax, ellipse, pred_data['color'], alpha=0.3, linewidth=1, label=label)
        
        # Plot final TTA consensuated predictions (thick red)
        for i, ellipse in enumerate(tta_ellipses):
            plot_ellipse(ax, ellipse, 'red', alpha=0.9, linewidth=2,
                        label='TTA Consensuated' if img_idx == 0 and i == 0 else None)
        
        ax.axis('off')
        
        # Print statistics for this image
        print(f"   GT: {len(target_ellipses)}, Base: {len(base_ellipses)}, "
              f"Individual TTA: {sum(len(p['ellipses']) for p in individual_predictions)}, "
              f"TTA Final: {len(tta_ellipses)}")
    
    # Hide unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')
    
    # Add a single legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:  # Only add legend if there are any handles
        # Sort legend to put main categories first
        main_categories = ['Ground Truth', 'Base Model', 'TTA Consensuated']
        main_handles = []
        main_labels = []
        other_handles = []
        other_labels = []
        
        for handle, label in zip(handles, labels):
            if label in main_categories:
                main_handles.append(handle)
                main_labels.append(label)
            else:
                other_handles.append(handle)
                other_labels.append(label)
        
        all_handles = main_handles + other_handles
        all_labels = main_labels + other_labels
        
        fig.legend(all_handles, all_labels, bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    plt.tight_layout()
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\n📊 Prediction Statistics:")
    print(f"   Ground Truth: {len(target_ellipses)} ellipses")
    print(f"   Base Model: {len(base_ellipses)} ellipses")
    if len(base_scores) > 0:
        print(f"   Base Scores: {base_scores.cpu().numpy()}")
    print(f"   Individual TTA: {sum(len(p['ellipses']) for p in individual_predictions)} total ellipses")
    print(f"   TTA Consensuated: {len(tta_ellipses)} ellipses")
    if len(tta_scores) > 0:
        print(f"   TTA Scores: {tta_scores.cpu().numpy()}")
    
    print(f"\n🔧 Individual Transform Results:")
    for pred_data in individual_predictions:
        count = len(pred_data['ellipses'])
        if count > 0:
            scores = pred_data['scores'].cpu().numpy()
            print(f"   {pred_data['name']}: {count} ellipses (scores: {scores})")
        else:
            print(f"   {pred_data['name']}: {count} ellipses")
    
    plt.show()

#%%
def test_custom_image(image_path, min_score=0.5):
    """
    Convenience function to test a custom image with base vs TTA predictions.
    
    Args:
        image_path: Path to the image file to test
        min_score: Minimum confidence threshold for predictions
    
    Example:
        test_custom_image("../docs/friends.jpg")
        test_custom_image("C:/path/to/your/image.jpg", min_score=0.3)
    """
    compare_base_vs_tta_predictions(
        custom_image_path=image_path,
        min_score=min_score
    )

# %%
