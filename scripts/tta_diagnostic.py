"""
TTA Diagnostic Script

This script helps debug why TTA might be performing worse than standard predictions.
It provides detailed analysis of each step in the TTA pipeline.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.tta.tta_transforms import tta_predict, TTA_TRANSFORMS
from ellipse_rcnn.error_utils.python_files import EllipseParGErrors


def analyze_tta_components(
    model_path_or_repo: str = "MJGT/ellipse-rcnn-FDDB",
    data_path: str = r"data\FDDB",
    n_images: int = 5,
    min_score: float = 0.6
):
    """
    Detailed analysis of TTA components to identify issues.
    """
    print("=== TTA Component Analysis ===")
    
    # Load dataset and model
    ds = FDDB(data_path)
    model = EllipseRCNN.from_pretrained(model_path_or_repo)
    model.eval()
    device = torch.device("cpu")
    
    # Sample images
    indices = np.random.choice(len(ds), min(n_images, len(ds)), replace=False)
    print(f"Analyzing {len(indices)} images...")
    
    results = {
        'image_idx': [],
        'transform_name': [],
        'num_predictions': [],
        'mean_score': [],
        'center_error': [],
        'consensuated_center_error': []
    }
    
    for idx in tqdm(indices):
        image_tensor, target = ds[idx]
        target_ellipses = target["ellipse_params"]
        
        if target_ellipses.numel() == 0:
            continue
        
        target_ParG = target_ellipses[:, [2, 3, 0, 1, 4]].detach().cpu().numpy()
        
        print(f"\n--- Image {idx} ---")
        print(f"Ground truth ellipses: {len(target_ParG)}")
        
        # 1. Standard prediction
        with torch.no_grad():
            standard_pred = model([image_tensor])
        
        standard_ellipses = standard_pred[0]["ellipse_params"]
        standard_scores = standard_pred[0]["scores"]
        
        if standard_ellipses.numel() > 0:
            score_mask = standard_scores > min_score
            filtered_ellipses = standard_ellipses[score_mask]
            filtered_scores = standard_scores[score_mask]
            
            print(f"Standard predictions: {len(standard_ellipses)} total, {len(filtered_ellipses)} above threshold")
            if len(filtered_scores) > 0:
                print(f"Standard mean score: {filtered_scores.mean():.3f}")
        else:
            print("Standard predictions: 0")
            continue
        
        # 2. Individual TTA transforms
        print("\nIndividual TTA transforms:")
        individual_predictions = []
        
        for transform_config in TTA_TRANSFORMS:
            forward_func, forward_kwargs = transform_config['forward']
            reverse_func, reverse_kwargs = transform_config['reverse']
            
            # Apply transformation
            augmented_img = forward_func(image_tensor, **forward_kwargs)
            
            # Run inference
            with torch.no_grad():
                predictions = model([augmented_img.to(device)])
            
            pred_dict = predictions[0]
            
            if pred_dict["ellipse_params"].numel() == 0:
                print(f"  {transform_config['name']}: 0 predictions")
                continue
            
            # Apply reverse transformation
            original_H, original_W = image_tensor.shape[1:]
            reversed_pred = reverse_func(pred_dict, original_H, original_W, **reverse_kwargs)
            
            # Filter by score
            score_mask = reversed_pred["scores"] > min_score
            if score_mask.any():
                filtered_ellipses = reversed_pred["ellipse_params"][score_mask]
                filtered_scores = reversed_pred["scores"][score_mask]
                
                # Calculate errors
                pred_ParG = filtered_ellipses[:, [2, 3, 0, 1, 4]].detach().cpu().numpy()
                
                center_errors = []
                for pred in pred_ParG:
                    min_error = float('inf')
                    for target in target_ParG:
                        errors = EllipseParGErrors(target, pred)
                        min_error = min(min_error, errors[0])  # center error
                    center_errors.append(min_error)
                
                mean_center_error = np.mean(center_errors) if center_errors else float('inf')
                
                print(f"  {transform_config['name']}: {len(filtered_ellipses)} predictions, "
                      f"mean score: {filtered_scores.mean():.3f}, "
                      f"center error: {mean_center_error:.2f}")
                
                results['image_idx'].append(idx)
                results['transform_name'].append(transform_config['name'])
                results['num_predictions'].append(len(filtered_ellipses))
                results['mean_score'].append(filtered_scores.mean().item())
                results['center_error'].append(mean_center_error)
                
                individual_predictions.append(reversed_pred)
            else:
                print(f"  {transform_config['name']}: 0 predictions above threshold")
        
        # 3. Consensuated prediction
        if individual_predictions:
            from ellipse_rcnn.tta.tta_transforms import consensuate_predictions
            final_predictions = consensuate_predictions(individual_predictions, min_score)
            
            if final_predictions['ellipse_params'].numel() > 0:
                consensuated_ellipses = final_predictions['ellipse_params']
                consensuated_scores = final_predictions['scores']
                
                pred_ParG = consensuated_ellipses[:, [2, 3, 0, 1, 4]].detach().cpu().numpy()
                
                center_errors = []
                for pred in pred_ParG:
                    min_error = float('inf')
                    for target in target_ParG:
                        errors = EllipseParGErrors(target, pred)
                        min_error = min(min_error, errors[0])
                    center_errors.append(min_error)
                
                mean_center_error = np.mean(center_errors) if center_errors else float('inf')
                
                print(f"Consensuated: {len(consensuated_ellipses)} predictions, "
                      f"mean score: {consensuated_scores.mean():.3f}, "
                      f"center error: {mean_center_error:.2f}")
                
                # Store consensuated results
                for i, result_idx in enumerate([j for j, x in enumerate(results['image_idx']) if x == idx]):
                    if result_idx < len(results['consensuated_center_error']):
                        results['consensuated_center_error'][result_idx] = mean_center_error
                    else:
                        results['consensuated_center_error'].append(mean_center_error)
    
    # Analysis
    print("\n=== ANALYSIS ===")
    
    if results['transform_name']:
        import pandas as pd
        df = pd.DataFrame(results)
        
        # Which transforms perform best/worst?
        transform_analysis = df.groupby('transform_name').agg({
            'center_error': ['mean', 'count'],
            'mean_score': 'mean',
            'num_predictions': 'mean'
        }).round(3)
        
        print("\nTransform Performance:")
        print(transform_analysis)
        
        # Score vs Error correlation
        if len(df) > 1:
            correlation = df['mean_score'].corr(df['center_error'])
            print(f"\nScore vs Error Correlation: {correlation:.3f}")
            
            # Plot transform performance
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            transform_means = df.groupby('transform_name')['center_error'].mean().sort_values()
            transform_means.plot(kind='bar')
            plt.title('Mean Center Error by Transform')
            plt.xticks(rotation=45)
            plt.ylabel('Center Error (pixels)')
            
            plt.subplot(2, 2, 2)
            df.groupby('transform_name')['mean_score'].mean().plot(kind='bar')
            plt.title('Mean Prediction Score by Transform')
            plt.xticks(rotation=45)
            plt.ylabel('Mean Score')
            
            plt.subplot(2, 2, 3)
            plt.scatter(df['mean_score'], df['center_error'], alpha=0.7)
            plt.xlabel('Mean Score')
            plt.ylabel('Center Error')
            plt.title('Score vs Error Scatter')
            
            plt.subplot(2, 2, 4)
            df.groupby('transform_name')['num_predictions'].mean().plot(kind='bar')
            plt.title('Mean Number of Predictions by Transform')
            plt.xticks(rotation=45)
            plt.ylabel('Number of Predictions')
            
            plt.tight_layout()
            plt.show()
    
    return results


def check_transform_reversibility():
    """Test if transformations and their reversals are working correctly."""
    print("\n=== Testing Transform Reversibility ===")
    
    from ellipse_rcnn.tta.tta_transforms import TTA_TRANSFORMS
    
    # Create a test ellipse
    test_ellipse = torch.tensor([[20.0, 15.0, 100.0, 50.0, 0.5]])  # [a, b, cx, cy, theta]
    test_boxes = torch.tensor([[80.0, 35.0, 120.0, 65.0]])  # [x1, y1, x2, y2]
    test_scores = torch.tensor([0.9])
    test_labels = torch.tensor([1])
    
    original_pred = {
        'ellipse_params': test_ellipse,
        'boxes': test_boxes,
        'scores': test_scores,
        'labels': test_labels
    }
    
    H, W = 200, 300
    
    for transform_config in TTA_TRANSFORMS:
        name = transform_config['name']
        forward_func, forward_kwargs = transform_config['forward']
        reverse_func, reverse_kwargs = transform_config['reverse']
        
        print(f"{name}:")
        
        # Test 1: Direct reverse transformation (what happens to predictions from transformed image)
        reversed_pred = reverse_func(original_pred, H, W, **reverse_kwargs)
        rev_ellipse = reversed_pred['ellipse_params'][0]
        orig_ellipse = test_ellipse[0]
        
        print(f"  Original: a={orig_ellipse[0]:.1f}, b={orig_ellipse[1]:.1f}, "
              f"cx={orig_ellipse[2]:.1f}, cy={orig_ellipse[3]:.1f}, θ={orig_ellipse[4]:.3f}")
        print(f"  Reversed: a={rev_ellipse[0]:.1f}, b={rev_ellipse[1]:.1f}, "
              f"cx={rev_ellipse[2]:.1f}, cy={rev_ellipse[3]:.1f}, θ={rev_ellipse[4]:.3f}")
        
        # Test 2: Double application test (should return to original for symmetric transforms)
        if name in ['Identity', 'Horizontal Flip']:  # These should be involutions
            double_pred = reverse_func(reversed_pred, H, W, **reverse_kwargs)
            double_ellipse = double_pred['ellipse_params'][0]
            
            # Calculate round-trip error
            center_error = torch.sqrt((double_ellipse[2] - orig_ellipse[2])**2 + 
                                     (double_ellipse[3] - orig_ellipse[3])**2)
            angle_error = abs(double_ellipse[4] - orig_ellipse[4])
            size_error = torch.sqrt((double_ellipse[0] - orig_ellipse[0])**2 + 
                                   (double_ellipse[1] - orig_ellipse[1])**2)
            
            print(f"  Double-apply test: center_err={center_error:.3f}px, angle_err={angle_error:.3f}rad, size_err={size_error:.3f}")
            
            if center_error > 0.001 or angle_error > 0.001 or size_error > 0.001:
                print(f"  ⚠️  WARNING: {name} should be an involution but isn't!")
        
        # Check for obvious errors
        if torch.any(rev_ellipse[:2] <= 0):
            print(f"  ⚠️  WARNING: Negative or zero semi-axes!")
        if torch.any(torch.isnan(rev_ellipse)) or torch.any(torch.isinf(rev_ellipse)):
            print(f"  ⚠️  WARNING: NaN or Inf values!")
        if rev_ellipse[2] < 0 or rev_ellipse[2] > W or rev_ellipse[3] < 0 or rev_ellipse[3] > H:
            print(f"  ⚠️  WARNING: Center outside image bounds!")
            
        # Check magnitude of transformation (should be reasonable)
        center_change = torch.sqrt((rev_ellipse[2] - orig_ellipse[2])**2 + 
                                  (rev_ellipse[3] - orig_ellipse[3])**2)
        if center_change > W * 0.5:  # If center moves more than half image width
            print(f"  ⚠️  WARNING: Very large center displacement ({center_change:.1f}px)!")
        
        print()


if __name__ == "__main__":
    # Test transform reversibility first
    check_transform_reversibility()
    
    # Then analyze TTA components
    results = analyze_tta_components(n_images=3)
    
    print("\nDiagnostic complete!")
