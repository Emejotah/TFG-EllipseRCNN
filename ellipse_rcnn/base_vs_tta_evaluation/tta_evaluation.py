"""
TTA vs Base Model Evaluation Script

Compares the performance of base model predictions vs TTA consensuated predictions
using various ellipse error metrics.
"""

import torch
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
try:
    from sklearn.metrics import precision_recall_curve, average_precision_score
except ImportError:
    print("Warning: sklearn not available. Installing with: pip install scikit-learn")
    
    def precision_recall_curve(y_true, y_score):
        """Simple fallback implementation of precision-recall curve."""
        # Sort by score (descending)
        sorted_indices = np.argsort(-np.array(y_score))
        y_true_sorted = np.array(y_true)[sorted_indices]
        y_score_sorted = np.array(y_score)[sorted_indices]
        
        # Calculate precision and recall at each threshold
        precision = []
        recall = []
        thresholds = []
        
        total_positives = np.sum(y_true_sorted)
        if total_positives == 0:
            return np.array([1.0]), np.array([0.0]), np.array([0.0])
        
        tp = 0
        fp = 0
        
        for i, (true_label, score) in enumerate(zip(y_true_sorted, y_score_sorted)):
            if true_label == 1:
                tp += 1
            else:
                fp += 1
            
            current_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            current_recall = tp / total_positives
            
            precision.append(current_precision)
            recall.append(current_recall)
            thresholds.append(score)
        
        return np.array(precision), np.array(recall), np.array(thresholds)
    
    def average_precision_score(y_true, y_score):
        """Simple fallback implementation of average precision."""
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # Simple trapezoidal integration
        return np.trapz(precision, recall) if len(precision) > 1 else 0.0
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.tta.tta_transforms import tta_predict
from ellipse_rcnn.error_utils.python_files.ellipse_errors import (
    EllipseAlgebraicError,
    EllipseNaturalError, 
    EllipseParGErrors
)
from ellipse_rcnn.error_utils.python_files.ellipse_conversions import GtoA, GtoN


def ellipse_params_to_ParG(ellipse_params: torch.Tensor) -> np.ndarray:
    """
    Convert ellipse parameters tensor [a, b, x, y, theta] to geometric parameters.
    
    Args:
        ellipse_params: Tensor of shape (N, 5) with [a, b, x, y, theta]
        
    Returns:
        ParG: Numpy array of shape (N, 5) with [x, y, a, b, theta]
    """
    if ellipse_params.dim() == 1:
        ellipse_params = ellipse_params.unsqueeze(0)
    
    # Convert from [a, b, x, y, theta] to [x, y, a, b, theta]
    ParG = ellipse_params[:, [2, 3, 0, 1, 4]].cpu().numpy()
    return ParG


def calculate_ellipse_errors(pred_ellipses: torch.Tensor, 
                           target_ellipses: torch.Tensor) -> Dict[str, List[float]]:
    """
    Calculate various ellipse error metrics between predictions and targets.
    
    Args:
        pred_ellipses: Predicted ellipse parameters
        target_ellipses: Ground truth ellipse parameters
        
    Returns:
        Dictionary containing error metrics for each prediction-target pair
    """
    errors = {
        'center_error': [],
        'angle_error': [],
        'major_axis_error': [],
        'minor_axis_error': [],
        'area_error': [],
        'algebraic_error': [],
        'natural_error': []
    }
    
    # Handle empty predictions or targets
    if pred_ellipses.numel() == 0 or target_ellipses.numel() == 0:
        return errors
    
    # Convert to ParG format (geometric parameters)
    pred_ParG = ellipse_params_to_ParG(pred_ellipses)
    target_ParG = ellipse_params_to_ParG(target_ellipses)
    
    # For each prediction, find the closest target ellipse
    for i in range(pred_ParG.shape[0]):
        pred_center = pred_ParG[i, :2]
        
        # Find closest target by center distance
        target_centers = target_ParG[:, :2]
        distances = np.linalg.norm(target_centers - pred_center, axis=1)
        closest_idx = np.argmin(distances)
        
        # Calculate geometric errors
        geom_errors = EllipseParGErrors(target_ParG[closest_idx], pred_ParG[i])
        errors['center_error'].append(geom_errors[0])
        errors['angle_error'].append(geom_errors[1])
        errors['major_axis_error'].append(geom_errors[2])
        errors['minor_axis_error'].append(geom_errors[3])
        errors['area_error'].append(geom_errors[4])
        
        # Calculate algebraic error
        try:
            target_ParA = GtoA(target_ParG[closest_idx], 1)  # code=1 for ellipse
            pred_ParA = GtoA(pred_ParG[i], 1)
            alg_error = EllipseAlgebraicError(target_ParA, pred_ParA)
            errors['algebraic_error'].append(alg_error)
        except:
            errors['algebraic_error'].append(np.nan)
        
        # Calculate natural error
        try:
            target_ParN = GtoN(target_ParG[closest_idx])
            pred_ParN = GtoN(pred_ParG[i])
            nat_error = EllipseNaturalError(target_ParN, pred_ParN)
            errors['natural_error'].append(nat_error)
        except:
            errors['natural_error'].append(np.nan)
    
    return errors


def compute_error_statistics(errors: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean, std, and median statistics for each error type."""
    stats = {}
    
    for error_type, error_values in errors.items():
        if len(error_values) > 0:
            # Filter out NaN values
            valid_values = [v for v in error_values if not np.isnan(v)]
            if len(valid_values) > 0:
                stats[error_type] = {
                    'mean': np.mean(valid_values),
                    'std': np.std(valid_values),
                    'median': np.median(valid_values),
                    'count': len(valid_values)
                }
            else:
                stats[error_type] = {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'count': 0}
        else:
            stats[error_type] = {'mean': np.nan, 'std': np.nan, 'median': np.nan, 'count': 0}
    
    return stats


def calculate_precision_recall_metrics(predictions: torch.Tensor, 
                                     targets: torch.Tensor, 
                                     scores: torch.Tensor,
                                     iou_threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """
    Calculate precision-recall metrics for ellipse detection.
    
    Args:
        predictions: Predicted ellipse parameters (N, 5)
        targets: Ground truth ellipse parameters (M, 5)
        scores: Confidence scores for predictions (N,)
        iou_threshold: IoU threshold for considering a detection as true positive
        
    Returns:
        Dictionary containing precision, recall, thresholds, and AP score
    """
    if predictions.numel() == 0 or targets.numel() == 0:
        return {
            'precision': np.array([1.0]),
            'recall': np.array([0.0]),
            'thresholds': np.array([0.0]),
            'average_precision': 0.0,
            'true_positives': [],
            'false_positives': [],
            'matched_targets': set()
        }
    
    # Convert to numpy for easier processing
    pred_np = predictions.cpu().numpy()
    target_np = targets.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Sort predictions by score (descending)
    sorted_indices = np.argsort(-scores_np)
    pred_sorted = pred_np[sorted_indices]
    scores_sorted = scores_np[sorted_indices]
    
    # Calculate IoU between all prediction-target pairs
    num_preds = len(pred_sorted)
    num_targets = len(target_np)
    
    # For ellipses, we'll use center distance and size similarity as a proxy for IoU
    # since calculating exact ellipse IoU is computationally expensive
    def ellipse_similarity(pred, target):
        """Calculate similarity between two ellipses based on center distance and size."""
        # Extract center coordinates and axes
        pred_center = pred[[2, 3]]  # [x, y]
        target_center = target[[2, 3]]  # [x, y]
        pred_axes = pred[[0, 1]]  # [a, b]
        target_axes = target[[0, 1]]  # [a, b]
        
        # Calculate center distance relative to ellipse size
        center_dist = np.linalg.norm(pred_center - target_center)
        avg_size = (np.mean(pred_axes) + np.mean(target_axes)) / 2
        
        if avg_size == 0:
            return 0.0
            
        center_score = max(0, 1 - center_dist / (2 * avg_size))
        
        # Calculate size similarity
        size_ratio = min(np.min(pred_axes) / np.max(target_axes), 
                        np.min(target_axes) / np.max(pred_axes))
        
        # Combined similarity score
        return center_score * size_ratio
    
    # Match predictions to targets
    y_true = []
    y_scores = []
    matched_targets = set()
    true_positives = []
    false_positives = []
    
    for i, (pred, score) in enumerate(zip(pred_sorted, scores_sorted)):
        best_similarity = 0
        best_target_idx = -1
        
        # Find best matching target
        for j, target in enumerate(target_np):
            if j not in matched_targets:
                similarity = ellipse_similarity(pred, target)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_target_idx = j
        
        # Check if it's a true positive
        if best_similarity >= iou_threshold and best_target_idx != -1:
            y_true.append(1)
            matched_targets.add(best_target_idx)
            true_positives.append(i)
        else:
            y_true.append(0)
            false_positives.append(i)
        
        y_scores.append(score)
    
    # Calculate precision-recall curve
    if len(y_true) > 0:
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        average_precision = average_precision_score(y_true, y_scores)
    else:
        precision, recall, thresholds = np.array([1.0]), np.array([0.0]), np.array([0.0])
        average_precision = 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'average_precision': average_precision,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'matched_targets': matched_targets,
        'total_targets': num_targets
    }


def analyze_false_positives(predictions: torch.Tensor, 
                          targets: torch.Tensor, 
                          scores: torch.Tensor,
                          false_positive_indices: List[int]) -> Dict[str, float]:
    """
    Analyze characteristics of false positive detections.
    
    Args:
        predictions: All predicted ellipse parameters
        targets: Ground truth ellipse parameters  
        scores: Confidence scores for predictions
        false_positive_indices: Indices of false positive predictions
        
    Returns:
        Dictionary with false positive analysis
    """
    if not false_positive_indices or predictions.numel() == 0:
        return {
            'avg_fp_score': 0.0,
            'avg_fp_size': 0.0,
            'fp_score_distribution': [],
            'fp_size_distribution': [],
            'num_false_positives': 0
        }
    
    fp_predictions = predictions[false_positive_indices]
    fp_scores = scores[false_positive_indices]
    
    # Calculate false positive characteristics
    fp_sizes = torch.sqrt(fp_predictions[:, 0] * fp_predictions[:, 1])  # geometric mean of axes
    
    return {
        'avg_fp_score': float(torch.mean(fp_scores)),
        'avg_fp_size': float(torch.mean(fp_sizes)),
        'fp_score_distribution': fp_scores.cpu().numpy().tolist(),
        'fp_size_distribution': fp_sizes.cpu().numpy().tolist(),
        'num_false_positives': len(false_positive_indices)
    }


def plot_precision_recall_curves(base_pr_data: Dict, tta_pr_data: Dict, save_path: str = None):
    """
    Plot precision-recall curves for base and TTA models.
    
    Args:
        base_pr_data: Precision-recall data for base model
        tta_pr_data: Precision-recall data for TTA model
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Precision-Recall Curves
    plt.subplot(1, 2, 1)
    plt.plot(base_pr_data['recall'], base_pr_data['precision'], 
             label=f'Base Model (AP={base_pr_data["average_precision"]:.3f})', 
             linewidth=2, color='blue')
    plt.plot(tta_pr_data['recall'], tta_pr_data['precision'], 
             label=f'TTA Model (AP={tta_pr_data["average_precision"]:.3f})', 
             linewidth=2, color='red')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    # Plot 2: False Positive Analysis
    plt.subplot(1, 2, 2)
    base_fp_scores = base_pr_data.get('fp_analysis', {}).get('fp_score_distribution', [])
    tta_fp_scores = tta_pr_data.get('fp_analysis', {}).get('fp_score_distribution', [])
    
    if base_fp_scores or tta_fp_scores:
        bins = np.linspace(0, 1, 20)
        if base_fp_scores:
            plt.hist(base_fp_scores, bins=bins, alpha=0.7, label='Base FP Scores', color='blue')
        if tta_fp_scores:
            plt.hist(tta_fp_scores, bins=bins, alpha=0.7, label='TTA FP Scores', color='red')
        
        plt.xlabel('Confidence Score')
        plt.ylabel('Number of False Positives')
        plt.title('False Positive Score Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No False Positives Data', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('False Positive Score Distribution')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Precision-Recall curves saved to {save_path}")
    
    plt.show()


def plot_detection_statistics(detection_stats: Dict, save_path: str = None):
    """
    Plot detection statistics including detection counts and error distributions.
    
    Args:
        detection_stats: Dictionary containing detection statistics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Detection counts comparison
    plt.subplot(2, 3, 1)
    categories = ['Targets', 'Base', 'TTA']
    means = [
        detection_stats['mean_target_ellipses'],
        detection_stats['mean_base_detections'], 
        detection_stats['mean_tta_detections']
    ]
    colors = ['green', 'blue', 'red']
    bars = plt.bar(categories, means, color=colors, alpha=0.7)
    plt.ylabel('Average Detections per Image')
    plt.title('Detection Counts Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{mean:.2f}', ha='center', va='bottom')
    
    # Plot 2: Detection count distributions
    plt.subplot(2, 3, 2)
    base_detections = detection_stats['base_detections']
    tta_detections = detection_stats['tta_detections']
    target_ellipses = detection_stats['target_ellipses']
    
    max_count = max(max(base_detections + tta_detections + target_ellipses), 10)
    bins = range(0, max_count + 2)
    
    plt.hist(target_ellipses, bins=bins, alpha=0.7, label='Targets', color='green')
    plt.hist(base_detections, bins=bins, alpha=0.7, label='Base', color='blue')
    plt.hist(tta_detections, bins=bins, alpha=0.7, label='TTA', color='red')
    
    plt.xlabel('Number of Detections')
    plt.ylabel('Number of Images')
    plt.title('Detection Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Precision and Recall comparison
    if 'base_pr_metrics' in detection_stats and 'tta_pr_metrics' in detection_stats:
        plt.subplot(2, 3, 3)
        base_ap = detection_stats['base_pr_metrics']['average_precision']
        tta_ap = detection_stats['tta_pr_metrics']['average_precision']
        
        categories = ['Base Model', 'TTA Model']
        ap_scores = [base_ap, tta_ap]
        colors = ['blue', 'red']
        bars = plt.bar(categories, ap_scores, color=colors, alpha=0.7)
        plt.ylabel('Average Precision (AP)')
        plt.title('Average Precision Comparison')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ap in zip(bars, ap_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{ap:.3f}', ha='center', va='bottom')
    
    # Plot 4: False Positive Analysis
    plt.subplot(2, 3, 4)
    if 'base_pr_metrics' in detection_stats and 'tta_pr_metrics' in detection_stats:
        base_fp = detection_stats['base_pr_metrics'].get('fp_analysis', {})
        tta_fp = detection_stats['tta_pr_metrics'].get('fp_analysis', {})
        
        categories = ['Base FP', 'TTA FP']
        fp_counts = [
            base_fp.get('num_false_positives', 0),
            tta_fp.get('num_false_positives', 0)
        ]
        colors = ['blue', 'red']
        bars = plt.bar(categories, fp_counts, color=colors, alpha=0.7)
        plt.ylabel('Total False Positives')
        plt.title('False Positive Count Comparison')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, count in zip(bars, fp_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{count}', ha='center', va='bottom')
    
    # Plot 5 & 6: Score distributions for true vs false positives
    if 'base_pr_metrics' in detection_stats:
        plt.subplot(2, 3, 5)
        base_fp_scores = detection_stats['base_pr_metrics'].get('fp_analysis', {}).get('fp_score_distribution', [])
        if base_fp_scores:
            plt.hist(base_fp_scores, bins=15, alpha=0.7, color='red', label='False Positives')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('Base Model: FP Score Distribution')
        plt.grid(True, alpha=0.3)
        
    if 'tta_pr_metrics' in detection_stats:
        plt.subplot(2, 3, 6)
        tta_fp_scores = detection_stats['tta_pr_metrics'].get('fp_analysis', {}).get('fp_score_distribution', [])
        if tta_fp_scores:
            plt.hist(tta_fp_scores, bins=15, alpha=0.7, color='red', label='False Positives')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title('TTA Model: FP Score Distribution')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Detection statistics plots saved to {save_path}")
    
    plt.show()


def evaluate_model(model_repo: str = "MJGT/ellipse-rcnn-FDDB",
                  data_path: str = "data/FDDB",
                  num_images: int = 100,
                  min_score: float = 0.75,
                  device: str = "cuda") -> Dict:
    """
    Evaluate base model vs TTA performance on FDDB dataset.
    
    Args:
        model_repo: HuggingFace model repository
        data_path: Path to FDDB dataset
        num_images: Number of images to evaluate
        min_score: Minimum confidence score threshold
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation results
    """
    print(f"🚀 Starting TTA vs Base Model Evaluation")
    print(f"Model: {model_repo}")
    print(f"Dataset: {data_path}")
    print(f"Number of images: {num_images}")
    print(f"Device: {device}")
    print("="*60)
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")
    
    # Load dataset
    print("📂 Loading FDDB dataset...")
    dataset = FDDB(data_path)
    print(f"✅ Dataset loaded with {len(dataset)} images")
    
    # Load model
    print(f"📦 Loading model from {model_repo}...")
    model = EllipseRCNN.from_pretrained(model_repo)
    model.eval().to(device)
    print("✅ Model loaded successfully")
    
    # Initialize result containers
    base_errors = {
        'center_error': [], 'angle_error': [], 'major_axis_error': [],
        'minor_axis_error': [], 'area_error': [], 'algebraic_error': [], 'natural_error': []
    }
    tta_errors = {
        'center_error': [], 'angle_error': [], 'major_axis_error': [],
        'minor_axis_error': [], 'area_error': [], 'algebraic_error': [], 'natural_error': []
    }
    
    detection_stats = {
        'base_detections': [],
        'tta_detections': [],
        'target_ellipses': [],
        'base_predictions': [],  # Store all predictions with scores for PR analysis
        'tta_predictions': [],   # Store all predictions with scores for PR analysis
        'all_targets': []        # Store all targets for PR analysis
    }
    
    # Limit evaluation to specified number of images
    num_images = min(num_images, len(dataset))
    indices = list(range(num_images))
    
    print(f"🔍 Evaluating on {num_images} images...")
    
    # Process each image
    for idx in tqdm(indices, desc="Processing images", unit="img"):
        image_tensor, target = dataset[idx]
        image_tensor = image_tensor.to(device)
        target_ellipses = target["ellipse_params"]
        
        # Skip images with no ground truth ellipses
        if target_ellipses.numel() == 0:
            continue
        
        detection_stats['target_ellipses'].append(len(target_ellipses))
        detection_stats['all_targets'].append(target_ellipses.clone())
        
        # Base model prediction
        with torch.no_grad():
            base_pred = model([image_tensor])
            base_prediction = base_pred[0]
        
        # Store all base predictions for PR analysis (before score filtering)
        if base_prediction["ellipse_params"].numel() > 0:
            detection_stats['base_predictions'].append({
                'ellipse_params': base_prediction["ellipse_params"].clone(),
                'scores': base_prediction["scores"].clone(),
                'target_ellipses': target_ellipses.clone()
            })
            
            # Apply consistent score threshold
            base_score_mask = base_prediction["scores"] > min_score
            base_ellipses = base_prediction["ellipse_params"][base_score_mask]
            base_scores_filtered = base_prediction["scores"][base_score_mask]
        else:
            detection_stats['base_predictions'].append({
                'ellipse_params': torch.empty(0, 5),
                'scores': torch.empty(0),
                'target_ellipses': target_ellipses.clone()
            })
            base_ellipses = torch.empty(0, 5)
            base_scores_filtered = torch.empty(0)
        
        detection_stats['base_detections'].append(len(base_ellipses))
        
        # Debug print for first few images
        if idx < 3:
            print(f"  Image {idx+1}: Targets={len(target_ellipses)}, Base={len(base_ellipses)} (scores: {base_scores_filtered.cpu().numpy() if len(base_scores_filtered) > 0 else 'none'})")
        
        # Calculate base model errors
        if base_ellipses.numel() > 0:
            errors = calculate_ellipse_errors(base_ellipses, target_ellipses)
            for key, values in errors.items():
                base_errors[key].extend(values)
        
        # TTA prediction with consensuation
        # NOTE: Pass min_score=0.0 to get all TTA predictions, then filter consistently
        tta_predictions = tta_predict(
            model=model,
            image_tensor=image_tensor,
            device=device,
            min_score=0.0,  # Get all predictions from TTA
            consensuate=True,
            visualize=False,
            consensuation_method='quality'
        )
        tta_prediction = tta_predictions[0]
        
        # Store all TTA predictions for PR analysis (before score filtering)
        if tta_prediction["ellipse_params"].numel() > 0:
            detection_stats['tta_predictions'].append({
                'ellipse_params': tta_prediction["ellipse_params"].clone(),
                'scores': tta_prediction["scores"].clone(),
                'target_ellipses': target_ellipses.clone()
            })
            
            # Apply consistent score threshold
            tta_score_mask = tta_prediction["scores"] > min_score
            tta_ellipses = tta_prediction["ellipse_params"][tta_score_mask]
            tta_scores_filtered = tta_prediction["scores"][tta_score_mask]
        else:
            detection_stats['tta_predictions'].append({
                'ellipse_params': torch.empty(0, 5),
                'scores': torch.empty(0),
                'target_ellipses': target_ellipses.clone()
            })
            tta_ellipses = torch.empty(0, 5)
            tta_scores_filtered = torch.empty(0)
        
        detection_stats['tta_detections'].append(len(tta_ellipses))
        
        # Debug print for first few images
        if idx < 3:
            print(f"                    TTA={len(tta_ellipses)} (scores: {tta_scores_filtered.cpu().numpy() if len(tta_scores_filtered) > 0 else 'none'})")
        
        # Calculate TTA errors
        if tta_ellipses.numel() > 0:
            errors = calculate_ellipse_errors(tta_ellipses, target_ellipses)
            for key, values in errors.items():
                tta_errors[key].extend(values)
    
    print("📊 Computing statistics...")
    
    # Compute error statistics
    base_stats = compute_error_statistics(base_errors)
    tta_stats = compute_error_statistics(tta_errors)
    
    # Compute precision-recall metrics
    print("📈 Computing precision-recall metrics...")
    
    # Combine all predictions and targets for base model
    all_base_predictions = torch.cat([p['ellipse_params'] for p in detection_stats['base_predictions'] 
                                     if p['ellipse_params'].numel() > 0], dim=0) if any(p['ellipse_params'].numel() > 0 for p in detection_stats['base_predictions']) else torch.empty(0, 5)
    all_base_scores = torch.cat([p['scores'] for p in detection_stats['base_predictions'] 
                                if p['scores'].numel() > 0], dim=0) if any(p['scores'].numel() > 0 for p in detection_stats['base_predictions']) else torch.empty(0)
    all_targets = torch.cat([p['target_ellipses'] for p in detection_stats['base_predictions'] 
                            if p['target_ellipses'].numel() > 0], dim=0) if any(p['target_ellipses'].numel() > 0 for p in detection_stats['base_predictions']) else torch.empty(0, 5)
    
    # Calculate PR metrics for base model
    base_pr_metrics = calculate_precision_recall_metrics(all_base_predictions, all_targets, all_base_scores)
    base_fp_analysis = analyze_false_positives(all_base_predictions, all_targets, all_base_scores, base_pr_metrics['false_positives'])
    base_pr_metrics['fp_analysis'] = base_fp_analysis
    
    # Combine all predictions and targets for TTA model  
    all_tta_predictions = torch.cat([p['ellipse_params'] for p in detection_stats['tta_predictions'] 
                                    if p['ellipse_params'].numel() > 0], dim=0) if any(p['ellipse_params'].numel() > 0 for p in detection_stats['tta_predictions']) else torch.empty(0, 5)
    all_tta_scores = torch.cat([p['scores'] for p in detection_stats['tta_predictions'] 
                               if p['scores'].numel() > 0], dim=0) if any(p['scores'].numel() > 0 for p in detection_stats['tta_predictions']) else torch.empty(0)
    
    # Calculate PR metrics for TTA model
    tta_pr_metrics = calculate_precision_recall_metrics(all_tta_predictions, all_targets, all_tta_scores)
    tta_fp_analysis = analyze_false_positives(all_tta_predictions, all_targets, all_tta_scores, tta_pr_metrics['false_positives'])
    tta_pr_metrics['fp_analysis'] = tta_fp_analysis
    
    # Compute detection statistics
    detection_summary = {
        'mean_base_detections': np.mean(detection_stats['base_detections']),
        'mean_tta_detections': np.mean(detection_stats['tta_detections']),
        'mean_target_ellipses': np.mean(detection_stats['target_ellipses']),
        'total_images_processed': len(detection_stats['target_ellipses']),
        'base_pr_metrics': base_pr_metrics,
        'tta_pr_metrics': tta_pr_metrics
    }
    
    # Add PR metrics to detection_stats for plotting
    detection_stats['base_pr_metrics'] = base_pr_metrics
    detection_stats['tta_pr_metrics'] = tta_pr_metrics
    
    # Prepare results
    results = {
        'evaluation_config': {
            'model_repo': model_repo,
            'data_path': data_path,
            'num_images_evaluated': len(detection_stats['target_ellipses']),
            'min_score': min_score,
            'device': str(device)
        },
        'base_model_stats': base_stats,
        'tta_model_stats': tta_stats,
        'detection_summary': detection_summary,
        'raw_detection_stats': detection_stats
    }
    
    return results


def print_comparison_results(results: Dict):
    """Print formatted comparison results."""
    print("\n" + "="*80)
    print("📈 EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    detection_summary = results['detection_summary']
    print(f"Images processed: {detection_summary['total_images_processed']}")
    print(f"Avg target ellipses per image: {detection_summary['mean_target_ellipses']:.2f}")
    print(f"Avg base detections per image: {detection_summary['mean_base_detections']:.2f}")
    print(f"Avg TTA detections per image: {detection_summary['mean_tta_detections']:.2f}")
    
    detection_improvement = ((detection_summary['mean_tta_detections'] - 
                            detection_summary['mean_base_detections']) / 
                           max(detection_summary['mean_base_detections'], 0.001)) * 100
    print(f"TTA detection improvement: {detection_improvement:+.1f}%")
    
    # Print precision-recall metrics
    if 'base_pr_metrics' in detection_summary and 'tta_pr_metrics' in detection_summary:
        base_ap = detection_summary['base_pr_metrics']['average_precision']
        tta_ap = detection_summary['tta_pr_metrics']['average_precision']
        base_fp_count = detection_summary['base_pr_metrics'].get('fp_analysis', {}).get('num_false_positives', 0)
        tta_fp_count = detection_summary['tta_pr_metrics'].get('fp_analysis', {}).get('num_false_positives', 0)
        
        print(f"\n📊 PRECISION-RECALL METRICS")
        print(f"Base Model Average Precision (AP): {base_ap:.3f}")
        print(f"TTA Model Average Precision (AP):  {tta_ap:.3f}")
        print(f"AP Improvement: {((tta_ap - base_ap) / max(base_ap, 0.001)) * 100:+.1f}%")
        print(f"\n🚫 FALSE POSITIVE ANALYSIS")
        print(f"Base Model False Positives: {base_fp_count}")
        print(f"TTA Model False Positives:  {tta_fp_count}")
        fp_reduction = ((base_fp_count - tta_fp_count) / max(base_fp_count, 1)) * 100
        print(f"False Positive Reduction: {fp_reduction:+.1f}%")
    
    print("\n" + "-"*80)
    print("ERROR METRICS COMPARISON")
    print("-"*80)
    print(f"{'Metric':<20} {'Base Mean':<12} {'TTA Mean':<12} {'Improvement':<12} {'Base Count':<12} {'TTA Count':<12}")
    print("-"*80)
    
    base_stats = results['base_model_stats']
    tta_stats = results['tta_model_stats']
    
    for metric in ['center_error', 'angle_error', 'major_axis_error', 'minor_axis_error', 
                   'area_error', 'algebraic_error', 'natural_error']:
        
        base_mean = base_stats.get(metric, {}).get('mean', np.nan)
        tta_mean = tta_stats.get(metric, {}).get('mean', np.nan)
        base_count = base_stats.get(metric, {}).get('count', 0)
        tta_count = tta_stats.get(metric, {}).get('count', 0)
        
        if not np.isnan(base_mean) and not np.isnan(tta_mean) and base_mean > 0:
            improvement = ((base_mean - tta_mean) / base_mean) * 100
            improvement_str = f"{improvement:+.1f}%"
        else:
            improvement_str = "N/A"
        
        print(f"{metric:<20} {base_mean:<12.4f} {tta_mean:<12.4f} {improvement_str:<12} {base_count:<12} {tta_count:<12}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate TTA vs Base Model Performance")
    parser.add_argument("--model_repo", type=str, default="MJGT/ellipse-rcnn-FDDB",
                       help="HuggingFace model repository")
    parser.add_argument("--data_path", type=str, default="../../data/FDDB",
                       help="Path to FDDB dataset")
    parser.add_argument("--num_images", type=int, default=100,
                       help="Number of images to evaluate")
    parser.add_argument("--min_score", type=float, default=0.7,
                       help="Minimum confidence score threshold")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run evaluation on")
    parser.add_argument("--output_file", type=str, default="tta_evaluation_results.json",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_model(
        model_repo=args.model_repo,
        data_path=args.data_path,
        num_images=args.num_images,
        min_score=args.min_score,
        device=args.device
    )
    
    # Print results
    print_comparison_results(results)
    
    # Generate and save plots
    print("\n📈 Generating visualization plots...")
    
    # Generate precision-recall curves
    if 'base_pr_metrics' in results['detection_summary'] and 'tta_pr_metrics' in results['detection_summary']:
        base_pr_data = results['detection_summary']['base_pr_metrics']
        tta_pr_data = results['detection_summary']['tta_pr_metrics']
        pr_plot_path = args.output_file.replace('.json', '_precision_recall_curves.png')
        plot_precision_recall_curves(base_pr_data, tta_pr_data, pr_plot_path)
    
    # Generate detection statistics plots
    detection_plot_path = args.output_file.replace('.json', '_detection_statistics.png')
    # Combine detection_summary with raw_detection_stats for plotting
    plot_data = results['detection_summary'].copy()
    plot_data.update(results['raw_detection_stats'])
    plot_detection_statistics(plot_data, detection_plot_path)
    
    # Save results to file
    print(f"\n💾 Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
