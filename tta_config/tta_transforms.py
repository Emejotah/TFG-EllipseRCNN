"""
Test Time Augmentation (TTA) for Ellipse R-CNN.

This module provides clean, robust TTA functionality with proper coordinate transformations
and prediction consensuation for improved ellipse detection performance.
"""

import torch
import torchvision.transforms.functional as F
from torchvision.ops import nms
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from matplotlib.collections import EllipseCollection
from matplotlib.axes import Axes
from scipy.spatial.distance import cdist, euclidean

# --- Configuration ---
TTA_CONFIG = {
    'rotation_angle': 10,  # degrees for small rotations
    'brightness_factor': 1.2,  # brightness adjustment
    'contrast_factor': 1.2,   # contrast adjustment
    'min_score_threshold': 0.75,  # confidence threshold
    'consensuation_distance_threshold': 30.0,  # pixels
    'size_similarity_threshold': 0.5,  # minimum size ratio for grouping ellipses
    'nms_iou_threshold': 0.5,  # IoU threshold for NMS
    'apply_nms': True,  # Whether to apply NMS to final predictions
}

# --- Quality-Aware Consensuation Configuration ---
QUALITY_CONFIG = {
    'high_quality_threshold': 0.8,     # Transforms for reference consensus
    'min_inclusion_quality': 0.3,      # Minimum quality to be considered
    'consistency_distance_base': 20.0,  # Base distance threshold (pixels)
    'quality_exponent': 2.0,           # How much to emphasize quality in weighting
    'fallback_to_single_best': True,   # Use best single prediction if consensus fails
    'adaptive_threshold_multiplier': 2.0,  # Multiplier for adaptive thresholds
}

# --- Consensus Validation Configuration ---
VALIDATION_CONFIG = {
    'center_deviation_threshold': 7.0,    #  More lenient center tolerance
    'angle_deviation_threshold': 15.0,     #  More lenient angle tolerance  
    'area_deviation_threshold': 0.4,       #  Allow up to 40% area difference
    'center_weight': 0.45,                 #  Less emphasis on center precision
    'angle_weight': 0.45,                  #  Less emphasis on angle precision
    'area_weight': 0.1,                    #  More tolerance for size variation
}

def update_tta_configs(tta_config=None, quality_config=None, validation_config=None):
    """
    Update TTA configuration parameters.
    
    Args:
        tta_config: Dictionary with TTA_CONFIG updates
        quality_config: Dictionary with QUALITY_CONFIG updates  
        validation_config: Dictionary with VALIDATION_CONFIG updates
    """
    global TTA_CONFIG, QUALITY_CONFIG, VALIDATION_CONFIG
    
    if tta_config:
        TTA_CONFIG.update(tta_config)
    if quality_config:
        QUALITY_CONFIG.update(quality_config)
    if validation_config:
        VALIDATION_CONFIG.update(validation_config)

def get_current_configs():
    """Get current configuration values."""
    return {
        'tta_config': TTA_CONFIG.copy(),
        'quality_config': QUALITY_CONFIG.copy(),
        'validation_config': VALIDATION_CONFIG.copy()
    }

# --- Forward Transformations ---

def identity_transform(image: torch.Tensor) -> torch.Tensor:
    """Identity transformation (no change)."""
    return image

def hflip_transform(image: torch.Tensor) -> torch.Tensor:
    """Horizontal flip transformation."""
    return F.hflip(image)

def rotate_transform(image: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate image by a given angle."""
    return F.rotate(image, angle)

def brightness_contrast_transform(image: torch.Tensor) -> torch.Tensor:
    """Adjust brightness and contrast."""
    bright_image = F.adjust_brightness(image, TTA_CONFIG['brightness_factor'])
    return F.adjust_contrast(bright_image, TTA_CONFIG['contrast_factor'])

def gamma_transform(image: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply gamma correction to image."""
    return F.adjust_gamma(image, gamma)

# --- Reverse Coordinate Transformations ---

def identity_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int) -> Dict[str, torch.Tensor]:
    """Identity reverse transformation."""
    return pred_dict.copy()

def hflip_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int) -> Dict[str, torch.Tensor]:
    """Reverse horizontal flip for coordinates."""
    boxes = pred_dict['boxes'].clone()
    ellipse_params = pred_dict['ellipse_params'].clone()

    # Flip bounding boxes horizontally
    boxes[:, [0, 2]] = original_W - boxes[:, [2, 0]]
    
    # Flip ellipse centers horizontally and mirror angle
    ellipse_params[:, 2] = original_W - ellipse_params[:, 2]
    ellipse_params[:, 4] = -ellipse_params[:, 4]             # Mirror angle (theta, index 4)

    new_pred_dict = pred_dict.copy()
    new_pred_dict['boxes'] = boxes
    new_pred_dict['ellipse_params'] = ellipse_params
    new_pred_dict['labels'] = pred_dict['labels'] # Ensure labels are copied
    new_pred_dict['scores'] = pred_dict['scores'] # Ensure scores are copied
    return new_pred_dict

def rotate_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int, angle: float) -> Dict[str, torch.Tensor]:
    """Reverse rotation transformation for coordinates."""
    boxes = pred_dict['boxes'].clone()
    ellipse_params = pred_dict['ellipse_params'].clone()
    
    # Convert reverse angle to radians
    angle_rad = math.radians(angle) 
    
    # Image center for rotation (match torchvision's convention)
    center_x = original_W // 2
    center_y = original_H // 2

    # Rotate bounding boxes
    rotated_boxes = []
    if boxes.numel() > 0:
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            corners = torch.tensor([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
            ], dtype=torch.float32)

            corners_translated = corners - torch.tensor([center_x, center_y])
            
            # Clockwise rotation matrix
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            rot_matrix = torch.tensor([
                [cos_a, sin_a],
                [-sin_a, cos_a]
            ], dtype=torch.float32)
            
            rotated_corners = torch.matmul(corners_translated, rot_matrix.T)
            rotated_corners_translated_back = rotated_corners + torch.tensor([center_x, center_y])
            
            new_x1 = torch.min(rotated_corners_translated_back[:, 0])
            new_y1 = torch.min(rotated_corners_translated_back[:, 1])
            new_x2 = torch.max(rotated_corners_translated_back[:, 0])
            new_y2 = torch.max(rotated_corners_translated_back[:, 1])
            rotated_boxes.append(torch.tensor([new_x1, new_y1, new_x2, new_y2]))
    
    if len(rotated_boxes) > 0:
        boxes = torch.stack(rotated_boxes)
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)

    # Rotate ellipse centers and angles
    if ellipse_params.numel() > 0:
        x_coords = ellipse_params[:, 2]
        y_coords = ellipse_params[:, 3]
        
        # Translate to origin
        x_translated = x_coords - center_x
        y_translated = y_coords - center_y
        
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Apply clockwise rotation
        rotated_x = x_translated * cos_a + y_translated * sin_a
        rotated_y = -x_translated * sin_a + y_translated * cos_a
        
        # Translate back
        ellipse_params[:, 2] = rotated_x + center_x
        ellipse_params[:, 3] = rotated_y + center_y
        
        # Update ellipse angle (subtract for reversal)
        ellipse_params[:, 4] = ellipse_params[:, 4] - angle_rad

    new_pred_dict = pred_dict.copy()
    new_pred_dict['boxes'] = boxes
    new_pred_dict['ellipse_params'] = ellipse_params
    new_pred_dict['labels'] = pred_dict['labels']
    new_pred_dict['scores'] = pred_dict['scores']
    return new_pred_dict

def photometric_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int) -> Dict[str, torch.Tensor]:
    """Photometric transformations don't affect coordinates."""
    return pred_dict.copy()

# --- TTA Transform Definitions ---
# Selected high-performing transformations based on pixel error analysis

TTA_TRANSFORMS = [
    {
        'name': 'Original',
        'forward': (identity_transform, {}),
        'reverse': (identity_reverse, {}),
        'color': '#FF0000'  # Red
    },
    {
        'name': 'Gamma 0.7',
        'forward': (gamma_transform, {'gamma': 0.7}),
        'reverse': (photometric_reverse, {}),
        'color': '#00FF00'  # Green
    },
    {
        'name': 'Gamma 1.3',
        'forward': (gamma_transform, {'gamma': 1.3}),
        'reverse': (photometric_reverse, {}),
        'color': '#0000FF'  # Blue
    },
    {
        'name': 'Brightness/Contrast',
        'forward': (brightness_contrast_transform, {}),
        'reverse': (photometric_reverse, {}),
        'color': '#FFFF00'  # Yellow
    },
    {
        'name': 'Rotation +10°',
        'forward': (rotate_transform, {'angle': 10}),
        'reverse': (rotate_reverse, {'angle': -10}),
        'color': '#FF00FF'  # Magenta
    },
    {
        'name': 'Rotation -10°',
        'forward': (rotate_transform, {'angle': -10}),
        'reverse': (rotate_reverse, {'angle': 10}),
        'color': '#00FFFF'  # Cyan
    },
    {
        'name': 'Horizontal Flip',
        'forward': (hflip_transform, {}),
        'reverse': (hflip_reverse, {}),
        'color': '#FFA500'  # Orange
    },
]

# --- Transform Quality Scoring ---

def get_transform_quality(transform_name: str) -> float:
    """
    Assign quality scores to transformations based on expected performance.
    
    Args:
        transform_name: Name of the transformation
        
    Returns:
        Quality score between 0.0 and 1.0 (higher = better)
    """
    # Normalize transform name for consistent matching
    name = transform_name.lower()
    
    # High quality transforms (reliable, minimal degradation)
    if 'original' in name:
        return 1.0  # Perfect quality - no transformation
    elif 'horizontal flip' in name:
        return 0.95  # Very high quality - simple geometric transform
    elif 'brightness' in name or 'contrast' in name:
        return 0.85  # High quality - photometric enhancement
    elif 'gamma' in name:
        return 0.85  # High quality - gamma correction
    elif 'rotation' in name:
        if '10' in name:  # Small rotations
            return 0.75  # Good quality - small angle
        elif '15' in name:
            return 0.65  # Medium quality
        else:  # Larger rotations
            return 0.5   # Lower quality due to interpolation artifacts
    elif 'scale' in name:
        if '0.8' in name or '1.2' in name:  # Moderate scaling
            return 0.6   # Medium quality
        else:  # Extreme scaling
            return 0.4   # Lower quality
    else:
        return 0.5  # Default medium quality

# --- Deviation Calculation Functions ---

def calculate_ellipse_deviation(ellipse1: torch.Tensor, ellipse2: torch.Tensor) -> Dict[str, float]:
    """
    Calculate deviation metrics between two ellipses.
    
    Args:
        ellipse1, ellipse2: Ellipse tensors [a, b, cx, cy, theta]
    
    Returns:
        Dictionary with deviation metrics
    """
    # Center distance (pixels)
    center1 = ellipse1[2:4]  # [cx, cy]
    center2 = ellipse2[2:4]
    center_distance = torch.norm(center2 - center1).item()
    
    # Angle difference (degrees)
    angle1 = ellipse1[4].item()
    angle2 = ellipse2[4].item()
    angle_diff = abs(angle2 - angle1)
    # Handle angle wrapping
    angle_diff = min(angle_diff, math.pi - angle_diff)
    angle_diff_deg = math.degrees(angle_diff)
    
    # Area difference (relative)
    area1 = math.pi * ellipse1[0].item() * ellipse1[1].item()
    area2 = math.pi * ellipse2[0].item() * ellipse2[1].item()
    area_diff = abs(area2 - area1) / max(area1, 1e-6)
    
    return {
        'center_distance': center_distance,
        'angle_difference': angle_diff_deg,
        'area_difference': area_diff
    }

def calculate_weighted_deviation_score(deviation: Dict[str, float]) -> float:
    """
    Calculate weighted deviation score from individual metrics.
    
    Args:
        deviation: Dictionary with center_distance, angle_difference, area_difference
    
    Returns:
        Weighted deviation score (0 = identical, higher = more different)
    """
    center_norm = deviation['center_distance'] / VALIDATION_CONFIG['center_deviation_threshold']
    angle_norm = deviation['angle_difference'] / VALIDATION_CONFIG['angle_deviation_threshold']
    area_norm = deviation['area_difference'] / VALIDATION_CONFIG['area_deviation_threshold']
    
    weighted_score = (
        VALIDATION_CONFIG['center_weight'] * center_norm +
        VALIDATION_CONFIG['angle_weight'] * angle_norm +
        VALIDATION_CONFIG['area_weight'] * area_norm
    )
    
    return weighted_score

def is_ellipse_consensus_valid(individual_ellipses: List[torch.Tensor], 
                             consensus_ellipse: torch.Tensor) -> bool:
    """
    Validate if consensus ellipse is acceptable based on individual deviations.
    
    Args:
        individual_ellipses: List of individual ellipse predictions
        consensus_ellipse: Consensuated ellipse
    
    Returns:
        True if consensus is valid, False if it should be rejected
    """
    if not individual_ellipses:
        return False
    
    total_deviation = 0.0
    valid_predictions = 0
    
    for ellipse in individual_ellipses:
        deviation = calculate_ellipse_deviation(ellipse, consensus_ellipse)
        
        # Check individual thresholds
        if (deviation['center_distance'] <= VALIDATION_CONFIG['center_deviation_threshold'] and
            deviation['angle_difference'] <= VALIDATION_CONFIG['angle_deviation_threshold'] and
            deviation['area_difference'] <= VALIDATION_CONFIG['area_deviation_threshold']):
            
            weighted_score = calculate_weighted_deviation_score(deviation)
            total_deviation += weighted_score
            valid_predictions += 1
    
    # Require at least 3 predictions within thresholds for valid consensus
    if valid_predictions >= 3:
        avg_deviation = total_deviation / valid_predictions
        return avg_deviation <= 1.0  # Normalized threshold
    
    return False

# --- Quality-Aware Consensuation Functions ---

def consensuate_predictions(predictions_list: List[Dict[str, torch.Tensor]], 
                          transform_names: List[str] = None,
                          min_score: float = None, 
                          distance_threshold: float = None) -> Dict[str, torch.Tensor]:
    """
    Quality-aware consensuation of ellipse predictions from multiple TTA transformations.
    
    Uses a two-stage approach:
    1. Establish reference consensus from high-quality transforms
    2. Selectively integrate medium/low quality predictions based on consistency
    
    Args:
        predictions_list: List of prediction dictionaries
        transform_names: List of transform names (for quality scoring)
        min_score: Minimum confidence score threshold
        distance_threshold: Base distance threshold for grouping (pixels)
    
    Returns:
        Consensuated prediction dictionary
    """
    if min_score is None:
        min_score = TTA_CONFIG['min_score_threshold']
    if distance_threshold is None:
        distance_threshold = QUALITY_CONFIG['consistency_distance_base']
    
    # Detect device from the first available tensor
    device = torch.device('cpu')  # Default to CPU
    if predictions_list:
        for pred_dict in predictions_list:
            if pred_dict["ellipse_params"].numel() > 0:
                device = pred_dict["ellipse_params"].device
                break
            elif pred_dict["scores"].numel() > 0:
                device = pred_dict["scores"].device
                break
    
    if not predictions_list:
        return _empty_prediction_dict(device)
    
    # If no transform names provided, create dummy names
    if transform_names is None:
        transform_names = [f"Transform_{i}" for i in range(len(predictions_list))]
    
    # Ensure we have the same number of names as predictions
    if len(transform_names) != len(predictions_list):
        print(f"⚠️  Warning: {len(transform_names)} transform names for {len(predictions_list)} predictions")
        transform_names = transform_names[:len(predictions_list)]
        while len(transform_names) < len(predictions_list):
            transform_names.append("Unknown")
    
    # Stage 1: Collect and score all valid predictions
    scored_predictions = []
    
    for i, (pred_dict, transform_name) in enumerate(zip(predictions_list, transform_names)):
        if pred_dict["ellipse_params"].numel() == 0:
            continue
        
        # Ensure scores are on the same device for comparison
        scores_tensor = pred_dict["scores"].to(device)
        
        # Filter by confidence threshold
        score_mask = scores_tensor > min_score
        if not score_mask.any():
            continue
        
        # Get quality score for this transform
        transform_quality = get_transform_quality(transform_name)
        
        # Extract valid predictions (ensure all tensors are on same device)
        ellipse_params = pred_dict["ellipse_params"].to(device)[score_mask]
        scores = scores_tensor[score_mask]
        labels = pred_dict["labels"].to(device)[score_mask]
        boxes = pred_dict["boxes"].to(device)[score_mask]
        
        # Store each individual prediction with metadata
        for j in range(len(ellipse_params)):
            scored_predictions.append({
                'ellipse': ellipse_params[j],
                'confidence': scores[j].item(),
                'label': labels[j],
                'box': boxes[j],
                'transform_name': transform_name,
                'transform_quality': transform_quality,
                'prediction_index': i,
                'within_prediction_index': j
            })
    
    if not scored_predictions:
        return _empty_prediction_dict(device)
    
    # Stage 2: Establish reference consensus from high-quality transforms
    high_quality_preds = [p for p in scored_predictions 
                         if p['transform_quality'] >= QUALITY_CONFIG['high_quality_threshold']]
    
    if high_quality_preds:
        # Use high-quality predictions to establish reference
        reference_consensus = _establish_reference_consensus(high_quality_preds, distance_threshold)
    else:
        # Fallback: use best available predictions
        print("🔄 No high-quality transforms available, using best available predictions")
        scored_predictions.sort(key=lambda x: x['transform_quality'] * x['confidence'], reverse=True)
        reference_consensus = _establish_reference_consensus(scored_predictions, distance_threshold)
    
    if not reference_consensus:
        # Ultimate fallback: return single best prediction
        if QUALITY_CONFIG['fallback_to_single_best'] and scored_predictions:
            best_pred = max(scored_predictions, key=lambda x: x['transform_quality'] * x['confidence'])
            return _single_prediction_to_dict(best_pred)
        else:
            return _empty_prediction_dict(device)
    
    # Stage 3: Selectively integrate remaining predictions
    final_consensus = []
    
    for ref_center, ref_group in reference_consensus:
        # Start with reference group
        consensus_group = ref_group.copy()
        
        # Try to add consistent predictions from remaining transforms
        remaining_preds = [p for p in scored_predictions 
                          if p['transform_quality'] < QUALITY_CONFIG['high_quality_threshold']
                          and p['transform_quality'] >= QUALITY_CONFIG['min_inclusion_quality']]
        
        for pred in remaining_preds:
            pred_center = pred['ellipse'][2:4]  # cx, cy
            # Ensure both tensors are on the same device
            pred_center = pred_center.to(ref_center.device)
            distance_to_ref = torch.norm(pred_center - ref_center).item()
            
            # Adaptive threshold based on transform quality
            adaptive_threshold = distance_threshold * (QUALITY_CONFIG['adaptive_threshold_multiplier'] - pred['transform_quality'])
            
            if distance_to_ref < adaptive_threshold:
                consensus_group.append(pred)
        
        if consensus_group:  # Should always be true since we start with ref_group
            final_consensus.append(consensus_group)
    
    # Stage 4: Generate final consensuated predictions
    return _generate_final_consensus(final_consensus)

def _establish_reference_consensus(predictions: List[Dict], distance_threshold: float) -> List[Tuple[torch.Tensor, List[Dict]]]:
    """
    Establish reference consensus groups from high-quality predictions.
    
    Returns:
        List of (center, group) tuples where center is reference center and group is list of predictions
    """
    if not predictions:
        return []
    
    consensus_groups = []
    used_indices = set()
    
    for i, pred in enumerate(predictions):
        if i in used_indices:
            continue
            
        current_center = pred['ellipse'][2:4]  # cx, cy
        group = [pred]
        used_indices.add(i)
        
        # Find nearby predictions
        for j, other_pred in enumerate(predictions):
            if j in used_indices:
                continue
                
            other_center = other_pred['ellipse'][2:4]
            # Ensure both centers are on the same device
            other_center = other_center.to(current_center.device)
            distance = torch.norm(current_center - other_center).item()
            
            if distance < distance_threshold:
                group.append(other_pred)
                used_indices.add(j)
        
        if group:
            # Calculate group consensus center for reference
            # Ensure all tensors are on the same device
            first_device = group[0]['ellipse'].device
            group_centers = torch.stack([p['ellipse'][2:4].to(first_device) for p in group])
            consensus_center = torch.mean(group_centers, dim=0)
            consensus_groups.append((consensus_center, group))
    
    return consensus_groups

def _generate_final_consensus(consensus_groups: List[List[Dict]]) -> Dict[str, torch.Tensor]:
    """
    Generate final consensuated predictions from consensus groups.
    """
    # Detect device from first available tensor
    device = torch.device('cpu')
    if consensus_groups:
        for group in consensus_groups:
            if group and 'ellipse' in group[0]:
                device = group[0]['ellipse'].device
                break
    
    if not consensus_groups:
        return _empty_prediction_dict(device)
    
    final_ellipses = []
    final_scores = []
    final_labels = []
    final_boxes = []
    
    for group in consensus_groups:
        if not group:
            continue
            
        # Calculate quality-weighted consensus for this group
        ellipses = torch.stack([p['ellipse'] for p in group])
        confidences = torch.tensor([p['confidence'] for p in group], device=device)
        qualities = torch.tensor([p['transform_quality'] for p in group], device=device)
        
        # Combined weights: quality^exponent * confidence
        quality_weights = torch.pow(qualities, QUALITY_CONFIG['quality_exponent'])
        combined_weights = quality_weights * confidences
        combined_weights = combined_weights / torch.sum(combined_weights)  # Normalize
        
        # Quality-weighted consensus ellipse
        consensus_ellipse = _weighted_ellipse_consensus_quality(ellipses, combined_weights)
        
        # Validate consensus based on individual deviations
        individual_ellipses = [p['ellipse'] for p in group]
        if not is_ellipse_consensus_valid(individual_ellipses, consensus_ellipse):
            continue  # Skip this consensus group if validation fails
        
        # Consensus score (weighted average)
        consensus_score = torch.sum(confidences * combined_weights)
        
        # Consensus label (mode of labels)
        labels = torch.tensor([p['label'] for p in group], device=device)
        consensus_label = torch.mode(labels).values
        
        # Consensus box (weighted average)
        boxes = torch.stack([p['box'] for p in group])
        consensus_box = torch.sum(boxes.float() * combined_weights.unsqueeze(1), dim=0)
        
        final_ellipses.append(consensus_ellipse)
        final_scores.append(consensus_score)
        final_labels.append(consensus_label)
        final_boxes.append(consensus_box)
    
    if final_ellipses:
        return {
            'boxes': torch.stack(final_boxes),
            'ellipse_params': torch.stack(final_ellipses),
            'labels': torch.stack(final_labels),
            'scores': torch.stack(final_scores),
        }
    else:
        return _empty_prediction_dict(device)

def _weighted_l1_median(X: np.ndarray, WX: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Computes weighted geometric median (L1-median)
    :param X: the list of sample points, a 2D ndarray
    :param WX: the list of weights  
    :param eps: acceptable error margin
    :return: first estimate meeting eps
    """
    y = np.average(X, axis=0, weights=WX)
    while True:
        while np.any(cdist(X, [y]) == 0):
            y += 0.1 * np.ones(len(y))
        W = np.expand_dims(WX, axis=1) / cdist(X, [y])  # element-wise operation
        y1 = np.sum(W * X, 0) / np.sum(W)
        if euclidean(y, y1) < eps:
            return y1
        y = y1

def _weighted_median(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted median of tensor values.
    
    Args:
        values: Tensor of values to find median of
        weights: Tensor of weights corresponding to values
        
    Returns:
        Weighted median value
    """
    # Sort values and corresponding weights
    sorted_indices = torch.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]
    
    # Calculate cumulative weights
    cumulative_weights = torch.cumsum(sorted_weights, dim=0)
    total_weight = torch.sum(weights)
    
    # Find the median position
    median_weight = total_weight / 2.0
    
    # Find the index where cumulative weight exceeds median_weight
    median_idx = torch.searchsorted(cumulative_weights, median_weight, right=True)
    median_idx = torch.clamp(median_idx, 0, len(sorted_values) - 1)
    
    return sorted_values[median_idx]

def _weighted_ellipse_consensus_quality(ellipses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute quality-weighted consensus of ellipse parameters using weighted-l1-median for centers
    and weighted median/mean for other parameters.
    
    Args:
        ellipses: Tensor of shape (N, 5) containing [a, b, cx, cy, theta]
        weights: Tensor of shape (N,) containing normalized weights
        
    Returns:
        Consensus ellipse tensor of shape (5,)
    """
    if len(ellipses) == 1:
        return ellipses[0]
    
    device = ellipses.device
    
    # Semi-axes: weighted median for robustness
    a_consensus = _weighted_median(ellipses[:, 0], weights)
    b_consensus = _weighted_median(ellipses[:, 1], weights)
    
    # Centers: weighted L1-median for robustness against outliers
    centers = ellipses[:, 2:4].cpu().numpy()  # Convert to numpy for L1-median computation
    weights_np = weights.cpu().numpy()
    center_consensus = _weighted_l1_median(centers, weights_np)
    cx_consensus = torch.tensor(center_consensus[0], device=device)
    cy_consensus = torch.tensor(center_consensus[1], device=device)
    
    # Angles: weighted circular mean (this is already robust for angles)
    angles = ellipses[:, 4]
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    weighted_cos = torch.sum(cos_angles * weights)
    weighted_sin = torch.sum(sin_angles * weights)
    theta_consensus = torch.atan2(weighted_sin, weighted_cos)
    
    return torch.tensor([a_consensus, b_consensus, cx_consensus, cy_consensus, theta_consensus], 
                       device=device)

def _single_prediction_to_dict(pred: Dict) -> Dict[str, torch.Tensor]:
    """Convert a single prediction to the standard prediction dictionary format."""
    return {
        'boxes': pred['box'].unsqueeze(0),
        'ellipse_params': pred['ellipse'].unsqueeze(0),
        'labels': pred['label'].unsqueeze(0),
        'scores': torch.tensor([pred['confidence']]),
    }

def apply_nms_to_predictions(pred_dict: Dict[str, torch.Tensor], 
                           iou_threshold: float = None) -> Dict[str, torch.Tensor]:
    """
    Apply Non-Maximum Suppression to prediction results.
    
    Args:
        pred_dict: Dictionary containing 'boxes', 'ellipse_params', 'labels', 'scores'
        iou_threshold: IoU threshold for NMS (default from config)
        
    Returns:
        Filtered prediction dictionary after NMS
    """
    if iou_threshold is None:
        iou_threshold = TTA_CONFIG['nms_iou_threshold']
    
    # Skip if no predictions
    if pred_dict['ellipse_params'].numel() == 0:
        return pred_dict
    
    boxes = pred_dict['boxes']
    scores = pred_dict['scores']
    
    # Apply NMS using bounding boxes
    keep_indices = nms(boxes, scores, iou_threshold)
    
    # Filter all predictions based on NMS results
    filtered_pred = {
        'boxes': boxes[keep_indices],
        'ellipse_params': pred_dict['ellipse_params'][keep_indices],
        'labels': pred_dict['labels'][keep_indices],
        'scores': scores[keep_indices],
    }
    
    return filtered_pred

def _empty_prediction_dict(device: torch.device = None) -> Dict[str, torch.Tensor]:
    """Return an empty prediction dictionary."""
    if device is None:
        device = torch.device('cpu')
    return {
        'boxes': torch.empty((0, 4), device=device),
        'ellipse_params': torch.empty((0, 5), device=device),
        'labels': torch.empty((0,), dtype=torch.int64, device=device),
        'scores': torch.empty((0,), dtype=torch.float32, device=device),
    }

# --- Main TTA Functions ---

def tta_predict(model: Any, image_tensor: torch.Tensor, device: torch.device, 
                min_score: float = None, consensuate: bool = True, 
                visualize: bool = False,
                apply_nms: bool = True, nms_iou_threshold: float = None) -> List[Dict[str, torch.Tensor]]:
    """
    Perform Test Time Augmentation prediction on an image.
    
    Args:
        model: The ellipse detection model
        image_tensor: Input image tensor [C, H, W]
        device: Device to run inference on
        min_score: Minimum confidence threshold
        consensuate: Whether to consensuate predictions
        visualize: Whether to show visualization
        apply_nms: Whether to apply Non-Maximum Suppression (default from config)
        nms_iou_threshold: IoU threshold for NMS (default from config)
    
    Returns:
        List containing prediction dictionary
    """
    if min_score is None:
        min_score = TTA_CONFIG['min_score_threshold']
    if apply_nms is None:
        apply_nms = TTA_CONFIG['apply_nms']
    if nms_iou_threshold is None:
        nms_iou_threshold = TTA_CONFIG['nms_iou_threshold']
    
    model.eval()
    model.to(device)
    
    original_H, original_W = image_tensor.shape[1:]
    all_predictions = []
    all_transform_names = []
    
    # Visualization setup if requested
    if visualize:
        fig_combined, ax_combined = plt.subplots(1, 1, figsize=(12, 10))
        fig_combined.patch.set_alpha(0)
        ax_combined.imshow(image_tensor.permute(1, 2, 0).cpu(), cmap="grey")
        legend_elements = []
    
    # Apply each TTA transformation
    for transform_config in TTA_TRANSFORMS:
        forward_func, forward_kwargs = transform_config['forward']
        reverse_func, reverse_kwargs = transform_config['reverse']
        transform_name = transform_config['name']
        
        # Apply forward transformation
        augmented_img = forward_func(image_tensor, **forward_kwargs)
        
        # Run inference
        with torch.no_grad():
            predictions = model([augmented_img.to(device)])
        
        pred_dict = predictions[0]
        
        # Skip if no predictions
        if pred_dict["ellipse_params"].numel() == 0:
            continue
        
        # Apply reverse transformation
        reversed_pred = reverse_func(pred_dict, original_H, original_W, **reverse_kwargs)
        all_predictions.append(reversed_pred)
        all_transform_names.append(transform_name)
        
        # Add to visualization if requested
        if visualize:
            score_mask = reversed_pred["scores"] > min_score
            if score_mask.any():
                _plot_ellipses(reversed_pred["ellipse_params"][score_mask], 
                             ax_combined, transform_config['color'])
                legend_elements.append(plt.Line2D([0], [0], color=transform_config['color'], 
                                                lw=4, label=transform_config['name']))
    
    # Consensuate predictions if requested
    if consensuate and all_predictions:
        final_predictions = consensuate_predictions(all_predictions, all_transform_names, min_score)
    elif all_predictions:
        # Concatenate all predictions without consensuation
        final_predictions = {
            'boxes': torch.cat([p['boxes'] for p in all_predictions]),
            'ellipse_params': torch.cat([p['ellipse_params'] for p in all_predictions]),
            'labels': torch.cat([p['labels'] for p in all_predictions]),
            'scores': torch.cat([p['scores'] for p in all_predictions]),
        }
    else:
        final_predictions = _empty_prediction_dict(device)
    
    # Apply NMS if requested and we have predictions
    if apply_nms and final_predictions['boxes'].numel() > 0:
        final_predictions = apply_nms_to_predictions(final_predictions, nms_iou_threshold)
    
    # Finalize visualization
    if visualize and legend_elements:
        ax_combined.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        ax_combined.set_title("TTA Predictions (Individual Transformations)")
        plt.tight_layout()
        plt.show(block=False)
        
        # Show consensuated results if available
        if consensuate and final_predictions['ellipse_params'].numel() > 0:
            fig_consensus, ax_consensus = plt.subplots(1, 1, figsize=(12, 10))
            fig_consensus.patch.set_alpha(0)
            ax_consensus.imshow(image_tensor.permute(1, 2, 0).cpu(), cmap="grey")
            
            score_mask = final_predictions["scores"] > min_score
            if score_mask.any():
                _plot_ellipses(final_predictions["ellipse_params"][score_mask], 
                             ax_consensus, 'white')
            
            ax_consensus.set_title("Consensuated Predictions (Quality-Aware Method) - White", fontsize=14)
            plt.show(block=False)
    
    return [final_predictions]

def _plot_ellipses(ellipse_params: torch.Tensor, ax: Axes, color: str, alpha: float = 1.0) -> None:
    """Helper function to plot ellipses."""
    if ellipse_params.numel() == 0:
        return
    
    a, b, cx, cy, theta = ellipse_params.unbind(-1)
    a, b, theta, cx, cy = map(lambda t: t.detach().cpu().numpy(), (a, b, theta, cx, cy))
    
    ec = EllipseCollection(
        a * 2, b * 2, np.degrees(theta),
        units="xy", offsets=np.column_stack((cx, cy)),
        transOffset=ax.transData, facecolors="None",
        edgecolors=color, linewidths=2.5, alpha=alpha
    )
    ax.add_collection(ec)


