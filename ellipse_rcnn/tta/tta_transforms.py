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

# --- Quality-Aware Consensuation Configuration ---
QUALITY_CONFIG = {
    'high_quality_threshold': 0.8,     # Transforms for reference consensus
    'min_inclusion_quality': 0.3,      # Minimum quality to be considered
    'consistency_distance_base': 20.0,  # Base distance threshold (pixels)
    'quality_exponent': 2.0,           # How much to emphasize quality in weighting
    'fallback_to_single_best': True,   # Use best single prediction if consensus fails
    'adaptive_threshold_multiplier': 2.0,  # Multiplier for adaptive thresholds
}

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

# --- Original Simple Consensuation Function ---

def consensuate_predictions_simple(predictions_list: List[Dict[str, torch.Tensor]], 
                          min_score: float = None, 
                          distance_threshold: float = None,
                          size_similarity_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Simple consensuation of ellipse predictions from multiple TTA transformations.
    
    Args:
        predictions_list: List of prediction dictionaries
        min_score: Minimum confidence score threshold
        distance_threshold: Maximum distance for grouping ellipses (pixels)
        size_similarity_threshold: Maximum relative size difference for grouping (0-1)
    
    Returns:
        Consensuated prediction dictionary
    """
    if min_score is None:
        min_score = TTA_CONFIG['min_score_threshold']
    if distance_threshold is None:
        distance_threshold = TTA_CONFIG['consensuation_distance_threshold']
    
    if not predictions_list:
        return _empty_prediction_dict()
    
    # Filter and collect valid predictions
    valid_ellipses = []
    valid_scores = []
    valid_labels = []
    valid_boxes = []
    
    for pred_dict in predictions_list:
        if pred_dict["ellipse_params"].numel() == 0:
            continue
            
        score_mask = pred_dict["scores"] > min_score
        if score_mask.any():
            valid_ellipses.append(pred_dict["ellipse_params"][score_mask])
            valid_scores.append(pred_dict["scores"][score_mask])
            valid_labels.append(pred_dict["labels"][score_mask])
            valid_boxes.append(pred_dict["boxes"][score_mask])
    
    if not valid_ellipses:
        return _empty_prediction_dict()
    
    # Concatenate all predictions
    all_ellipses = torch.cat(valid_ellipses, dim=0)
    all_scores = torch.cat(valid_scores, dim=0)
    all_labels = torch.cat(valid_labels, dim=0)
    all_boxes = torch.cat(valid_boxes, dim=0)
    
    # Group nearby and similar ellipses
    consensuated_ellipses = []
    consensuated_scores = []
    consensuated_labels = []
    consensuated_boxes = []
    
    used_indices = set()
    
    for i in range(len(all_ellipses)):
        if i in used_indices:
            continue
            
        # Find ellipses close to current one
        current_center = all_ellipses[i, 2:4]  # cx, cy
        current_size = all_ellipses[i, 0:2]    # a, b
        group_indices = [i]
        
        for j in range(i + 1, len(all_ellipses)):
            if j in used_indices:
                continue
                
            other_center = all_ellipses[j, 2:4]
            other_size = all_ellipses[j, 0:2]
            
            # Check center distance
            center_distance = torch.norm(current_center - other_center).item()
            
            # Check size similarity (relative difference)
            size_ratio = torch.min(current_size / (other_size + 1e-8))  # Avoid division by zero
            size_similarity = size_ratio.item()
            
            # Group if both center distance and size similarity are within thresholds
            if (center_distance < distance_threshold and 
                size_similarity > size_similarity_threshold):
                group_indices.append(j)
        
        used_indices.update(group_indices)
        
        # Consensuate the group with score weighting
        group_ellipses = all_ellipses[group_indices]
        group_scores = all_scores[group_indices]
        group_labels = all_labels[group_indices]
        group_boxes = all_boxes[group_indices]
        
        # Calculate consensuated parameters with score weighting
        consensuated_ellipse = _consensuate_ellipse_group_weighted(group_ellipses, group_scores)
        consensuated_score = torch.mean(group_scores)
        consensuated_label = torch.mode(group_labels).values
        consensuated_box = torch.mean(group_boxes, dim=0)
        
        consensuated_ellipses.append(consensuated_ellipse)
        consensuated_scores.append(consensuated_score)
        consensuated_labels.append(consensuated_label)
        consensuated_boxes.append(consensuated_box)
    
    if consensuated_ellipses:
        return {
            'boxes': torch.stack(consensuated_boxes),
            'ellipse_params': torch.stack(consensuated_ellipses),
            'labels': torch.stack(consensuated_labels),
            'scores': torch.stack(consensuated_scores),
        }
    else:
        return _empty_prediction_dict()

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
        top_predictions = scored_predictions[:min(3, len(scored_predictions))]  # Use top 3
        reference_consensus = _establish_reference_consensus(top_predictions, distance_threshold)
    
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

def _weighted_ellipse_consensus_quality(ellipses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute quality-weighted consensus of ellipse parameters.
    
    Args:
        ellipses: Tensor of shape (N, 5) containing [a, b, cx, cy, theta]
        weights: Tensor of shape (N,) containing normalized weights
        
    Returns:
        Consensus ellipse tensor of shape (5,)
    """
    if len(ellipses) == 1:
        return ellipses[0]
    
    device = ellipses.device
    
    # Semi-axes: weighted mean
    a_consensus = torch.sum(ellipses[:, 0] * weights)
    b_consensus = torch.sum(ellipses[:, 1] * weights)
    
    # Centers: weighted mean
    cx_consensus = torch.sum(ellipses[:, 2] * weights)
    cy_consensus = torch.sum(ellipses[:, 3] * weights)
    
    # Angles: weighted circular mean
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
    """
    Consensuate ellipse predictions from multiple TTA transformations.
    
    Args:
        predictions_list: List of prediction dictionaries
        min_score: Minimum confidence score threshold
        distance_threshold: Maximum distance for grouping ellipses (pixels)
        size_similarity_threshold: Maximum relative size difference for grouping (0-1)
    
    Returns:
        Consensuated prediction dictionary
    """
    if min_score is None:
        min_score = TTA_CONFIG['min_score_threshold']
    if distance_threshold is None:
        distance_threshold = TTA_CONFIG['consensuation_distance_threshold']
    
    if not predictions_list:
        return _empty_prediction_dict()
    
    # Filter and collect valid predictions
    valid_ellipses = []
    valid_scores = []
    valid_labels = []
    valid_boxes = []
    
    for pred_dict in predictions_list:
        if pred_dict["ellipse_params"].numel() == 0:
            continue
            
        score_mask = pred_dict["scores"] > min_score
        if score_mask.any():
            valid_ellipses.append(pred_dict["ellipse_params"][score_mask])
            valid_scores.append(pred_dict["scores"][score_mask])
            valid_labels.append(pred_dict["labels"][score_mask])
            valid_boxes.append(pred_dict["boxes"][score_mask])
    
    if not valid_ellipses:
        return _empty_prediction_dict()
    
    # Concatenate all predictions
    all_ellipses = torch.cat(valid_ellipses, dim=0)
    all_scores = torch.cat(valid_scores, dim=0)
    all_labels = torch.cat(valid_labels, dim=0)
    all_boxes = torch.cat(valid_boxes, dim=0)
    
    # Group nearby and similar ellipses
    consensuated_ellipses = []
    consensuated_scores = []
    consensuated_labels = []
    consensuated_boxes = []
    
    used_indices = set()
    
    for i in range(len(all_ellipses)):
        if i in used_indices:
            continue
            
        # Find ellipses close to current one
        current_center = all_ellipses[i, 2:4]  # cx, cy
        current_size = all_ellipses[i, 0:2]    # a, b
        group_indices = [i]
        
        for j in range(i + 1, len(all_ellipses)):
            if j in used_indices:
                continue
                
            other_center = all_ellipses[j, 2:4]
            other_size = all_ellipses[j, 0:2]
            
            # Check center distance
            center_distance = torch.norm(current_center - other_center).item()
            
            # Check size similarity (relative difference)
            size_ratio = torch.min(current_size / (other_size + 1e-8))  # Avoid division by zero
            size_similarity = size_ratio.item()
            
            # Group if both center distance and size similarity are within thresholds
            if (center_distance < distance_threshold and 
                size_similarity > size_similarity_threshold):
                group_indices.append(j)
        
        used_indices.update(group_indices)
        
        # Consensuate the group with score weighting
        group_ellipses = all_ellipses[group_indices]
        group_scores = all_scores[group_indices]
        group_labels = all_labels[group_indices]
        group_boxes = all_boxes[group_indices]
        
        # Calculate consensuated parameters with score weighting
        consensuated_ellipse = _consensuate_ellipse_group_weighted(group_ellipses, group_scores)
        consensuated_score = torch.mean(group_scores)
        consensuated_label = torch.mode(group_labels).values
        consensuated_box = torch.mean(group_boxes, dim=0)
        
        consensuated_ellipses.append(consensuated_ellipse)
        consensuated_scores.append(consensuated_score)
        consensuated_labels.append(consensuated_label)
        consensuated_boxes.append(consensuated_box)
    
    if consensuated_ellipses:
        return {
            'boxes': torch.stack(consensuated_boxes),
            'ellipse_params': torch.stack(consensuated_ellipses),
            'labels': torch.stack(consensuated_labels),
            'scores': torch.stack(consensuated_scores),
        }
    else:
        return _empty_prediction_dict()

def _consensuate_ellipse_group(ellipses: torch.Tensor) -> torch.Tensor:
    """Consensuate a group of ellipses using appropriate statistics."""
    # Semi-axes: mean
    a_consensuated = torch.mean(ellipses[:, 0])
    b_consensuated = torch.mean(ellipses[:, 1])
    
    # Centers: median (robust to outliers)
    cx_consensuated = torch.median(ellipses[:, 2])
    cy_consensuated = torch.median(ellipses[:, 3])
    
    # Angles: circular mean (handle both real and complex operations properly)
    angles = ellipses[:, 4]
    if len(angles) == 1:
        theta_consensuated = angles[0]
    else:
        # Use real tensors for cos and sin, then combine for circular mean
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)
        mean_cos = torch.mean(cos_angles)
        mean_sin = torch.mean(sin_angles)
        theta_consensuated = torch.atan2(mean_sin, mean_cos)
    
    return torch.tensor([a_consensuated, b_consensuated, cx_consensuated, cy_consensuated, theta_consensuated])

def _consensuate_ellipse_group_weighted(ellipses: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Consensuate a group of ellipses using score-weighted statistics."""
    if len(ellipses) == 1:
        return ellipses[0]
    
    # Normalize weights
    weights = scores / torch.sum(scores)
    
    # Semi-axes: weighted mean
    a_consensuated = torch.sum(ellipses[:, 0] * weights)
    b_consensuated = torch.sum(ellipses[:, 1] * weights)
    
    # Centers: weighted mean (more reliable than median with weights)
    cx_consensuated = torch.sum(ellipses[:, 2] * weights)
    cy_consensuated = torch.sum(ellipses[:, 3] * weights)
    
    # Angles: weighted circular mean
    angles = ellipses[:, 4]
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    weighted_cos = torch.sum(cos_angles * weights)
    weighted_sin = torch.sum(sin_angles * weights)
    theta_consensuated = torch.atan2(weighted_sin, weighted_cos)
    
    return torch.tensor([a_consensuated, b_consensuated, cx_consensuated, cy_consensuated, theta_consensuated])

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
                visualize: bool = False, consensuation_method: str = 'quality',
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
        consensuation_method: 'simple' or 'quality' for consensuation method
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
        if consensuation_method == 'quality':
            final_predictions = consensuate_predictions(all_predictions, all_transform_names, min_score)
        else:  # 'simple'
            final_predictions = consensuate_predictions_simple(all_predictions, min_score)
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
    
    # Apply Non-Maximum Suppression to reduce duplicates
    if TTA_CONFIG['apply_nms'] and final_predictions['ellipse_params'].numel() > 0:
        final_predictions = apply_nms_to_predictions(final_predictions)
    
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
            
            ax_consensus.set_title(f"Consensuated Predictions ({consensuation_method.capitalize()} Method) - White", fontsize=14)
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


