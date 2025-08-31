"""
Test Time Augmentation (TTA) for Ellipse R-CNN.

This module provides clean, robust TTA functionality with proper coordinate transformations
and prediction consensuation for improved ellipse detection performance.
"""

import torch
import torchvision.transforms.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Callable, Dict, Any, Tuple
from matplotlib.collections import EllipseCollection
from matplotlib.axes import Axes

# --- Configuration ---
TTA_CONFIG = {
    'rotation_angle': 10,  # degrees
    'brightness_factor': 1.2,
    'contrast_factor': 1.2,
    'min_score_threshold': 0.5, 
    'consensuation_distance_threshold': 30.0, 
    # New transformation parameters
    'scale_factors': [0.8, 1.2, 1.5],  # Multi-scale testing scales
    'gamma_values': [0.7, 1.3],  # Gamma correction values
    'additional_rotations': [15, 45, 90],  # Additional rotation angles
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

# --- Forward Transformations ---

def identity_transform(image: torch.Tensor) -> torch.Tensor:
    """Identity transformation (no change)."""
    return image

def hflip_transform(image: torch.Tensor) -> torch.Tensor:
    """Horizontal flip transformation."""
    return F.hflip(image)

def rotate_transform(image: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate image by specified angle (degrees)."""
    return F.rotate(image, angle)

def brightness_contrast_transform(image: torch.Tensor) -> torch.Tensor:
    """Adjust brightness and contrast."""
    bright_image = F.adjust_brightness(image, TTA_CONFIG['brightness_factor'])
    return F.adjust_contrast(bright_image, TTA_CONFIG['contrast_factor'])

def scale_transform(image: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """Scale image by specified factor."""
    original_size = image.shape[-2:]  # (H, W)
    new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
    scaled = F.resize(image, new_size)
    # Resize back to original dimensions
    return F.resize(scaled, original_size)

def gamma_transform(image: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply gamma correction to image."""
    return F.adjust_gamma(image, gamma)

# --- Reverse Coordinate Transformations ---

def identity_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int) -> Dict[str, torch.Tensor]:
    """Identity reverse transformation."""
    return pred_dict.copy()

def hflip_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int) -> Dict[str, torch.Tensor]:
    """Reverse horizontal flip for coordinates."""
    result = pred_dict.copy()
    boxes = pred_dict['boxes'].clone()
    ellipse_params = pred_dict['ellipse_params'].clone()

    if ellipse_params.numel() > 0:
        # Flip bounding boxes: [x1, y1, x2, y2] -> [W-x2, y1, W-x1, y2]
        boxes[:, [0, 2]] = original_W - boxes[:, [2, 0]]
        
        # Flip ellipse center x-coordinate and mirror angle
        ellipse_params[:, 2] = original_W - ellipse_params[:, 2]  # cx
        ellipse_params[:, 4] = -ellipse_params[:, 4]  # theta (mirror angle)

    result['boxes'] = boxes
    result['ellipse_params'] = ellipse_params
    return result

def rotate_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int, angle: float) -> Dict[str, torch.Tensor]:
    """Reverse rotation transformation for coordinates."""
    result = pred_dict.copy()
    boxes = pred_dict['boxes'].clone()
    ellipse_params = pred_dict['ellipse_params'].clone()
    
    if ellipse_params.numel() == 0:
        return result
    
    # Get device from input tensors
    device = boxes.device if boxes.numel() > 0 else ellipse_params.device
    
    angle_rad = math.radians(angle)
    center_x = original_W // 2
    center_y = original_H // 2

    # --- Rotate Bounding Boxes ---
    rotated_boxes = []
    if boxes.numel() > 0:
        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[i].tolist()
            corners = torch.tensor([
                [x1, y1], [x2, y1], [x1, y2], [x2, y2]
            ], dtype=torch.float32, device=device)

            corners_translated = corners - torch.tensor([center_x, center_y], device=device)
            
            # Clockwise rotation matrix (matches torchvision)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            rot_matrix = torch.tensor([
                [cos_a, sin_a],
                [-sin_a, cos_a]
            ], dtype=torch.float32, device=device)
            
            rotated_corners = torch.matmul(corners_translated, rot_matrix.T)
            rotated_corners_translated_back = rotated_corners + torch.tensor([center_x, center_y], device=device)
            
            new_x1 = torch.min(rotated_corners_translated_back[:, 0])
            new_y1 = torch.min(rotated_corners_translated_back[:, 1])
            new_x2 = torch.max(rotated_corners_translated_back[:, 0])
            new_y2 = torch.max(rotated_corners_translated_back[:, 1])
            rotated_boxes.append(torch.tensor([new_x1, new_y1, new_x2, new_y2], device=device))
    
        if rotated_boxes:
            boxes = torch.stack(rotated_boxes)

    # --- Rotate Ellipse Centers ---
    if ellipse_params.numel() > 0:
        x_coords = ellipse_params[:, 2]
        y_coords = ellipse_params[:, 3]
        
        # Translate to origin
        x_translated = x_coords - center_x
        y_translated = y_coords - center_y
        
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Apply clockwise rotation (matches torchvision convention)
        rotated_x = x_translated * cos_a + y_translated * sin_a
        rotated_y = -x_translated * sin_a + y_translated * cos_a
        
        # Translate back
        ellipse_params[:, 2] = rotated_x + center_x
        ellipse_params[:, 3] = rotated_y + center_y
        
        # Update angle - subtract for proper reversal
        ellipse_params[:, 4] = ellipse_params[:, 4] - angle_rad

    result['boxes'] = boxes
    result['ellipse_params'] = ellipse_params
    return result

def photometric_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int) -> Dict[str, torch.Tensor]:
    """Photometric transformations don't affect coordinates."""
    return pred_dict.copy()

def scale_reverse(pred_dict: Dict[str, torch.Tensor], original_H: int, original_W: int, scale_factor: float) -> Dict[str, torch.Tensor]:
    """Reverse scale transformation for coordinates.
    
    Since we scale the image and then resize back to original dimensions,
    the coordinates need to be adjusted by the inverse scale factor.
    """
    result = pred_dict.copy()
    boxes = pred_dict['boxes'].clone()
    ellipse_params = pred_dict['ellipse_params'].clone()
    
    if ellipse_params.numel() == 0:
        return result
    
    # The scaling was applied and then resized back, so coordinates are effectively
    # scaled by the inverse factor due to the resize operation
    inverse_scale = 1.0 / scale_factor
    
    # Scale bounding boxes
    if boxes.numel() > 0:
        boxes = boxes * inverse_scale
    
    # Scale ellipse parameters
    if ellipse_params.numel() > 0:
        # Scale semi-axes and centers
        ellipse_params[:, 0] = ellipse_params[:, 0] * inverse_scale  # a (semi-major axis)
        ellipse_params[:, 1] = ellipse_params[:, 1] * inverse_scale  # b (semi-minor axis)
        ellipse_params[:, 2] = ellipse_params[:, 2] * inverse_scale  # cx (center x)
        ellipse_params[:, 3] = ellipse_params[:, 3] * inverse_scale  # cy (center y)
        # theta (angle) remains unchanged
    
    result['boxes'] = boxes
    result['ellipse_params'] = ellipse_params
    return result

# --- TTA Transform Definitions ---

TTA_TRANSFORMS = [
    {
        'name': 'Original',
        'forward': (identity_transform, {}),
        'reverse': (identity_reverse, {}),
        'color': '#FF0000'  # Red
    },
    {
        'name': 'Horizontal Flip',
        'forward': (hflip_transform, {}),
        'reverse': (hflip_reverse, {}),
        'color': '#00FF00'  # Green
    },
    {
        'name': f'Rotation -{TTA_CONFIG["rotation_angle"]}°',
        'forward': (rotate_transform, {'angle': -TTA_CONFIG['rotation_angle']}),
        'reverse': (rotate_reverse, {'angle': TTA_CONFIG['rotation_angle']}),
        'color': '#0000FF'  # Blue
    },
    {
        'name': f'Rotation +{TTA_CONFIG["rotation_angle"]}°',
        'forward': (rotate_transform, {'angle': TTA_CONFIG['rotation_angle']}),
        'reverse': (rotate_reverse, {'angle': -TTA_CONFIG['rotation_angle']}),
        'color': '#FFFF00'  # Yellow
    },
    {
        'name': 'Brightness/Contrast',
        'forward': (brightness_contrast_transform, {}),
        'reverse': (photometric_reverse, {}),
        'color': '#FF00FF'  # Magenta
    },
    # Multi-Scale Testing
    {
        'name': f'Scale {TTA_CONFIG["scale_factors"][0]}x',
        'forward': (scale_transform, {'scale_factor': TTA_CONFIG["scale_factors"][0]}),
        'reverse': (scale_reverse, {'scale_factor': TTA_CONFIG["scale_factors"][0]}),
        'color': '#00FFFF'  # Cyan
    },
    {
        'name': f'Scale {TTA_CONFIG["scale_factors"][1]}x',
        'forward': (scale_transform, {'scale_factor': TTA_CONFIG["scale_factors"][1]}),
        'reverse': (scale_reverse, {'scale_factor': TTA_CONFIG["scale_factors"][1]}),
        'color': '#FFA500'  # Orange
    },
    {
        'name': f'Scale {TTA_CONFIG["scale_factors"][2]}x',
        'forward': (scale_transform, {'scale_factor': TTA_CONFIG["scale_factors"][2]}),
        'reverse': (scale_reverse, {'scale_factor': TTA_CONFIG["scale_factors"][2]}),
        'color': '#800080'  # Purple
    },
    # Additional Rotations - Positive angles
    {
        'name': f'Rotation +{TTA_CONFIG["additional_rotations"][0]}°',
        'forward': (rotate_transform, {'angle': TTA_CONFIG["additional_rotations"][0]}),
        'reverse': (rotate_reverse, {'angle': -TTA_CONFIG["additional_rotations"][0]}),
        'color': '#008000'  # Dark Green
    },
    {
        'name': f'Rotation +{TTA_CONFIG["additional_rotations"][1]}°',
        'forward': (rotate_transform, {'angle': TTA_CONFIG["additional_rotations"][1]}),
        'reverse': (rotate_reverse, {'angle': -TTA_CONFIG["additional_rotations"][1]}),
        'color': '#000080'  # Navy
    },
    {
        'name': f'Rotation +{TTA_CONFIG["additional_rotations"][2]}°',
        'forward': (rotate_transform, {'angle': TTA_CONFIG["additional_rotations"][2]}),
        'reverse': (rotate_reverse, {'angle': -TTA_CONFIG["additional_rotations"][2]}),
        'color': '#800000'  # Maroon
    },
    # Additional Rotations - Negative angles
    {
        'name': f'Rotation -{TTA_CONFIG["additional_rotations"][0]}°',
        'forward': (rotate_transform, {'angle': -TTA_CONFIG["additional_rotations"][0]}),
        'reverse': (rotate_reverse, {'angle': TTA_CONFIG["additional_rotations"][0]}),
        'color': '#808000'  # Olive
    },
    {
        'name': f'Rotation -{TTA_CONFIG["additional_rotations"][1]}°',
        'forward': (rotate_transform, {'angle': -TTA_CONFIG["additional_rotations"][1]}),
        'reverse': (rotate_reverse, {'angle': TTA_CONFIG["additional_rotations"][1]}),
        'color': '#008080'  # Teal
    },
    {
        'name': f'Rotation -{TTA_CONFIG["additional_rotations"][2]}°',
        'forward': (rotate_transform, {'angle': -TTA_CONFIG["additional_rotations"][2]}),
        'reverse': (rotate_reverse, {'angle': TTA_CONFIG["additional_rotations"][2]}),
        'color': '#808080'  # Gray
    },
    # Gamma Corrections
    {
        'name': f'Gamma {TTA_CONFIG["gamma_values"][0]}',
        'forward': (gamma_transform, {'gamma': TTA_CONFIG["gamma_values"][0]}),
        'reverse': (photometric_reverse, {}),
        'color': '#FFB6C1'  # Light Pink
    },
    {
        'name': f'Gamma {TTA_CONFIG["gamma_values"][1]}',
        'forward': (gamma_transform, {'gamma': TTA_CONFIG["gamma_values"][1]}),
        'reverse': (photometric_reverse, {}),
        'color': '#98FB98'  # Pale Green
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
        return 1.0  # Perfect reference
    elif 'horizontal flip' in name:
        return 0.95  # Geometric transform, very reliable
    elif 'brightness' in name or 'contrast' in name:
        return 0.85  # Photometric, doesn't affect geometry
    elif 'gamma' in name:
        return 0.8  # Photometric, slight quality impact
    
    # Medium quality transforms (some degradation expected)
    elif 'rotation' in name:
        # Extract angle if possible for fine-grained scoring
        if '10°' in name or '10 ' in name:
            return 0.9   # Small rotations are quite good
        elif '15°' in name or '15 ' in name:
            return 0.7   # Medium rotations have some errors
        elif '45°' in name or '45 ' in name:
            return 0.4   # Large rotations often problematic
        elif '90°' in name or '90 ' in name:
            return 0.3   # Very large rotations very problematic
        else:
            return 0.6   # Unknown rotation angle
    
    elif 'scale' in name:
        # Extract scale factor for fine-grained scoring
        if '0.8' in name:
            return 0.75  # Downscaling generally OK
        elif '1.2' in name:
            return 0.75  # Moderate upscaling OK
        elif '1.5' in name:
            return 0.5   # Large upscaling often problematic
        else:
            return 0.6   # Unknown scale factor
    
    # Default for unknown transforms
    else:
        return 0.5  # Neutral quality for unknown transforms

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
            
        # Filter by confidence threshold
        score_mask = pred_dict["scores"] > min_score
        if not score_mask.any():
            continue
        
        # Get quality score for this transform
        transform_quality = get_transform_quality(transform_name)
        
        # Extract valid predictions
        ellipse_params = pred_dict["ellipse_params"][score_mask]
        scores = pred_dict["scores"][score_mask]
        labels = pred_dict["labels"][score_mask]
        boxes = pred_dict["boxes"][score_mask]
        
        # Ensure all tensors are on the same device
        ellipse_params = ellipse_params.to(device)
        scores = scores.to(device)
        labels = labels.to(device)
        boxes = boxes.to(device)
        
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
    for group in consensus_groups:
        if group and 'ellipse' in group[0]:
            device = group[0]['ellipse'].device
            break
    
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
        consensus_ellipse = _weighted_ellipse_consensus(ellipses, combined_weights)
        
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

def _weighted_ellipse_consensus(ellipses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
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

def tta_predict_with_details(model: Any, image_tensor: torch.Tensor, device: torch.device,
                           min_score: float = None, consensuate: bool = True, 
                           visualize: bool = False) -> tuple:
    """
    Enhanced TTA prediction that returns per-transformation details.
    
    Args:
        model: EllipseRCNN model
        image_tensor: Input image tensor (C, H, W)
        device: Device to run inference on
        min_score: Minimum confidence score threshold
        consensuate: Whether to consensuate predictions
        visualize: Whether to show visualization
    
    Returns:
        Tuple of (final_predictions, per_transform_details)
    """
    if min_score is None:
        min_score = TTA_CONFIG['min_score_threshold']
    
    model.eval()
    model.to(device)
    
    original_H, original_W = image_tensor.shape[1:]
    all_predictions = []
    all_transform_names = []
    per_transform_details = []
    
    # Apply each TTA transformation
    for transform_config in TTA_TRANSFORMS:
        try:
            transform_name = transform_config['name']
            forward_func, forward_kwargs = transform_config['forward']
            reverse_func, reverse_kwargs = transform_config['reverse']
            
            # Apply forward transformation
            augmented_img = forward_func(image_tensor, **forward_kwargs)
            
            # Run inference (ensure image is on correct device)
            with torch.no_grad():
                augmented_img_gpu = augmented_img.to(device) if augmented_img.device != device else augmented_img
                predictions = model([augmented_img_gpu])
            
            pred_dict = predictions[0]
            
            # Defensive check for prediction dict structure
            if not isinstance(pred_dict, dict):
                print(f"⚠️  Warning: prediction is not a dict for transform {transform_name}")
                continue
            
            # Check required keys exist
            required_keys = ["ellipse_params", "scores"]
            if not all(key in pred_dict for key in required_keys):
                print(f"⚠️  Warning: missing keys in prediction for transform {transform_name}")
                continue
        except Exception as transform_error:
            print(f"⚠️ Transform {transform_config['name']} failed: {transform_error}")
            continue
        
        # Store raw prediction stats before filtering
        ellipse_params = pred_dict["ellipse_params"]
        scores = pred_dict["scores"]
        
        # Check tensor validity
        if ellipse_params.numel() == 0 or scores.numel() == 0:
            raw_detections = 0
            raw_scores = []
        else:
            raw_detections = len(ellipse_params)
            raw_scores = scores.cpu().numpy()
        
        # Apply reverse transformation
        if ellipse_params.numel() > 0:
            reversed_pred = reverse_func(pred_dict, original_H, original_W, **reverse_kwargs)
            
            # Ensure all tensors in reversed_pred are on the correct device
            reversed_pred = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in reversed_pred.items()}
            
            # Filter by score threshold
            score_mask = reversed_pred["scores"] > min_score
            filtered_detections = score_mask.sum().item()
            filtered_scores = reversed_pred["scores"][score_mask].cpu().numpy()
            
            if score_mask.any():
                filtered_pred = {
                    "ellipse_params": reversed_pred["ellipse_params"][score_mask],
                    "scores": reversed_pred["scores"][score_mask],
                    "labels": reversed_pred["labels"][score_mask],
                    "boxes": reversed_pred["boxes"][score_mask]
                }
                all_predictions.append(filtered_pred)
                all_transform_names.append(transform_name)
            else:
                filtered_pred = None
        else:
            reversed_pred = None
            filtered_pred = None
            filtered_detections = 0
            filtered_scores = []
        
        # Store detailed stats for this transformation
        transform_details = {
            'name': transform_config['name'],
            'raw_detections': raw_detections,
            'filtered_detections': filtered_detections,
            'raw_avg_confidence': np.mean(raw_scores) if len(raw_scores) > 0 else 0.0,
            'filtered_avg_confidence': np.mean(filtered_scores) if len(filtered_scores) > 0 else 0.0,
            'raw_max_confidence': np.max(raw_scores) if len(raw_scores) > 0 else 0.0,
            'filtered_max_confidence': np.max(filtered_scores) if len(filtered_scores) > 0 else 0.0,
            'contribution': 1 if filtered_pred is not None else 0,
            'predictions': filtered_pred  # Store actual predictions for error calculation
        }
        per_transform_details.append(transform_details)
    
    # Return results based on mode
    # COMMENTED OUT: Consensuation for individual transform testing
    # if consensuate and all_predictions:
    #     # Consensuated mode: return one combined prediction
    #     try:
    #         final_predictions = consensuate_predictions(all_predictions, all_transform_names, min_score)
    #     except Exception as consensus_error:
    #         print(f"⚠️ Consensuation failed: {consensus_error}")
    #         print(f"   Falling back to original prediction")
    #         # Fall back to first prediction if consensuation fails
    #         final_predictions = all_predictions[0] if all_predictions else _empty_prediction_dict(device)
    #     return [final_predictions], per_transform_details
    if all_predictions:
        # Individual mode: return all individual predictions separately
        return all_predictions, per_transform_details
    else:
        # No predictions found
        empty_pred = _empty_prediction_dict(device)
        return [empty_pred], per_transform_details

def tta_predict(model: Any, image_tensor: torch.Tensor, device: torch.device, 
                min_score: float = None, consensuate: bool = True, 
                visualize: bool = False) -> List[Dict[str, torch.Tensor]]:
    """
    Perform Test Time Augmentation prediction on an image.
    
    Args:
        model: The ellipse detection model
        image_tensor: Input image tensor [C, H, W]
        device: Device to run inference on
        min_score: Minimum confidence threshold
        consensuate: Whether to consensuate predictions
        visualize: Whether to show visualization
    
    Returns:
        List containing prediction dictionary
    """
    if min_score is None:
        min_score = TTA_CONFIG['min_score_threshold']
    
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
        
        # Apply forward transformation
        augmented_img = forward_func(image_tensor, **forward_kwargs)
        
        # Run inference (ensure image is on correct device)
        with torch.no_grad():
            augmented_img_gpu = augmented_img.to(device) if augmented_img.device != device else augmented_img
            predictions = model([augmented_img_gpu])
        
        pred_dict = predictions[0]
        
        # Skip if no predictions
        if pred_dict["ellipse_params"].numel() == 0:
            continue
        
        # Apply reverse transformation
        reversed_pred = reverse_func(pred_dict, original_H, original_W, **reverse_kwargs)
        all_predictions.append(reversed_pred)
        all_transform_names.append(transform_config['name'])
        
        # Add to visualization if requested
        if visualize:
            score_mask = reversed_pred["scores"] > min_score
            if score_mask.any():
                _plot_ellipses(reversed_pred["ellipse_params"][score_mask], 
                             ax_combined, transform_config['color'])
                legend_elements.append(plt.Line2D([0], [0], color=transform_config['color'], 
                                                lw=4, label=transform_config['name']))
    
    # Return results based on mode  
    # COMMENTED OUT: Consensuation for individual transform testing
    # if consensuate and all_predictions:
    #     # Consensuated mode: return one combined prediction
    #     final_predictions = consensuate_predictions(all_predictions, all_transform_names, min_score)
    #     return [final_predictions]
    if all_predictions:
        # Individual mode: return all individual predictions
        return all_predictions
    else:
        # No predictions found
        empty_pred = _empty_prediction_dict(device)
        return [empty_pred]


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
