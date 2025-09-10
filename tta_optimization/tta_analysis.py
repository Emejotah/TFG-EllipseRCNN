#!/usr/bin/env python3
"""
TTA Analysis for Ellipse R-CNN on FDDB Dataset

This module performs a comprehensive Test Time Augmentation analysis to evaluate
which transformations improve the model performance on the FDDB face detection dataset.

The analysis includes:
1. Dataset partitioning (10% subset for analysis)
2. Base model evaluation
3. TTA evaluation with multiple transformations
4. Comprehensive metrics collection and analysis
5. Detailed error analysis for center, angle, and consensuation quality
"""

import sys
import json
import math
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from datetime import datetime

# Add parent directory to import ellipse_rcnn modules
sys.path.insert(0, '..')

from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.core.ops import ellipse_area
from tta_transforms import tta_predict_with_details, TTA_TRANSFORMS
from fddb_partitioner import FDDBSimplePartitioner


class TTAAnalyzer:
    """
    Comprehensive TTA Analysis for Ellipse R-CNN
    
    Evaluates model performance with and without TTA, collecting detailed metrics
    for quantitative analysis of transformation effectiveness.
    """
    
    def __init__(self, model_repo: str, data_root: str = "../data/FDDB", device: str = "cpu"):
        """
        Initialize the TTA analyzer.
        
        Args:
            model_repo: Hugging Face model repository (e.g., "MJGT/ellipse-rcnn-FDDB")
            data_root: Path to FDDB dataset root
            device: Device to run inference on
        """
        self.model_repo = model_repo
        self.data_root = Path(data_root)
        self.device = torch.device(device)
        
        print(f"🔬 TTA Analyzer initialized")
        print(f"🤗 Model repo: {self.model_repo}")
        print(f"📁 Data root: {self.data_root}")
        print(f"💻 Device: {self.device}")
        
        # Load model
        self.model = self._load_model()
        
        # Initialize metrics storage
        self.results = {
            'metadata': {
                'model_repo': self.model_repo,
                'data_root': str(self.data_root),
                'device': str(self.device),
                'transforms_evaluated': [t['name'] for t in TTA_TRANSFORMS],
                'max_images_limit': None  # Will be set during analysis
            },
            'base_model_results': {},
            'tta_results': {},
            'per_transform_analysis': {},
            'error_analysis': {},
            'summary_metrics': {}
        }
    
    def _load_model(self) -> EllipseRCNN:
        """Load the pre-trained EllipseRCNN model from Hugging Face."""
        print(f"\n📥 Loading model from Hugging Face: {self.model_repo}")
        
        try:
            model = EllipseRCNN.from_pretrained(self.model_repo)
            model.eval()
            model.to(self.device)
            
            print(f"✅ Model loaded successfully from Hugging Face")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load model from Hugging Face: {e}")
            print(f"💡 Make sure the model repository '{self.model_repo}' exists and is accessible")
            raise
    
    def run_full_analysis(self, optimization_fraction: float = 0.1, min_score: float = 0.5, max_images: int = None):
        """
        Run the complete TTA analysis pipeline.
        
        Args:
            optimization_fraction: Fraction of dataset to use for analysis
            min_score: Minimum confidence score for predictions
            max_images: Maximum number of images to analyze (None for no limit)
        """
        print(f"\n" + "="*60)
        print(f"🚀 STARTING COMPREHENSIVE TTA ANALYSIS")
        print(f"="*60)
        
        # Step 1: Calculate target dataset size
        print(f"\n📊 Step 1: Calculating Target Dataset Size")
        target_count = self._calculate_target_count(optimization_fraction, max_images)
        
        # Step 2: Load dataset subset
        print(f"\n📂 Step 2: Loading Dataset Subset")
        dataset = self._load_dataset_subset(target_count)
        
        # Update metadata with actual parameters
        self.results['metadata']['max_images_limit'] = max_images
        self.results['metadata']['actual_images_analyzed'] = len(dataset)
        self.results['metadata']['optimization_fraction'] = optimization_fraction
        self.results['metadata']['min_score'] = min_score
        
        # Step 3: Base model evaluation
        print(f"\n🔍 Step 3: Base Model Evaluation")
        base_results = self._evaluate_base_model(dataset, min_score)
        self.results['base_model_results'] = base_results
        
        # Step 4: TTA evaluation
        print(f"\n🔄 Step 4: TTA Evaluation")
        tta_results = self._evaluate_tta_model(dataset, min_score)
        self.results['tta_results'] = tta_results
        
        # Step 5: Per-transform analysis
        print(f"\n📈 Step 5: Per-Transform Analysis")
        transform_analysis = self._analyze_per_transform_performance(dataset, min_score)
        self.results['per_transform_analysis'] = transform_analysis
        
        # Step 6: Error analysis
        print(f"\n📏 Step 6: Error Analysis")
        error_analysis = self._perform_error_analysis(dataset, min_score)
        self.results['error_analysis'] = error_analysis
        
        # Step 7: Summary metrics
        print(f"\n📋 Step 7: Computing Summary Metrics")
        summary = self._compute_summary_metrics()
        self.results['summary_metrics'] = summary
        
        # Step 8: Save results
        print(f"\n💾 Step 8: Saving Results")
        self._save_results()
        
        print(f"\n" + "="*60)
        print(f"✅ TTA ANALYSIS COMPLETED SUCCESSFULLY")
        print(f"="*60)
        
        return self.results
    
    def _calculate_target_count(self, optimization_fraction: float, max_images: int = None) -> int:
        """Calculate the target number of images for analysis."""
        # Load full dataset to get total count
        try:
            full_dataset = FDDB(str(self.data_root), download=False)
            total_images = len(full_dataset.ellipse_dict)
            
            # Calculate target based on fraction
            target_from_fraction = int(total_images * optimization_fraction)
            
            # Apply max_images limit if specified
            if max_images is not None and max_images > 0:
                target_count = min(target_from_fraction, max_images)
                print(f"🔄 Target count limited from {target_from_fraction} to {target_count} by max_images")
            else:
                target_count = target_from_fraction
            
            print(f"📊 Dataset statistics:")
            print(f"   Total images in FDDB: {total_images}")
            print(f"   Optimization fraction: {optimization_fraction:.1%}")
            print(f"   Target images for analysis: {target_count}")
            
            return target_count
            
        except Exception as e:
            print(f"⚠️ Error calculating target count: {e}")
            # Return a safe default
            return max_images if max_images and max_images > 0 else 100
    
    def _load_dataset_subset(self, target_count: int) -> FDDB:
        """Load FDDB dataset subset with a fixed number of images."""
        # Create full dataset
        full_dataset = FDDB(str(self.data_root), download=False)
        
        # Get all available keys
        all_keys = list(full_dataset.ellipse_dict.keys())
        print(f"🔍 Total images available in FDDB: {len(all_keys)}")
        
        # Take the first N images to ensure we always get the requested count
        if target_count > len(all_keys):
            print(f"⚠️ Requested {target_count} images but only {len(all_keys)} available")
            target_count = len(all_keys)
        
        selected_keys = all_keys[:target_count]
        print(f"📂 Selected {len(selected_keys)} images for analysis")
        
        # Create ellipse_dict for selected images
        selected_ellipse_dict = {k: full_dataset.ellipse_dict[k] for k in selected_keys}
        
        # Create subset dataset
        subset_dataset = FDDB(str(self.data_root), ellipse_dict=selected_ellipse_dict)
        return subset_dataset
    
    def _evaluate_base_model(self, dataset: FDDB, min_score: float) -> Dict[str, Any]:
        """Evaluate base model performance without TTA."""
        print(f"🔍 Evaluating base model on {len(dataset)} images...")
        
        base_results = {
            'predictions': [],
            'targets': [],
            'metrics': {}
        }
        
        total_detections = 0
        total_targets = 0
        valid_images = 0
        
        for idx in tqdm(range(len(dataset)), desc="Base Model Evaluation"):
            try:
                image_tensor, target = dataset[idx]
                
                # Move target tensors to the same device as the model
                if isinstance(target, dict):
                    target = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in target.items()}
                
                # Run base model prediction
                with torch.no_grad():
                    image_batch = image_tensor.unsqueeze(0).to(self.device)
                    predictions = self.model(image_batch)
                
                pred_dict = predictions[0]
                
                # Filter by score
                if pred_dict["ellipse_params"].numel() > 0:
                    score_mask = pred_dict["scores"] > min_score
                    filtered_predictions = {
                        'ellipse_params': pred_dict["ellipse_params"][score_mask],
                        'scores': pred_dict["scores"][score_mask],
                        'boxes': pred_dict["boxes"][score_mask],
                        'labels': pred_dict["labels"][score_mask] if "labels" in pred_dict else torch.ones(score_mask.sum(), dtype=torch.int64)
                    }
                else:
                    filtered_predictions = {
                        'ellipse_params': torch.empty((0, 5), device=self.device),
                        'scores': torch.empty((0,), device=self.device),
                        'boxes': torch.empty((0, 4), device=self.device),
                        'labels': torch.empty((0,), dtype=torch.int64, device=self.device)
                    }
                
                base_results['predictions'].append(filtered_predictions)
                base_results['targets'].append(target)
                
                total_detections += len(filtered_predictions['ellipse_params'])
                total_targets += len(target['ellipse_params'])
                valid_images += 1
                
            except Exception as e:
                print(f"⚠️ Error processing image {idx}: {e}")
                continue
        
        # Compute base metrics
        base_results['metrics'] = {
            'total_images': valid_images,
            'total_detections': total_detections,
            'total_targets': total_targets,
            'avg_detections_per_image': total_detections / valid_images if valid_images > 0 else 0,
            'avg_targets_per_image': total_targets / valid_images if valid_images > 0 else 0
        }
        
        print(f"📊 Base model results:")
        print(f"   Images processed: {valid_images}")
        print(f"   Total detections: {total_detections}")
        print(f"   Total targets: {total_targets}")
        print(f"   Avg detections/image: {base_results['metrics']['avg_detections_per_image']:.2f}")
        
        return base_results
    
    def _evaluate_tta_model(self, dataset: FDDB, min_score: float) -> Dict[str, Any]:
        """Evaluate TTA model performance with consensuation."""
        print(f"🔄 Evaluating TTA model on {len(dataset)} images...")
        
        tta_results = {
            'predictions': [],
            'targets': [],
            'per_transform_details': [],
            'metrics': {}
        }
        
        total_detections = 0
        total_targets = 0
        valid_images = 0
        
        for idx in tqdm(range(len(dataset)), desc="TTA Model Evaluation"):
            try:
                image_tensor, target = dataset[idx]
                
                # Move target tensors to the same device as the model
                if isinstance(target, dict):
                    target = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in target.items()}
                
                # Run TTA prediction with detailed analysis
                try:
                    tta_preds, transform_details = tta_predict_with_details(
                        model=self.model,
                        image_tensor=image_tensor,
                        device=self.device,
                        min_score=min_score,
                        consensuate=True,  # MODIFIED: Enable consensuation with validation
                        visualize=False
                    )
                except Exception as tta_error:
                    print(f"⚠️ TTA prediction failed for image {idx}: {tta_error}")
                    # Create empty results as fallback
                    tta_preds = [{
                        'ellipse_params': torch.empty((0, 5), device=self.device),
                        'scores': torch.empty((0,), device=self.device),
                        'boxes': torch.empty((0, 4), device=self.device),
                        'labels': torch.empty((0,), dtype=torch.int64, device=self.device)
                    }]
                    transform_details = []
                
                # Use consensuated prediction for comparison (with validation)
                pred_dict = tta_preds[0] if tta_preds else {
                    'ellipse_params': torch.empty((0, 5), device=self.device),
                    'scores': torch.empty((0,), device=self.device),
                    'boxes': torch.empty((0, 4), device=self.device),
                    'labels': torch.empty((0,), dtype=torch.int64, device=self.device)
                }
                
                # Ensure pred_dict tensors are on correct device
                pred_dict = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in pred_dict.items()}
                
                # Ensure transform_details predictions are also on correct device
                for detail in transform_details:
                    if detail.get('predictions') is not None:
                        detail['predictions'] = {
                            k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in detail['predictions'].items()
                        }
                
                tta_results['predictions'].append(pred_dict)
                tta_results['targets'].append(target)
                tta_results['per_transform_details'].append(transform_details)
                
                total_detections += len(pred_dict['ellipse_params'])
                total_targets += len(target['ellipse_params'])
                valid_images += 1
                
            except Exception as e:
                print(f"⚠️ Error processing image {idx}: {e}")
                continue
        
        # Compute TTA metrics
        tta_results['metrics'] = {
            'total_images': valid_images,
            'total_detections': total_detections,
            'total_targets': total_targets,
            'avg_detections_per_image': total_detections / valid_images if valid_images > 0 else 0,
            'avg_targets_per_image': total_targets / valid_images if valid_images > 0 else 0
        }
        
        print(f"📊 TTA model results:")
        print(f"   Images processed: {valid_images}")
        print(f"   Total detections: {total_detections}")
        print(f"   Total targets: {total_targets}")
        print(f"   Avg detections/image: {tta_results['metrics']['avg_detections_per_image']:.2f}")
        
        return tta_results
    
    def _analyze_per_transform_performance(self, dataset: FDDB, min_score: float) -> Dict[str, Any]:
        """Analyze performance of individual transformations."""
        print(f"📈 Analyzing per-transform performance...")
        
        transform_stats = defaultdict(lambda: {
            'total_raw_detections': 0,
            'total_filtered_detections': 0,
            'total_contributions': 0,
            'avg_raw_confidence': 0,
            'avg_filtered_confidence': 0,
            'images_with_detections': 0,
            'images_processed': 0
        })
        
        # Aggregate statistics from TTA results
        if 'per_transform_details' in self.results.get('tta_results', {}):
            for image_details in self.results['tta_results']['per_transform_details']:
                for transform_detail in image_details:
                    name = transform_detail['name']
                    stats = transform_stats[name]
                    
                    stats['total_raw_detections'] += transform_detail['raw_detections']
                    stats['total_filtered_detections'] += transform_detail['filtered_detections']
                    stats['total_contributions'] += transform_detail['contribution']
                    stats['avg_raw_confidence'] += transform_detail['raw_avg_confidence']
                    stats['avg_filtered_confidence'] += transform_detail['filtered_avg_confidence']
                    stats['images_processed'] += 1
                    
                    if transform_detail['filtered_detections'] > 0:
                        stats['images_with_detections'] += 1
        
        # Calculate averages and final statistics
        final_transform_analysis = {}
        for name, stats in transform_stats.items():
            if stats['images_processed'] > 0:
                final_transform_analysis[name] = {
                    'total_raw_detections': stats['total_raw_detections'],
                    'total_filtered_detections': stats['total_filtered_detections'],
                    'total_contributions': stats['total_contributions'],
                    'avg_raw_confidence': stats['avg_raw_confidence'] / stats['images_processed'],
                    'avg_filtered_confidence': stats['avg_filtered_confidence'] / stats['images_processed'],
                    'images_with_detections': stats['images_with_detections'],
                    'images_processed': stats['images_processed'],
                    'detection_rate': stats['images_with_detections'] / stats['images_processed'],
                    'contribution_rate': stats['total_contributions'] / stats['images_processed']
                }
        
        print(f"📈 Per-transform analysis completed for {len(final_transform_analysis)} transforms")
        for name, analysis in final_transform_analysis.items():
            print(f"   {name}: {analysis['total_filtered_detections']} detections, "
                  f"{analysis['detection_rate']:.1%} detection rate, "
                  f"{analysis['contribution_rate']:.1%} contribution rate")
        
        return final_transform_analysis
    
    def _perform_error_analysis(self, dataset: FDDB, min_score: float) -> Dict[str, Any]:
        """Perform detailed error analysis comparing predictions to ground truth."""
        print(f"📏 Performing error analysis...")
        
        base_results = self.results.get('base_model_results', {})
        tta_results = self.results.get('tta_results', {})
        
        if not base_results or not tta_results:
            print("⚠️ Missing base or TTA results for error analysis")
            return {}
        
        error_metrics = {
            'base_model_errors': {
                'center_errors': [],
                'angle_errors': [],
                'area_errors': []
            },
            'tta_model_errors': {
                'center_errors': [],
                'angle_errors': [],
                'area_errors': []
            },
            'per_transform_errors': defaultdict(lambda: {
                'center_errors': [],
                'angle_errors': [],
                'area_errors': []
            }),
            'improvement_analysis': {}
        }
        
        base_preds = base_results.get('predictions', [])
        tta_preds = tta_results.get('predictions', [])
        targets = base_results.get('targets', [])
        transform_details = tta_results.get('per_transform_details', [])
        
        for idx in range(min(len(base_preds), len(tta_preds), len(targets))):
            try:
                base_pred = base_preds[idx]
                tta_pred = tta_preds[idx]
                target = targets[idx]
                
                # Calculate errors for base model
                base_errors = self._calculate_prediction_errors(base_pred, target)
                for metric, values in base_errors.items():
                    error_metrics['base_model_errors'][metric].extend(values)
                
                # Calculate errors for TTA model
                tta_errors = self._calculate_prediction_errors(tta_pred, target)
                for metric, values in tta_errors.items():
                    error_metrics['tta_model_errors'][metric].extend(values)
                
                # Calculate errors for individual transforms
                if idx < len(transform_details):
                    for transform_detail in transform_details[idx]:
                        if transform_detail.get('predictions') is not None:
                            transform_errors = self._calculate_prediction_errors(
                                transform_detail['predictions'], target
                            )
                            transform_name = transform_detail['name']
                            for metric, values in transform_errors.items():
                                error_metrics['per_transform_errors'][transform_name][metric].extend(values)
                
            except Exception as e:
                print(f"⚠️ Error in error analysis for image {idx}: {e}")
                continue
        
        # Compute improvement analysis
        error_metrics['improvement_analysis'] = self._compute_improvement_analysis(
            error_metrics['base_model_errors'],
            error_metrics['tta_model_errors']
        )
        
        print(f"📏 Error analysis completed")
        if error_metrics['base_model_errors']['center_errors']:
            print(f"   Base model center error (avg): {np.mean(error_metrics['base_model_errors']['center_errors']):.2f} pixels")
        else:
            print(f"   Base model center error (avg): No valid predictions")
            
        if error_metrics['tta_model_errors']['center_errors']:
            print(f"   TTA model center error (avg): {np.mean(error_metrics['tta_model_errors']['center_errors']):.2f} pixels")
        else:
            print(f"   TTA model center error (avg): No valid predictions")
            
        if 'center_improvement' in error_metrics['improvement_analysis']:
            print(f"   Improvement: {error_metrics['improvement_analysis']['center_improvement']:.2f} pixels")
        else:
            print(f"   Improvement: Unable to calculate (insufficient data)")
        
        return error_metrics
    
    def _calculate_prediction_errors(self, predictions: Dict, target: Dict) -> Dict[str, List[float]]:
        """Calculate errors between predictions and ground truth."""
        errors = {
            'center_errors': [],
            'angle_errors': [],
            'area_errors': []
        }
        
        pred_ellipses = predictions.get('ellipse_params', torch.empty((0, 5)))
        target_ellipses = target.get('ellipse_params', torch.empty((0, 5)))
        
        if pred_ellipses.numel() == 0 or target_ellipses.numel() == 0:
            return errors
        
        # For each prediction, find closest ground truth
        for pred_ellipse in pred_ellipses:
            if target_ellipses.numel() > 0:
                # Find closest target by center distance
                pred_center = pred_ellipse[2:4]  # cx, cy
                target_centers = target_ellipses[:, 2:4]
                distances = torch.norm(target_centers - pred_center.unsqueeze(0), dim=1)
                closest_idx = torch.argmin(distances)
                closest_target = target_ellipses[closest_idx]
                
                # Calculate center error (pixels)
                center_error = torch.norm(pred_center - closest_target[2:4]).item()
                errors['center_errors'].append(center_error)
                
                # Calculate angle error (degrees) - corrected for [-π/2, π/2] range
                angle_diff = abs(pred_ellipse[4].item() - closest_target[4].item())
                angle_error = min(angle_diff, math.pi - angle_diff) * 180 / math.pi
                errors['angle_errors'].append(angle_error)
                
                # Calculate area error (relative)
                pred_area = ellipse_area(pred_ellipse.unsqueeze(0)).item()
                target_area = ellipse_area(closest_target.unsqueeze(0)).item()
                if target_area > 0:
                    area_error = abs(pred_area - target_area) / target_area
                    errors['area_errors'].append(area_error)
        
        return errors
    
    def _compute_improvement_analysis(self, base_errors: Dict, tta_errors: Dict) -> Dict[str, float]:
        """Compute improvement metrics comparing base and TTA models."""
        improvements = {}
        
        for metric in ['center_errors', 'angle_errors', 'area_errors']:
            base_vals = base_errors.get(metric, [])
            tta_vals = tta_errors.get(metric, [])
            
            if base_vals and tta_vals:
                base_avg = np.mean(base_vals)
                tta_avg = np.mean(tta_vals)
                improvement = base_avg - tta_avg
                improvement_pct = (improvement / base_avg * 100) if base_avg > 0 else 0
                
                improvements[f"{metric.split('_')[0]}_improvement"] = improvement
                improvements[f"{metric.split('_')[0]}_improvement_pct"] = improvement_pct
        
        return improvements
    
    def _compute_summary_metrics(self) -> Dict[str, Any]:
        """Compute high-level summary metrics for the analysis."""
        summary = {
            'dataset_size': 0,
            'base_vs_tta_comparison': {},
            'best_transforms': {},
            'worst_transforms': {},
            'overall_improvements': {}
        }
        
        # Basic metrics
        base_metrics = self.results.get('base_model_results', {}).get('metrics', {})
        tta_metrics = self.results.get('tta_results', {}).get('metrics', {})
        
        summary['dataset_size'] = base_metrics.get('total_images', 0)
        
        # Compare base vs TTA
        if base_metrics and tta_metrics:
            summary['base_vs_tta_comparison'] = {
                'base_avg_detections': base_metrics.get('avg_detections_per_image', 0),
                'tta_avg_detections': tta_metrics.get('avg_detections_per_image', 0),
                'detection_improvement': tta_metrics.get('avg_detections_per_image', 0) - base_metrics.get('avg_detections_per_image', 0)
            }
        
        # Identify best and worst transforms
        transform_analysis = self.results.get('per_transform_analysis', {})
        if transform_analysis:
            # Sort by contribution rate
            sorted_transforms = sorted(
                transform_analysis.items(),
                key=lambda x: x[1].get('contribution_rate', 0),
                reverse=True
            )
            
            if len(sorted_transforms) >= 3:
                summary['best_transforms'] = {
                    'top_3': [t[0] for t in sorted_transforms[:3]],
                    'top_3_rates': [t[1].get('contribution_rate', 0) for t in sorted_transforms[:3]]
                }
                summary['worst_transforms'] = {
                    'bottom_3': [t[0] for t in sorted_transforms[-3:]],
                    'bottom_3_rates': [t[1].get('contribution_rate', 0) for t in sorted_transforms[-3:]]
                }
        
        # Overall improvements
        error_analysis = self.results.get('error_analysis', {})
        if error_analysis.get('improvement_analysis'):
            summary['overall_improvements'] = error_analysis['improvement_analysis']
        
        print(f"📋 Summary metrics computed")
        print(f"   Dataset size: {summary['dataset_size']} images")
        print(f"   Detection improvement: {summary['base_vs_tta_comparison'].get('detection_improvement', 0):.2f}")
        
        return summary
    
    def _save_results(self):
        """Save analysis results to JSON file."""
        output_file = Path("tta_analysis_results.json")
        
        # Convert any torch tensors to lists for JSON serialization
        serializable_results = self._make_json_serializable(self.results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to {output_file}")
        
        # Also save a human-readable summary
        self._save_summary_report()
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _save_summary_report(self):
        """Save a human-readable summary report."""
        report_file = Path("tta_analysis_summary.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# TTA Analysis Summary Report\n\n")
            f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset info
            summary = self.results.get('summary_metrics', {})
            metadata = self.results.get('metadata', {})
            f.write(f"## Dataset Information\n")
            f.write(f"- **Images analyzed:** {summary.get('dataset_size', 0)}\n")
            f.write(f"- **Optimization fraction:** {metadata.get('optimization_fraction', 0):.1%}\n")
            f.write(f"- **Max images limit:** {metadata.get('max_images_limit', 'None')}\n")
            f.write(f"- **Min score threshold:** {metadata.get('min_score', 0)}\n")
            f.write(f"- **Model repo:** {metadata['model_repo']}\n")
            f.write(f"- **Device used:** {metadata['device']}\n\n")
            
            # Performance comparison
            comparison = summary.get('base_vs_tta_comparison', {})
            f.write(f"## Performance Comparison\n")
            f.write(f"- **Base model detections/image:** {comparison.get('base_avg_detections', 0):.2f}\n")
            f.write(f"- **TTA model detections/image:** {comparison.get('tta_avg_detections', 0):.2f}\n")
            f.write(f"- **Improvement:** {comparison.get('detection_improvement', 0):.2f} detections/image\n\n")
            
            # Best transforms
            best = summary.get('best_transforms', {})
            if 'top_3' in best:
                f.write(f"## Best Performing Transforms\n")
                for i, (name, rate) in enumerate(zip(best['top_3'], best['top_3_rates'])):
                    f.write(f"{i+1}. **{name}:** {rate:.1%} contribution rate\n")
                f.write("\n")
            
            # Error improvements
            improvements = summary.get('overall_improvements', {})
            if improvements:
                f.write(f"## Error Improvements\n")
                f.write(f"- **Center error improvement:** {improvements.get('center_improvement', 0):.2f} pixels\n")
                f.write(f"- **Angle error improvement:** {improvements.get('angle_improvement', 0):.2f}°\n")
                f.write(f"- **Area error improvement:** {improvements.get('area_improvement', 0):.2%}\n\n")
            
            # Transform list
            f.write(f"## Transforms Evaluated\n")
            for i, transform_name in enumerate(self.results['metadata']['transforms_evaluated'], 1):
                f.write(f"{i}. {transform_name}\n")
        
        print(f"📋 Summary report saved to {report_file}")


def main():
    """Main function to run TTA analysis."""
    print("🔬 ELLIPSE R-CNN TTA ANALYSIS")
    print("="*50)
    
    # Configuration
    MODEL_REPO = "MJGT/ellipse-rcnn-FDDB"  # Hugging Face model repository
    DATA_ROOT = "../data/FDDB"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect CUDA
    OPTIMIZATION_FRACTION = 0.1  # 10% of dataset
    MIN_SCORE = 0.5  # Minimum confidence threshold
    MAX_IMAGES = 10  # Limit for testing - set to None for no limit
    
    print(f"🤗 Model: {MODEL_REPO}")
    print(f"📁 Data: {DATA_ROOT}")
    print(f"💻 Device: {DEVICE}")
    print(f"📊 Dataset fraction: {OPTIMIZATION_FRACTION:.1%}")
    print(f"🎯 Min score: {MIN_SCORE}")
    print(f"🔢 Max images: {MAX_IMAGES if MAX_IMAGES else 'No limit'}")
    
    # Create analyzer
    analyzer = TTAAnalyzer(MODEL_REPO, DATA_ROOT, DEVICE)
    
    # Run analysis
    try:
        results = analyzer.run_full_analysis(OPTIMIZATION_FRACTION, MIN_SCORE, MAX_IMAGES)
        
        print(f"\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"📊 Results available in:")
        print(f"   - tta_analysis_results.json (detailed)")
        print(f"   - tta_analysis_summary.md (summary)")
        
    except Exception as e:
        print(f"\n❌ ANALYSIS FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
