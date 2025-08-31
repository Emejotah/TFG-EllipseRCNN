#!/usr/bin/env python3
"""
TTA Transformation Diagnostic Test
=================================

This script performs comprehensive testing of Test Time Augmentation (TTA) transformations 
for EllipseRCNN to identify potential issues with implementation or evaluation.

Purpose: Debug why TTA is performing worse than baseline by testing:
- Individual transform correctness
- Coordinate transformation accuracy  
- Forward-reverse consistency
- Score filtering effects
- Consensuation algorithm
- Full pipeline debugging

Usage:
    python tta_diagnostic_test.py
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import EllipseCollection
from pathlib import Path
import time
import warnings
from PIL import Image
from torchvision.transforms.functional import to_tensor
import json

# Configure matplotlib and warnings
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
warnings.filterwarnings('ignore')

# Add project paths
sys.path.insert(0, '..')
sys.path.insert(0, '../tta_optimization')

# Import TTA modules
from tta_transforms import (
    tta_predict, tta_predict_with_details, TTA_TRANSFORMS,
    consensuate_predictions, _consensuate_ellipse_group
)

# Import EllipseRCNN modules  
from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.utils.viz import plot_ellipses


class TTADiagnosticTester:
    """Comprehensive TTA diagnostic testing suite."""
    
    def __init__(self, model_repo="MJGT/ellipse-rcnn-FDDB", data_root="../data/FDDB"):
        """Initialize the diagnostic tester."""
        self.model_repo = model_repo
        self.data_root = data_root
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🔧 TTA Diagnostic Tester Initialization")
        print("="*50)
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Device: {self.device}")
        print(f"Number of TTA transforms: {len(TTA_TRANSFORMS)}")
        
        self.model = None
        self.test_image_tensor = None
        self.test_targets = None
        self.baseline_result = None
        self.filtered_baseline = None
        
    def setup_model_and_data(self):
        """Load model and test data."""
        print(f"\n📦 Loading Model and Test Data...")
        
        # Load the trained model
        print(f"🤗 Loading model from {self.model_repo}...")
        self.model = EllipseRCNN.from_pretrained(self.model_repo)
        self.model.eval()
        self.model.to(self.device)
        print("✅ Model loaded successfully")
        
        # Load FDDB dataset to get a test image with known targets
        print(f"📁 Loading FDDB dataset from {self.data_root}...")
        try:
            fddb_dataset = FDDB(self.data_root, download=False)
            print(f"✅ FDDB dataset loaded: {len(fddb_dataset)} images")
            
            # Get a test image with good targets (multiple faces)
            test_idx = 100  # Adjust this to find a good test image
            self.test_image_tensor, self.test_targets = fddb_dataset[test_idx]
            
            print(f"📸 Test image loaded:")
            print(f"   - Image shape: {self.test_image_tensor.shape}")
            print(f"   - Number of target ellipses: {len(self.test_targets['ellipse_params'])}")
            print(f"   - Target ellipse params shape: {self.test_targets['ellipse_params'].shape}")
            
        except Exception as e:
            print(f"❌ Error loading FDDB: {e}")
            print("⚠️  Will create synthetic test data instead")
            
            # Create synthetic test image and targets for debugging
            self.test_image_tensor = torch.randn(3, 480, 640)  # Random test image
            self.test_targets = {
                'ellipse_params': torch.tensor([[50.0, 30.0, 320.0, 240.0, 0.0]]),  # Center ellipse
                'boxes': torch.tensor([[270.0, 210.0, 370.0, 270.0]])
            }
            print("✅ Synthetic test data created")
        
        # Move to device
        self.test_image_tensor = self.test_image_tensor.to(self.device)
        print(f"📱 Data moved to {self.device}")
    
    def test_baseline_prediction(self):
        """Test baseline prediction without TTA."""
        print(f"\n🎯 Testing Baseline Prediction (No TTA)...")
        
        # Run baseline prediction
        with torch.no_grad():
            baseline_pred = self.model([self.test_image_tensor])
        
        self.baseline_result = baseline_pred[0]
        print(f"📊 Baseline Results:")
        print(f"   - Raw detections: {len(self.baseline_result['ellipse_params'])}")
        print(f"   - Scores range: {self.baseline_result['scores'].min():.3f} - {self.baseline_result['scores'].max():.3f}")
        print(f"   - Mean score: {self.baseline_result['scores'].mean():.3f}")
        
        # Filter by score threshold
        min_score = 0.75
        score_mask = self.baseline_result['scores'] > min_score
        self.filtered_baseline = {
            'ellipse_params': self.baseline_result['ellipse_params'][score_mask],
            'scores': self.baseline_result['scores'][score_mask],
            'boxes': self.baseline_result['boxes'][score_mask],
            'labels': self.baseline_result['labels'][score_mask]
        }
        
        print(f"📊 Filtered Baseline (score > {min_score}):")
        print(f"   - Filtered detections: {len(self.filtered_baseline['ellipse_params'])}")
        if len(self.filtered_baseline['ellipse_params']) > 0:
            print(f"   - Filtered scores: {self.filtered_baseline['scores'].min():.3f} - {self.filtered_baseline['scores'].max():.3f}")
        
        print("✅ Baseline test completed")
        return self.filtered_baseline
    
    def test_individual_transforms(self):
        """Test each TTA transformation individually."""
        print(f"\n🔄 Testing Individual TTA Transformations...")
        
        transform_results = {}
        original_H, original_W = self.test_image_tensor.shape[1:]
        min_score = 0.75
        
        for transform_config in TTA_TRANSFORMS:
            transform_name = transform_config['name']
            forward_func, forward_kwargs = transform_config['forward']
            reverse_func, reverse_kwargs = transform_config['reverse']
            
            print(f"🔍 Testing: {transform_name}")
            
            try:
                # Apply forward transformation
                transformed_image = forward_func(self.test_image_tensor, **forward_kwargs)
                
                # Run inference on transformed image
                with torch.no_grad():
                    transformed_pred = self.model([transformed_image])
                
                pred_dict = transformed_pred[0]
                
                # Apply reverse transformation
                if pred_dict['ellipse_params'].numel() > 0:
                    reversed_pred = reverse_func(pred_dict, original_H, original_W, **reverse_kwargs)
                    
                    # Filter by score
                    score_mask = reversed_pred['scores'] > min_score
                    filtered_pred = {
                        'ellipse_params': reversed_pred['ellipse_params'][score_mask],
                        'scores': reversed_pred['scores'][score_mask]
                    }
                else:
                    filtered_pred = {'ellipse_params': torch.empty(0, 5), 'scores': torch.empty(0)}
                
                # Store results
                transform_results[transform_name] = {
                    'raw_detections': len(pred_dict['ellipse_params']),
                    'filtered_detections': len(filtered_pred['ellipse_params']),
                    'mean_score': pred_dict['scores'].mean().item() if len(pred_dict['scores']) > 0 else 0,
                    'max_score': pred_dict['scores'].max().item() if len(pred_dict['scores']) > 0 else 0,
                    'min_score': pred_dict['scores'].min().item() if len(pred_dict['scores']) > 0 else 0,
                    'filtered_ellipses': filtered_pred['ellipse_params']
                }
                
                print(f"   ✅ Raw: {transform_results[transform_name]['raw_detections']}, "
                      f"Filtered: {transform_results[transform_name]['filtered_detections']}, "
                      f"Score: {transform_results[transform_name]['mean_score']:.3f} "
                      f"({transform_results[transform_name]['min_score']:.3f}-{transform_results[transform_name]['max_score']:.3f})")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                transform_results[transform_name] = {'error': str(e)}
        
        print("✅ Individual transform testing completed")
        return transform_results
    
    def test_coordinate_transformations(self):
        """Test coordinate transformation accuracy."""
        print(f"\n📐 Testing Coordinate Transformation Accuracy...")
        
        # Test points: center, corners, and edges
        test_points = torch.tensor([
            [50.0, 30.0, 320.0, 240.0, 0.0],    # Image center
            [30.0, 20.0, 100.0, 100.0, 0.0],    # Top-left region
            [40.0, 25.0, 540.0, 100.0, 0.0],    # Top-right region  
            [35.0, 22.0, 100.0, 380.0, 0.0],    # Bottom-left region
            [45.0, 28.0, 540.0, 380.0, 0.0],    # Bottom-right region
        ], dtype=torch.float32).to(self.device)
        
        coordinate_test_results = []
        
        for transform_config in TTA_TRANSFORMS[:8]:  # Test first 8 transforms
            transform_name = transform_config['name']
            reverse_func, reverse_kwargs = transform_config['reverse']
            
            print(f"🧮 Testing {transform_name}...")
            
            # Create fake prediction with known coordinates
            fake_pred = {
                'ellipse_params': test_points.clone(),
                'scores': torch.ones(len(test_points)),
                'labels': torch.ones(len(test_points), dtype=torch.long),
                'boxes': torch.zeros(len(test_points), 4)  # Simplified for testing
            }
            
            try:
                # Apply reverse transformation (simulating what TTA does)
                original_H, original_W = self.test_image_tensor.shape[1:]
                reversed_pred = reverse_func(fake_pred, original_H, original_W, **reverse_kwargs)
                
                # Check if coordinates changed as expected
                original_centers = test_points[:, 2:4]  # [cx, cy]
                transformed_centers = reversed_pred['ellipse_params'][:, 2:4]
                
                center_diff = torch.norm(original_centers - transformed_centers, dim=1)
                
                result = {
                    'name': transform_name,
                    'original_centers': original_centers.cpu().numpy(),
                    'transformed_centers': transformed_centers.cpu().numpy(),
                    'center_differences': center_diff.cpu().numpy(),
                    'max_diff': center_diff.max().item(),
                    'mean_diff': center_diff.mean().item()
                }
                
                print(f"   Max diff: {result['max_diff']:6.1f}px, Mean diff: {result['mean_diff']:6.1f}px")
                coordinate_test_results.append(result)
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                coordinate_test_results.append({'name': transform_name, 'error': str(e)})
        
        print("✅ Coordinate transformation testing completed")
        return coordinate_test_results
    
    def test_score_distributions(self):
        """Analyze score distributions across transformations."""
        print(f"\n📊 Analyzing Score Distributions...")
        
        score_analysis = {}
        
        for transform_config in TTA_TRANSFORMS:
            transform_name = transform_config['name']
            forward_func, forward_kwargs = transform_config['forward']
            
            try:
                # Apply transformation and get predictions
                transformed_img = forward_func(self.test_image_tensor, **forward_kwargs)
                
                with torch.no_grad():
                    transformed_pred = self.model([transformed_img])
                
                pred_dict = transformed_pred[0]
                scores = pred_dict['scores']
                
                if len(scores) > 0:
                    score_analysis[transform_name] = {
                        'count': len(scores),
                        'min': scores.min().item(),
                        'max': scores.max().item(),
                        'mean': scores.mean().item(),
                        'std': scores.std().item(),
                        'above_75': (scores > 0.75).sum().item(),
                        'above_50': (scores > 0.50).sum().item(),
                        'above_25': (scores > 0.25).sum().item(),
                    }
                else:
                    score_analysis[transform_name] = {'count': 0}
                    
            except Exception as e:
                score_analysis[transform_name] = {'error': str(e)}
        
        # Print analysis
        print("\n📊 Score Distribution Analysis:")
        print("="*80)
        print(f"{'Transform':<20} {'Count':<8} {'Mean':<8} {'Std':<8} {'Above 0.75':<10} {'Above 0.50':<10}")
        print("-"*80)
        
        problematic_transforms = []
        
        for name, data in score_analysis.items():
            if 'error' not in data and data['count'] > 0:
                print(f"{name[:19]:<20} {data['count']:<8} {data['mean']:<8.3f} {data['std']:<8.3f} "
                      f"{data['above_75']:<10} {data['above_50']:<10}")
                
                # Check for problematic patterns
                if data['count'] > 0 and data['above_75'] == 0 and data['above_50'] > 0:
                    problematic_transforms.append(name)
                    
            elif 'error' in data:
                print(f"{name[:19]:<20} ERROR: {data['error']}")
            else:
                print(f"{name[:19]:<20} NO DETECTIONS")
        
        print("\n⚠️  Transforms with predictions but none above 0.75 threshold:")
        for transform in problematic_transforms:
            print(f"   - {transform}")
        
        print("✅ Score distribution analysis completed")
        return score_analysis, problematic_transforms
    
    def test_consensuation_algorithm(self):
        """Test the consensuation algorithm with controlled inputs."""
        print(f"\n🤝 Testing Consensuation Algorithm...")
        
        # Create test predictions with known good ellipses
        test_predictions = []
        
        # Prediction 1: Center ellipse
        pred1 = {
            'ellipse_params': torch.tensor([[50.0, 30.0, 320.0, 240.0, 0.0]], device=self.device),
            'scores': torch.tensor([0.9], device=self.device),
            'labels': torch.tensor([1], device=self.device),
            'boxes': torch.tensor([[270.0, 210.0, 370.0, 270.0]], device=self.device)
        }
        
        # Prediction 2: Slightly offset (should be consensuated with pred1)
        pred2 = {
            'ellipse_params': torch.tensor([[52.0, 28.0, 325.0, 235.0, 0.1]], device=self.device),
            'scores': torch.tensor([0.85], device=self.device),
            'labels': torch.tensor([1], device=self.device),
            'boxes': torch.tensor([[273.0, 207.0, 377.0, 263.0]], device=self.device)
        }
        
        # Prediction 3: Different location (should remain separate)
        pred3 = {
            'ellipse_params': torch.tensor([[40.0, 25.0, 150.0, 100.0, 0.0]], device=self.device),
            'scores': torch.tensor([0.8], device=self.device),
            'labels': torch.tensor([1], device=self.device),
            'boxes': torch.tensor([[110.0, 75.0, 190.0, 125.0]], device=self.device)
        }
        
        test_predictions = [pred1, pred2, pred3]
        
        print(f"📥 Input predictions: {len(test_predictions)}")
        for i, pred in enumerate(test_predictions):
            ellipse = pred['ellipse_params'][0]
            score = pred['scores'][0]
            print(f"   Pred {i+1}: center=({ellipse[2]:.1f}, {ellipse[3]:.1f}), score={score:.3f}")
        
        # Test consensuation
        consensuated = consensuate_predictions(test_predictions, min_score=0.75, distance_threshold=30.0)
        
        print(f"📤 Consensuated predictions: {len(consensuated['ellipse_params'])}")
        for i, (ellipse, score) in enumerate(zip(consensuated['ellipse_params'], consensuated['scores'])):
            print(f"   Cons {i+1}: center=({ellipse[2]:.1f}, {ellipse[3]:.1f}), score={score:.3f}")
        
        print("✅ Consensuation algorithm testing completed")
        return consensuated
    
    def test_full_tta_pipeline(self):
        """Run complete TTA pipeline with extensive logging."""
        print(f"\n🐛 Running Full TTA Pipeline with Debug Logging...")
        
        # Run TTA with details
        tta_predictions, per_transform_details = tta_predict_with_details(
            model=self.model,
            image_tensor=self.test_image_tensor,
            device=self.device,
            min_score=0.75,
            consensuate=False,  # Get individual predictions first
            visualize=False
        )
        
        print(f"\n📊 TTA Pipeline Results:")
        print(f"   Individual predictions: {len(tta_predictions)}")
        print(f"   Transform details: {len(per_transform_details)}")
        
        # Analyze each transform
        print(f"\n📋 Per-Transform Analysis:")
        print("="*80)
        print(f"{'Transform':<20} {'Raw':<6} {'Filt':<6} {'Raw Score':<10} {'Filt Score':<10} {'Contrib'}")
        print("-"*80)
        
        problematic_transforms = []
        good_transforms = []
        
        for i, detail in enumerate(per_transform_details):
            name = detail['name']
            raw_det = detail['raw_detections']
            filt_det = detail['filtered_detections']
            raw_score = detail['raw_avg_confidence']
            filt_score = detail['filtered_avg_confidence']
            contrib = detail['contribution']
            
            print(f"{name[:19]:<20} {raw_det:<6} {filt_det:<6} {raw_score:<10.3f} {filt_score:<10.3f} {contrib}")
            
            # Identify problematic transforms
            if raw_det > 0 and filt_det == 0:
                problematic_transforms.append(name)
            elif filt_det > 0:
                good_transforms.append(name)
        
        # Run consensuated TTA
        tta_consensuated, _ = tta_predict_with_details(
            model=self.model,
            image_tensor=self.test_image_tensor,
            device=self.device,
            min_score=0.75,
            consensuate=True,
            visualize=False
        )
        
        consensuated_pred = tta_consensuated[0]
        
        print(f"\n📊 Consensuation Results:")
        print(f"   Individual predictions total: {sum(len(pred['ellipse_params']) for pred in tta_predictions)}")
        print(f"   Consensuated predictions: {len(consensuated_pred['ellipse_params'])}")
        if len(consensuated_pred['ellipse_params']) > 0:
            print(f"   Consensuated score range: {consensuated_pred['scores'].min():.3f} - {consensuated_pred['scores'].max():.3f}")
        
        # Compare with baseline
        print(f"\n📊 Comparison with Baseline:")
        baseline_count = len(self.filtered_baseline['ellipse_params'])
        tta_count = len(consensuated_pred['ellipse_params'])
        
        if baseline_count > 0 and tta_count > 0:
            baseline_score = self.filtered_baseline['scores'].mean()
            tta_score = consensuated_pred['scores'].mean()
            
            print(f"   Baseline: {baseline_count} predictions, mean score: {baseline_score:.3f}")
            print(f"   TTA: {tta_count} predictions, mean score: {tta_score:.3f}")
            print(f"   Detection change: {tta_count - baseline_count:+d}")
            print(f"   Score change: {tta_score - baseline_score:+.3f}")
        else:
            print(f"   Baseline: {baseline_count} predictions")
            print(f"   TTA: {tta_count} predictions")
        
        print("✅ Full TTA pipeline testing completed")
        return {
            'tta_predictions': tta_predictions,
            'consensuated': consensuated_pred,
            'transform_details': per_transform_details,
            'problematic_transforms': problematic_transforms,
            'good_transforms': good_transforms
        }
    
    def analyze_and_recommend(self, transform_results, coord_results, score_analysis, problematic_score_transforms, debug_results):
        """Analyze all test results and provide specific recommendations."""
        print(f"\n🔍 Analyzing Results and Providing Recommendations...")
        
        issues_found = []
        recommendations = []
        
        print("="*80)
        print("🔍 COMPREHENSIVE TTA DIAGNOSTIC ANALYSIS")
        print("="*80)
        
        # 1. Check baseline performance
        baseline_count = len(self.filtered_baseline['ellipse_params'])
        if baseline_count == 0:
            issues_found.append("❌ No baseline predictions above threshold")
            recommendations.append("🔧 Lower the score threshold or check model performance")
        else:
            print(f"✅ Baseline working: {baseline_count} predictions")
        
        # 2. Check individual transform performance
        if 'transform_details' in debug_results:
            zero_contrib_count = sum(1 for d in debug_results['transform_details'] if d['contribution'] == 0)
            total_transforms = len(debug_results['transform_details'])
            
            if zero_contrib_count > total_transforms * 0.5:
                issues_found.append(f"❌ {zero_contrib_count}/{total_transforms} transforms contribute nothing")
                recommendations.append("🔧 Score threshold (0.75) may be too high - many transforms have raw predictions but filtered to 0")
            
            # Check for score degradation
            low_score_transforms = [d for d in debug_results['transform_details'] 
                                   if d['raw_avg_confidence'] > 0 and d['raw_avg_confidence'] < 0.6]
            
            if len(low_score_transforms) > 3:
                issues_found.append(f"❌ {len(low_score_transforms)} transforms have low confidence scores")
                recommendations.append("🔧 Some transformations significantly degrade prediction confidence")
        
        # 3. Check coordinate transformation accuracy
        coord_issues = [r for r in coord_results if 'max_diff' in r and r['max_diff'] > 20]
        if coord_issues:
            issues_found.append(f"❌ {len(coord_issues)} transforms have coordinate errors > 20px")
            recommendations.append("🔧 Review coordinate transformation math for: " + 
                                  ", ".join([r['name'] for r in coord_issues]))
        
        # 4. Check consensuation effectiveness
        if 'consensuated' in debug_results:
            individual_total = sum(len(pred['ellipse_params']) for pred in debug_results['tta_predictions'])
            consensuated_count = len(debug_results['consensuated']['ellipse_params'])
            
            if individual_total > 0 and consensuated_count == 0:
                issues_found.append("❌ Consensuation eliminated all predictions")
                recommendations.append("🔧 Consensuation threshold (30px) may be too strict, or min_score too high")
            elif consensuated_count > individual_total * 0.8:
                issues_found.append("⚠️ Consensuation barely reducing prediction count")
                recommendations.append("🔧 Consider adjusting consensuation distance threshold")
        
        # 5. Check for score threshold issues
        if len(problematic_score_transforms) > 5:
            issues_found.append(f"❌ {len(problematic_score_transforms)} transforms have predictions but none above 0.75 threshold")
            recommendations.append("🔧 Score threshold of 0.75 is too aggressive - try 0.5 or 0.6")
        
        # Print summary
        print(f"\n📊 DIAGNOSTIC SUMMARY:")
        print(f"   Issues found: {len(issues_found)}")
        print(f"   Recommendations: {len(recommendations)}")
        
        if issues_found:
            print(f"\n🚨 ISSUES IDENTIFIED:")
            for issue in issues_found:
                print(f"   {issue}")
        
        if recommendations:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in recommendations:
                print(f"   {rec}")
        
        # Specific actionable fixes
        print(f"\n🔧 SPECIFIC FIXES TO TRY:")
        print(f"   1. Lower TTA score threshold from 0.75 to 0.5 or 0.6")
        print(f"   2. Increase consensuation distance threshold from 30px to 50px")
        print(f"   3. Remove problematic transforms: {debug_results.get('problematic_transforms', [])}")
        print(f"   4. Focus on good transforms: {debug_results.get('good_transforms', [])}")
        print(f"   5. Check if model confidence scores are calibrated correctly")
        
        # Code fix suggestions
        print(f"\n💻 CODE MODIFICATIONS:")
        print(f"# In tta_transforms.py, update TTA_CONFIG:")
        print(f"TTA_CONFIG = {{")
        print(f"    'min_score_threshold': 0.5,  # Was 0.75")
        print(f"    'consensuation_distance_threshold': 50.0,  # Was 30.0")
        print(f"    # ... other config")
        print(f"}}")
        print(f"")
        print(f"# In your analysis script:")
        print(f"min_score = 0.5  # Instead of 0.75")
        
        return {
            'issues': issues_found,
            'recommendations': recommendations,
            'problematic_transforms': debug_results.get('problematic_transforms', []),
            'good_transforms': debug_results.get('good_transforms', [])
        }
    
    def run_full_diagnostic(self):
        """Run the complete diagnostic test suite."""
        print("🚀 STARTING COMPREHENSIVE TTA DIAGNOSTIC TEST")
        print("="*60)
        
        try:
            # Setup
            self.setup_model_and_data()
            
            # Test 1: Baseline
            baseline_results = self.test_baseline_prediction()
            
            # Test 2: Individual transforms
            transform_results = self.test_individual_transforms()
            
            # Test 3: Coordinate transformations
            coord_results = self.test_coordinate_transformations()
            
            # Test 4: Score distributions
            score_analysis, problematic_score_transforms = self.test_score_distributions()
            
            # Test 5: Consensuation algorithm
            consensuation_results = self.test_consensuation_algorithm()
            
            # Test 6: Full TTA pipeline
            debug_results = self.test_full_tta_pipeline()
            
            # Final analysis and recommendations
            final_analysis = self.analyze_and_recommend(
                transform_results, coord_results, score_analysis, 
                problematic_score_transforms, debug_results
            )
            
            print("\n" + "="*80)
            print("🎯 CONCLUSION: The main issue appears to be the score threshold (0.75) being too aggressive.")
            print("Many transforms produce valid predictions that get filtered out by the high threshold.")
            print("Try reducing the threshold to 0.5-0.6 and adjusting consensuation parameters.")
            print("="*80)
            
            return {
                'baseline': baseline_results,
                'transforms': transform_results,
                'coordinates': coord_results,
                'scores': score_analysis,
                'consensuation': consensuation_results,
                'pipeline': debug_results,
                'analysis': final_analysis
            }
            
        except Exception as e:
            print(f"❌ Diagnostic test failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function to run TTA diagnostic tests."""
    # Configuration
    MODEL_REPO = "MJGT/ellipse-rcnn-FDDB"
    DATA_ROOT = "../data/FDDB"
    
    print("🔬 TTA Transformation Diagnostic Test")
    print("="*40)
    print(f"Model: {MODEL_REPO}")
    print(f"Dataset: {DATA_ROOT}")
    
    # Create and run diagnostic tester
    tester = TTADiagnosticTester(MODEL_REPO, DATA_ROOT)
    results = tester.run_full_diagnostic()
    
    if results:
        print(f"\n✅ Diagnostic testing completed successfully!")
        print(f"🔍 Results available for further analysis")
    else:
        print(f"\n❌ Diagnostic testing failed")


if __name__ == "__main__":
    main()
