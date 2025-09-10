"""
TTA Configuration Evaluator

This script evaluates different TTA configurations to find optimal parameter settings.
It runs multiple evaluations with different TTA_CONFIG, QUALITY_CONFIG, and VALIDATION_CONFIG
values to minimize false positives and missed targets.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
from pathlib import Path
from tqdm import tqdm

# Add parent directories to path for imports
sys.path.insert(0, '..')
sys.path.insert(0, '../..')

import torch
import torchvision.transforms.functional as F
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.hf import EllipseRCNN

# Import from local tta_config module
from tta_config.tta_transforms import tta_predict, update_tta_configs, get_current_configs
from tta_config.configurations import (
    get_all_configurations, 
    get_top_performer_configurations,
    get_baseline_configurations,
    get_experimental_configurations,
    CONFIGURATION_INFO
)


class TTAConfigurationEvaluator:
    """
    Evaluates different TTA configurations to find optimal settings.
    """
    
    def __init__(self, model_repo: str = "MJGT/ellipse-rcnn-FDDB",
                 data_path: str = "../data/FDDB",
                 device: str = "cuda"):
        """
        Initialize the configuration evaluator.
        
        Args:
            model_repo: Hugging Face model repository
            data_path: Path to FDDB dataset
            device: Device to run inference on
        """
        self.model_repo = model_repo
        self.data_path = data_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"🔬 TTA Configuration Evaluator initialized")
        print(f"🤗 Model: {self.model_repo}")
        print(f"📁 Data: {self.data_path}")
        print(f"💻 Device: {self.device}")
        
        # GPU information
        if torch.cuda.is_available():
            print(f"🚀 CUDA Available: {torch.cuda.device_count()} GPU(s)")
            print(f"   Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"⚠️  CUDA not available, using CPU")
        
        # Load model and dataset
        self.model = EllipseRCNN.from_pretrained(self.model_repo)
        self.model.eval()
        self.model.to(self.device)
        
        print(f"✅ Model loaded on {next(self.model.parameters()).device}")
        
        # Results storage
        self.evaluation_results = []
        self.best_config = None
        self.best_score = float('inf')
        
    def define_configuration_grid(self, config_set="all"):
        """
        Get configuration combinations from the configurations module.
        
        Args:
            config_set: Which configuration set to use ('all', 'baseline', 'experimental', 'top_performers')
        """
        if config_set == "baseline":
            return get_baseline_configurations()
        elif config_set == "experimental":
            return get_experimental_configurations()
        elif config_set == "top_performers":
            return get_top_performer_configurations()
        else:  # "all" or any other value
            return get_all_configurations()
    
    def evaluate_single_configuration(self, config: Dict, num_images: int = 100, 
                                    random_seed: int = 42) -> Dict:
        """
        Evaluate a single configuration.
        
        Args:
            config: Configuration dictionary
            num_images: Number of images to evaluate
            random_seed: Random seed for reproducible results
            
        Returns:
            Evaluation results dictionary
        """
        # Update configurations
        update_tta_configs(
            tta_config=config.get('tta_config'),
            quality_config=config.get('quality_config'),
            validation_config=config.get('validation_config')
        )
        
        # Debug: Verify configuration changes
        current_configs = get_current_configs()
        current_tta = current_configs['tta_config']
        current_quality = current_configs['quality_config']
        current_validation = current_configs['validation_config']
        print(f"Debug - {config['name']}: min_score={current_tta.get('min_score_threshold', 'default')}, "
              f"consensus_dist={current_tta.get('consensuation_distance_threshold', 'default')}")
        
        # Load dataset subset
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        dataset = FDDB(self.data_path, download=False)
        total_images = len(dataset.ellipse_dict)
        
        if num_images > total_images:
            num_images = total_images
            
        # Select random subset
        all_keys = list(dataset.ellipse_dict.keys())
        selected_keys = np.random.choice(all_keys, num_images, replace=False)
        selected_ellipse_dict = {k: dataset.ellipse_dict[k] for k in selected_keys}
        subset_dataset = FDDB(self.data_path, ellipse_dict=selected_ellipse_dict)
        
        # Run evaluation
        results = {
            'config_name': config['name'],
            'config': config,
            'base_predictions': [],
            'tta_predictions': [],
            'targets': [],
            'base_fp_count': 0,
            'tta_fp_count': 0,
            'base_missed_count': 0,
            'tta_missed_count': 0,
            'base_precision': 0.0,
            'tta_precision': 0.0,
            'base_recall': 0.0,
            'tta_recall': 0.0,
            'improvement_score': 0.0
        }
        
        valid_images = 0
        # Add progress bar for image processing
        progress_bar = tqdm(
            range(len(subset_dataset)), 
            desc=f"🖼️  {config['name'][:15]:<15}", 
            unit="img",
            ncols=120,  # Wider progress bar
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        for idx in progress_bar:
            try:
                image_tensor, target = subset_dataset[idx]
                
                # Move target tensors to the same device as model
                if 'ellipse_params' in target and target['ellipse_params'].numel() > 0:
                    target['ellipse_params'] = target['ellipse_params'].to(self.device)
                
                # Base model prediction
                with torch.no_grad():
                    base_pred = self.model([image_tensor.to(self.device)])[0]
                
                # TTA prediction
                tta_pred = tta_predict(
                    self.model, image_tensor, self.device,
                    min_score=0.3, consensuate=True, visualize=False  # Lower min_score to capture more
                )[0]
                
                results['base_predictions'].append(base_pred)
                results['tta_predictions'].append(tta_pred)
                results['targets'].append(target)
                valid_images += 1
                
                # Update progress bar with more informative stats
                base_count = (base_pred['scores'] > 0.3).sum().item()
                tta_count = (tta_pred['scores'] > 0.3).sum().item()
                target_count = len(target['ellipse_params']) if 'ellipse_params' in target else 0
                
                progress_bar.set_postfix({
                    'Valid': valid_images,
                    'Base': base_count,
                    'TTA': tta_count, 
                    'GT': target_count
                })
                
            except Exception as e:
                progress_bar.write(f"⚠️ Error processing image {idx}: {e}")
                continue
        
        # Close progress bar
        progress_bar.close()
        
        if valid_images == 0:
            progress_bar.write("❌ No valid images processed")
            return results
        
        # Calculate metrics
        self._calculate_configuration_metrics(results)
        
        return results
    
    def _calculate_configuration_metrics(self, results: Dict):
        """Calculate precision, recall, and improvement metrics for a configuration."""
        base_tp, base_fp, base_fn = 0, 0, 0
        tta_tp, tta_fp, tta_fn = 0, 0, 0
        
        min_score = 0.3  # Match the score used in predictions
        
        for base_pred, tta_pred, target in zip(
            results['base_predictions'], 
            results['tta_predictions'], 
            results['targets']
        ):
            # Filter predictions by score
            base_mask = base_pred['scores'] > min_score
            tta_mask = tta_pred['scores'] > min_score
            
            base_ellipses = base_pred['ellipse_params'][base_mask]
            tta_ellipses = tta_pred['ellipse_params'][tta_mask]
            target_ellipses = target['ellipse_params']
            
            # Calculate TP/FP/FN for base model
            base_metrics = self._calculate_tp_fp_fn(base_ellipses, target_ellipses)
            base_tp += base_metrics['tp']
            base_fp += base_metrics['fp']
            base_fn += base_metrics['fn']
            
            # Calculate TP/FP/FN for TTA model
            tta_metrics = self._calculate_tp_fp_fn(tta_ellipses, target_ellipses)
            tta_tp += tta_metrics['tp']
            tta_fp += tta_metrics['fp'] 
            tta_fn += tta_metrics['fn']
        
        # Calculate precision and recall
        results['base_precision'] = base_tp / (base_tp + base_fp) if (base_tp + base_fp) > 0 else 0.0
        results['base_recall'] = base_tp / (base_tp + base_fn) if (base_tp + base_fn) > 0 else 0.0
        results['tta_precision'] = tta_tp / (tta_tp + tta_fp) if (tta_tp + tta_fp) > 0 else 0.0
        results['tta_recall'] = tta_tp / (tta_tp + tta_fn) if (tta_tp + tta_fn) > 0 else 0.0
        
        # Store counts
        results['base_fp_count'] = base_fp
        results['tta_fp_count'] = tta_fp
        results['base_missed_count'] = base_fn
        results['tta_missed_count'] = tta_fn
        
        # Calculate F1 scores
        base_f1 = 2 * (results['base_precision'] * results['base_recall']) / (results['base_precision'] + results['base_recall']) if (results['base_precision'] + results['base_recall']) > 0 else 0.0
        tta_f1 = 2 * (results['tta_precision'] * results['tta_recall']) / (results['tta_precision'] + results['tta_recall']) if (results['tta_precision'] + results['tta_recall']) > 0 else 0.0
        
        # Improvement score (higher is better)
        # Penalize false positives and missed targets, reward improvements
        fp_improvement = (base_fp - tta_fp) / max(base_fp, 1)  # Positive if TTA reduces FP
        fn_improvement = (base_fn - tta_fn) / max(base_fn, 1)  # Positive if TTA reduces FN
        f1_improvement = tta_f1 - base_f1
        
        results['improvement_score'] = fp_improvement + fn_improvement + f1_improvement
        
        # Results will be shown in progress bar, no need for separate prints
        
    def _calculate_tp_fp_fn(self, predictions: torch.Tensor, targets: torch.Tensor, 
                           distance_threshold: float = 20.0) -> Dict[str, int]:
        """Calculate True Positives, False Positives, and False Negatives."""
        if predictions.numel() == 0 and targets.numel() == 0:
            return {'tp': 0, 'fp': 0, 'fn': 0}
        elif predictions.numel() == 0:
            return {'tp': 0, 'fp': 0, 'fn': len(targets)}
        elif targets.numel() == 0:
            return {'tp': 0, 'fp': len(predictions), 'fn': 0}
        
        # Ensure both tensors are on GPU if available, otherwise same device
        if predictions.device != targets.device:
            # Move targets to same device as predictions (should be GPU)
            targets = targets.to(predictions.device)
        
        # Vectorized distance calculation for better GPU performance
        if len(predictions) == 0 or len(targets) == 0:
            tp = 0
            fp = len(predictions)
            fn = len(targets)
        else:
            # Extract centers: (N, 2) and (M, 2)
            pred_centers = predictions[:, 2:4]  # cx, cy for all predictions
            target_centers = targets[:, 2:4]    # cx, cy for all targets
            
            # Calculate pairwise distances: (N, M)
            # Using broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2) -> (N, M)
            distances = torch.norm(
                pred_centers.unsqueeze(1) - target_centers.unsqueeze(0), 
                dim=2
            )
            
            # Find best matches for each prediction
            matched_targets = set()
            tp = 0
            
            for i in range(len(predictions)):
                # Find closest unmatched target
                valid_distances = distances[i].clone()
                for matched_idx in matched_targets:
                    valid_distances[matched_idx] = float('inf')
                
                min_distance, closest_idx = torch.min(valid_distances, dim=0)
                
                if min_distance.item() < distance_threshold:
                    matched_targets.add(closest_idx.item())
                    tp += 1
        
        fp = len(predictions) - tp
        fn = len(targets) - len(matched_targets)
        
        return {'tp': tp, 'fp': fp, 'fn': fn}
    
    def run_full_evaluation(self, num_images: int = 100, random_seed: int = 42, config_set: str = "all"):
        """
        Run evaluation on all configurations.
        
        Args:
            num_images: Number of images to evaluate per configuration
            random_seed: Random seed for reproducible results
            config_set: Which configuration set to evaluate ('all', 'baseline', 'experimental', 'top_performers')
        """
        print("🚀 Starting TTA Configuration Evaluation")
        print("=" * 60)
        
        configurations = self.define_configuration_grid(config_set)
        
        print(f"⚙️  Configuration Set: {config_set}")
        print(f"📋 Evaluating {len(configurations)} configurations on {num_images} images")
        print(f"🎲 Random seed: {random_seed}")
        
        # Show which configurations will be evaluated
        if len(configurations) <= 10:
            print("\n📝 Configurations to evaluate:")
            for i, config in enumerate(configurations, 1):
                print(f"  {i:2d}. {config['name']}")
        else:
            print(f"\n📝 Configuration names: {', '.join(config['name'] for config in configurations[:5])}...")
        
        print()  # Empty line before progress bars
        
        # Progress bar for configurations
        config_progress = tqdm(
            configurations, 
            desc="🔧 Configurations", 
            unit="config",
            ncols=120,  # Wider progress bar
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}',
            leave=True  # Keep the bar after completion
        )
        
        for config in config_progress:
            try:
                config_progress.set_description(f"🔧 {config['name'][:20]:<20}")
                result = self.evaluate_single_configuration(config, num_images, random_seed)
                self.evaluation_results.append(result)
                
                # Track best configuration
                if result['improvement_score'] > self.best_score:
                    self.best_score = result['improvement_score']
                    self.best_config = config
                
                # Update config progress with detailed metrics
                config_progress.set_postfix({
                    'Best': self.best_config['name'][:12] if self.best_config else 'None',
                    'Score': f"{self.best_score:.3f}",
                    'P': f"{result['tta_precision']:.3f}",
                    'R': f"{result['tta_recall']:.3f}",
                    'FP': result['tta_fp_count'],
                    'FN': result['tta_missed_count']
                })
                    
            except Exception as e:
                config_progress.write(f"❌ Error evaluating {config['name']}: {e}")
                continue
        
        config_progress.close()
        
        # Generate summary
        self._generate_summary_report()
        self._save_results()
        self._plot_results()
        
    def _generate_summary_report(self):
        """Generate and print summary report."""
        if not self.evaluation_results:
            print("❌ No evaluation results available")
            return
            
        print("\n" + "=" * 60)
        print("📊 CONFIGURATION EVALUATION SUMMARY")
        print("=" * 60)
        
        # Sort by improvement score
        sorted_results = sorted(self.evaluation_results, 
                              key=lambda x: x['improvement_score'], reverse=True)
        
        print(f"\n🏆 Best Configuration: {sorted_results[0]['config_name']}")
        print(f"   Improvement Score: {sorted_results[0]['improvement_score']:.3f}")
        print(f"   TTA Precision: {sorted_results[0]['tta_precision']:.3f}")
        print(f"   TTA Recall: {sorted_results[0]['tta_recall']:.3f}")
        print(f"   FP Reduction: {sorted_results[0]['base_fp_count'] - sorted_results[0]['tta_fp_count']}")
        print(f"   FN Reduction: {sorted_results[0]['base_missed_count'] - sorted_results[0]['tta_missed_count']}")
        
        print(f"\n📈 All Configurations (sorted by improvement):")
        print("-" * 60)
        for i, result in enumerate(sorted_results):
            print(f"{i+1:2d}. {result['config_name']:20s} | "
                  f"Score: {result['improvement_score']:6.3f} | "
                  f"P: {result['tta_precision']:.3f} | "
                  f"R: {result['tta_recall']:.3f} | "
                  f"FP: {result['tta_fp_count']:3d} | "
                  f"FN: {result['tta_missed_count']:3d}")
    
    def _save_results(self):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        results_file = results_dir / f"tta_config_evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            # Make results JSON serializable
            serializable_results = []
            for result in self.evaluation_results:
                clean_result = {
                    'config_name': result['config_name'],
                    'config': result['config'],
                    'base_precision': result['base_precision'],
                    'base_recall': result['base_recall'],
                    'tta_precision': result['tta_precision'],
                    'tta_recall': result['tta_recall'],
                    'base_fp_count': result['base_fp_count'],
                    'tta_fp_count': result['tta_fp_count'],
                    'base_missed_count': result['base_missed_count'],
                    'tta_missed_count': result['tta_missed_count'],
                    'improvement_score': result['improvement_score']
                }
                serializable_results.append(clean_result)
            
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Detailed results saved to: {results_file}")
        
        # Save summary as CSV
        csv_data = []
        for result in self.evaluation_results:
            csv_data.append({
                'Configuration': result['config_name'],
                'Base_Precision': result['base_precision'],
                'Base_Recall': result['base_recall'],
                'TTA_Precision': result['tta_precision'],
                'TTA_Recall': result['tta_recall'],
                'Base_FP': result['base_fp_count'],
                'TTA_FP': result['tta_fp_count'],
                'Base_FN': result['base_missed_count'],
                'TTA_FN': result['tta_missed_count'],
                'Improvement_Score': result['improvement_score']
            })
        
        df = pd.DataFrame(csv_data)
        csv_file = results_dir / f"tta_config_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"📋 Summary saved to: {csv_file}")
    
    def _plot_results(self):
        """Generate visualization plots."""
        if not self.evaluation_results:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('TTA Configuration Evaluation Results', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        config_names = [r['config_name'] for r in self.evaluation_results]
        tta_precision = [r['tta_precision'] for r in self.evaluation_results]
        tta_recall = [r['tta_recall'] for r in self.evaluation_results]
        improvement_scores = [r['improvement_score'] for r in self.evaluation_results]
        fp_counts = [r['tta_fp_count'] for r in self.evaluation_results]
        fn_counts = [r['tta_missed_count'] for r in self.evaluation_results]
        
        # Plot 1: Precision vs Recall
        ax1.scatter(tta_recall, tta_precision, s=100, alpha=0.7)
        for i, name in enumerate(config_names):
            ax1.annotate(name, (tta_recall[i], tta_precision[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision vs Recall')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement Scores (maintain original order)
        bars = ax2.bar(range(len(improvement_scores)), improvement_scores, alpha=0.7)
        ax2.set_xlabel('Configurations')
        ax2.set_ylabel('Improvement Score')
        ax2.set_title('Configuration Improvement Scores')
        ax2.set_xticks(range(len(config_names)))
        ax2.set_xticklabels(config_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Color bars by performance
        for i, bar in enumerate(bars):
            if improvement_scores[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Plot 3: False Positives
        ax3.bar(range(len(config_names)), fp_counts, alpha=0.7, color='orange')
        ax3.set_xlabel('Configurations')
        ax3.set_ylabel('False Positives Count')
        ax3.set_title('False Positives by Configuration')
        ax3.set_xticks(range(len(config_names)))
        ax3.set_xticklabels(config_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Missed Targets (False Negatives)
        ax4.bar(range(len(config_names)), fn_counts, alpha=0.7, color='red')
        ax4.set_xlabel('Configurations')
        ax4.set_ylabel('Missed Targets Count')
        ax4.set_title('Missed Targets by Configuration')
        ax4.set_xticks(range(len(config_names)))
        ax4.set_xticklabels(config_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot in results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        plot_file = results_dir / f"tta_config_evaluation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {plot_file}")
        
        plt.show()


def main():
    """Main function to run TTA configuration evaluation."""
    parser = argparse.ArgumentParser(description='TTA Configuration Evaluator')
    parser.add_argument('--num_images', type=int, default=100,
                       help='Number of images to evaluate (default: 100)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--model_repo', type=str, default="MJGT/ellipse-rcnn-FDDB",
                       help='Hugging Face model repository')
    parser.add_argument('--data_path', type=str, default="../data/FDDB",
                       help='Path to FDDB dataset')
    parser.add_argument('--device', type=str, default="cuda",
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--config_set', type=str, default="all",
                       choices=['all', 'baseline', 'experimental', 'top_performers'],
                       help='Which configuration set to evaluate (default: all)')
    
    args = parser.parse_args()
    
    print("🔬 TTA Configuration Evaluator")
    print("=" * 50)
    print(f"🖼️  Images: {args.num_images}")
    print(f"🎲 Seed: {args.random_seed}")
    print(f"🤗 Model: {args.model_repo}")
    print(f"📁 Data: {args.data_path}")
    print(f"💻 Device: {args.device}")
    print(f"⚙️  Config Set: {args.config_set}")
    print()
    
    # Create evaluator
    evaluator = TTAConfigurationEvaluator(
        model_repo=args.model_repo,
        data_path=args.data_path,
        device=args.device
    )
    
    # Run full evaluation
    try:
        evaluator.run_full_evaluation(
            num_images=args.num_images,
            random_seed=args.random_seed,
            config_set=args.config_set
        )
        
        print("\n🎉 Configuration evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
