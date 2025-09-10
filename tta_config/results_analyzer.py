"""
TTA Configuration Results Analyzer

This script analyzes all CSV files in the results folder to identify the best performing
configurations across different metrics. It calculates mean values while ignoring zeros
to avoid bias in the analysis.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from datetime import datetime


class TTAResultsAnalyzer:
    """
    Analyzes TTA configuration evaluation results from multiple CSV files.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the results analyzer.
        
        Args:
            results_dir: Directory containing CSV result files
        """
        self.results_dir = Path(results_dir)
        self.csv_files = []
        self.combined_data = None
        self.aggregated_stats = None
        
        # Find all CSV files
        self._find_csv_files()
        
    def _find_csv_files(self):
        """Find all CSV files in the results directory."""
        if not self.results_dir.exists():
            raise FileNotFoundError(f"Results directory '{self.results_dir}' not found!")
        
        self.csv_files = list(self.results_dir.glob("tta_config_summary_*.csv"))
        
        if not self.csv_files:
            raise FileNotFoundError(f"No CSV files found in '{self.results_dir}'!")
        
        print(f"📁 Found {len(self.csv_files)} CSV files to analyze:")
        for i, file in enumerate(self.csv_files, 1):
            print(f"   {i:2d}. {file.name}")
        print()
    
    def load_and_combine_data(self):
        """Load and combine data from all CSV files."""
        print("📊 Loading and combining data from all CSV files...")
        
        all_dataframes = []
        
        for csv_file in self.csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['Source_File'] = csv_file.stem  # Add source file identifier
                all_dataframes.append(df)
                print(f"   ✅ Loaded {csv_file.name}: {len(df)} configurations")
            except Exception as e:
                print(f"   ❌ Error loading {csv_file.name}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No valid CSV files could be loaded!")
        
        # Combine all dataframes
        self.combined_data = pd.concat(all_dataframes, ignore_index=True)
        
        print(f"\n📋 Combined dataset:")
        print(f"   • Total rows: {len(self.combined_data)}")
        print(f"   • Unique configurations: {self.combined_data['Configuration'].nunique()}")
        print(f"   • Source files: {self.combined_data['Source_File'].nunique()}")
        print()
        
        # Show configuration distribution
        config_counts = self.combined_data['Configuration'].value_counts()
        print(f"🔢 Configuration occurrence counts:")
        for config, count in config_counts.head(10).items():
            print(f"   {config:<25}: {count:2d} runs")
        if len(config_counts) > 10:
            print(f"   ... and {len(config_counts) - 10} more configurations")
        print()
    
    def calculate_aggregated_statistics(self):
        """Calculate aggregated statistics for each configuration, ignoring zeros."""
        print("🧮 Calculating aggregated statistics (ignoring zeros)...")
        
        # Metrics to analyze
        metrics = [
            'Improvement_Score',
            'TTA_Precision', 
            'TTA_Recall',
            'TTA_FP',
            'TTA_FN',
            'Base_FP',
            'Base_FN'
        ]
        
        aggregated_data = []
        
        for config_name in self.combined_data['Configuration'].unique():
            config_data = self.combined_data[self.combined_data['Configuration'] == config_name]
            
            row = {'Configuration': config_name, 'Run_Count': len(config_data)}
            
            for metric in metrics:
                if metric in config_data.columns:
                    values = config_data[metric].values
                    
                    # For improvement score, precision, and recall: don't ignore zeros as they're meaningful
                    if metric in ['Improvement_Score', 'TTA_Precision', 'TTA_Recall']:
                        row[f'{metric}_Mean'] = np.mean(values)
                        row[f'{metric}_Std'] = np.std(values)
                        row[f'{metric}_Min'] = np.min(values)
                        row[f'{metric}_Max'] = np.max(values)
                    else:
                        # For counts (FP, FN): ignore zeros to avoid bias towards configurations with many zero runs
                        non_zero_values = values[values > 0]
                        if len(non_zero_values) > 0:
                            row[f'{metric}_Mean'] = np.mean(non_zero_values)
                            row[f'{metric}_Std'] = np.std(non_zero_values)
                            row[f'{metric}_Min'] = np.min(non_zero_values)
                            row[f'{metric}_Max'] = np.max(non_zero_values)
                            row[f'{metric}_NonZero_Count'] = len(non_zero_values)
                        else:
                            row[f'{metric}_Mean'] = 0
                            row[f'{metric}_Std'] = 0
                            row[f'{metric}_Min'] = 0
                            row[f'{metric}_Max'] = 0
                            row[f'{metric}_NonZero_Count'] = 0
                        
                        # Also include zero count for context
                        row[f'{metric}_Zero_Count'] = len(values) - len(non_zero_values)
            
            aggregated_data.append(row)
        
        self.aggregated_stats = pd.DataFrame(aggregated_data)
        
        print(f"✅ Calculated statistics for {len(self.aggregated_stats)} unique configurations")
        print()
    
    def get_best_performers(self, metric: str, ascending: bool = True, top_n: int = 10) -> pd.DataFrame:
        """
        Get the best performing configurations for a specific metric.
        
        Args:
            metric: Metric column name (with '_Mean' suffix)
            ascending: Whether lower values are better (True) or higher values are better (False)
            top_n: Number of top configurations to return
            
        Returns:
            DataFrame with top performing configurations
        """
        if self.aggregated_stats is None:
            raise ValueError("Must call calculate_aggregated_statistics() first!")
        
        if metric not in self.aggregated_stats.columns:
            available_metrics = [col for col in self.aggregated_stats.columns if col.endswith('_Mean')]
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {available_metrics}")
        
        # Sort by the specified metric
        sorted_data = self.aggregated_stats.sort_values(metric, ascending=ascending)
        
        # Filter out configurations with too few runs (less than 2) for reliability
        reliable_data = sorted_data[sorted_data['Run_Count'] >= 2]
        
        return reliable_data.head(top_n)
    
    def generate_comprehensive_report(self, top_n: int = 10):
        """Generate comprehensive analysis report."""
        print("📈 COMPREHENSIVE TTA CONFIGURATION ANALYSIS REPORT")
        print("=" * 80)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Generated: {timestamp}")
        print(f"Data sources: {len(self.csv_files)} CSV files")
        print(f"Total evaluations: {len(self.combined_data)}")
        print(f"Unique configurations: {self.combined_data['Configuration'].nunique()}")
        print()
        
        # 1. Best Overall Improvement Score
        print("🏆 TOP PERFORMERS - OVERALL IMPROVEMENT SCORE (Higher is Better)")
        print("-" * 80)
        best_improvement = self.get_best_performers('Improvement_Score_Mean', ascending=False, top_n=top_n)
        
        for i, (_, row) in enumerate(best_improvement.iterrows(), 1):
            print(f"{i:2d}. {row['Configuration']:<25} | "
                  f"Score: {row['Improvement_Score_Mean']:7.4f} ± {row['Improvement_Score_Std']:6.4f} | "
                  f"Runs: {row['Run_Count']:2d} | "
                  f"Range: [{row['Improvement_Score_Min']:6.3f}, {row['Improvement_Score_Max']:6.3f}]")
        print()
        
        # 2. Lowest False Positives
        print("🎯 BEST PRECISION - LOWEST FALSE POSITIVES (Lower is Better)")
        print("-" * 80)
        best_fp = self.get_best_performers('TTA_FP_Mean', ascending=True, top_n=top_n)
        
        for i, (_, row) in enumerate(best_fp.iterrows(), 1):
            fp_mean = row['TTA_FP_Mean']
            fp_std = row['TTA_FP_Std']
            fp_min = row['TTA_FP_Min']
            fp_max = row['TTA_FP_Max']
            zero_count = row.get('TTA_FP_Zero_Count', 0)
            nonzero_count = row.get('TTA_FP_NonZero_Count', 0)
            
            print(f"{i:2d}. {row['Configuration']:<25} | "
                  f"FP: {fp_mean:5.2f} ± {fp_std:5.2f} | "
                  f"Runs: {row['Run_Count']:2d} ({zero_count}z, {nonzero_count}nz) | "
                  f"Range: [{fp_min:.0f}, {fp_max:.0f}]")
        print()
        
        # 3. Lowest Missed Targets
        print("🔍 BEST RECALL - LOWEST MISSED TARGETS (Lower is Better)")
        print("-" * 80)
        best_fn = self.get_best_performers('TTA_FN_Mean', ascending=True, top_n=top_n)
        
        for i, (_, row) in enumerate(best_fn.iterrows(), 1):
            fn_mean = row['TTA_FN_Mean']
            fn_std = row['TTA_FN_Std']
            fn_min = row['TTA_FN_Min']
            fn_max = row['TTA_FN_Max']
            zero_count = row.get('TTA_FN_Zero_Count', 0)
            nonzero_count = row.get('TTA_FN_NonZero_Count', 0)
            
            print(f"{i:2d}. {row['Configuration']:<25} | "
                  f"FN: {fn_mean:5.2f} ± {fn_std:5.2f} | "
                  f"Runs: {row['Run_Count']:2d} ({zero_count}z, {nonzero_count}nz) | "
                  f"Range: [{fn_min:.0f}, {fn_max:.0f}]")
        print()
        
        # 4. Best Precision
        print("📐 HIGHEST PRECISION (Higher is Better)")
        print("-" * 80)
        best_precision = self.get_best_performers('TTA_Precision_Mean', ascending=False, top_n=top_n)
        
        for i, (_, row) in enumerate(best_precision.iterrows(), 1):
            print(f"{i:2d}. {row['Configuration']:<25} | "
                  f"Precision: {row['TTA_Precision_Mean']:.4f} ± {row['TTA_Precision_Std']:.4f} | "
                  f"Runs: {row['Run_Count']:2d} | "
                  f"Range: [{row['TTA_Precision_Min']:.3f}, {row['TTA_Precision_Max']:.3f}]")
        print()
        
        # 5. Best Recall
        print("🔎 HIGHEST RECALL (Higher is Better)")
        print("-" * 80)
        best_recall = self.get_best_performers('TTA_Recall_Mean', ascending=False, top_n=top_n)
        
        for i, (_, row) in enumerate(best_recall.iterrows(), 1):
            print(f"{i:2d}. {row['Configuration']:<25} | "
                  f"Recall: {row['TTA_Recall_Mean']:.4f} ± {row['TTA_Recall_Std']:.4f} | "
                  f"Runs: {row['Run_Count']:2d} | "
                  f"Range: [{row['TTA_Recall_Min']:.3f}, {row['TTA_Recall_Max']:.3f}]")
        print()
        
        # 6. Most Consistent Performers (lowest std deviation in improvement score)
        print("🎯 MOST CONSISTENT PERFORMERS (Lowest Improvement Score Std Dev)")
        print("-" * 80)
        consistent_performers = self.aggregated_stats[self.aggregated_stats['Run_Count'] >= 3].sort_values('Improvement_Score_Std')
        
        for i, (_, row) in enumerate(consistent_performers.head(top_n).iterrows(), 1):
            print(f"{i:2d}. {row['Configuration']:<25} | "
                  f"Std: {row['Improvement_Score_Std']:.4f} | "
                  f"Mean: {row['Improvement_Score_Mean']:7.4f} | "
                  f"Runs: {row['Run_Count']:2d} | "
                  f"Range: [{row['Improvement_Score_Min']:6.3f}, {row['Improvement_Score_Max']:6.3f}]")
        print()
        
        # 7. Configuration Insights
        print("💡 CONFIGURATION INSIGHTS")
        print("-" * 80)
        
        # Most tested configurations
        most_tested = self.aggregated_stats.sort_values('Run_Count', ascending=False).head(5)
        print("Most tested configurations:")
        for i, (_, row) in enumerate(most_tested.iterrows(), 1):
            print(f"  {i}. {row['Configuration']:<25}: {row['Run_Count']:2d} runs")
        print()
        
        # Configurations with perfect scores in some runs
        perfect_improvement = self.combined_data[self.combined_data['Improvement_Score'] > 0.5]
        if len(perfect_improvement) > 0:
            print("Configurations achieving high improvement scores (> 0.5):")
            perfect_configs = perfect_improvement['Configuration'].value_counts().head(5)
            for config, count in perfect_configs.items():
                print(f"  • {config}: {count} times")
        else:
            print("No configurations achieved improvement scores > 0.5")
        print()
        
        # Zero false positives achievers
        zero_fp = self.combined_data[self.combined_data['TTA_FP'] == 0]
        if len(zero_fp) > 0:
            print("Configurations achieving zero false positives:")
            zero_fp_configs = zero_fp['Configuration'].value_counts().head(5)
            for config, count in zero_fp_configs.items():
                print(f"  • {config}: {count} times")
        else:
            print("No configurations achieved zero false positives")
        print()
        
        return {
            'best_improvement': best_improvement,
            'best_fp': best_fp,
            'best_fn': best_fn,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'most_consistent': consistent_performers.head(top_n)
        }
    
    def save_analysis_report(self, results: Dict, output_file: str = None):
        """Save analysis results to CSV files."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"analysis_report_{timestamp}"
        
        # Create analysis results directory
        analysis_dir = self.results_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Save individual rankings
        for metric_name, df in results.items():
            if df is not None and len(df) > 0:
                csv_file = analysis_dir / f"{output_file}_{metric_name}.csv"
                df.to_csv(csv_file, index=False)
                print(f"💾 Saved {metric_name} rankings to: {csv_file}")
        
        # Save complete aggregated statistics
        stats_file = analysis_dir / f"{output_file}_complete_stats.csv"
        self.aggregated_stats.to_csv(stats_file, index=False)
        print(f"💾 Saved complete statistics to: {stats_file}")
        print()
    
    def run_complete_analysis(self, top_n: int = 10, save_results: bool = True):
        """Run the complete analysis pipeline."""
        print("🚀 Starting comprehensive TTA configuration analysis...")
        print("=" * 60)
        
        # Load and process data
        self.load_and_combine_data()
        self.calculate_aggregated_statistics()
        
        # Generate report
        results = self.generate_comprehensive_report(top_n)
        
        # Save results
        if save_results:
            self.save_analysis_report(results)
        
        print("✅ Analysis completed successfully!")
        return results


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='TTA Configuration Results Analyzer')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing CSV result files (default: results)')
    parser.add_argument('--top_n', type=int, default=10,
                       help='Number of top configurations to show for each metric (default: 10)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save analysis results to files')
    
    args = parser.parse_args()
    
    try:
        # Create analyzer and run analysis
        analyzer = TTAResultsAnalyzer(results_dir=args.results_dir)
        analyzer.run_complete_analysis(
            top_n=args.top_n,
            save_results=not args.no_save
        )
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
