"""
TTA Evaluation Script

This script evaluates the performance of Test Time Augmentation (TTA) by running
predictions on a sample batch of images with and without TTA, then calculating
various error metrics using the translated ellipse error utility functions.
"""

import torch
import numpy as np
import typer
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import List, Dict, Tuple

from ellipse_rcnn.data.craters import CraterEllipseDataset
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.tta.tta_transforms import tta_predict
from ellipse_rcnn.error_utils.python_files import (
    EllipseParGErrors, EllipseAlgebraicError, GtoA
)

app = typer.Typer(pretty_exceptions_show_locals=False)


def ellipse_params_to_ParG(ellipse_params: torch.Tensor) -> np.ndarray:
    """
    Convert ellipse parameters from model format to geometric format.
    
    Args:
        ellipse_params: Tensor with shape (N, 5) containing [a, b, cx, cy, theta]
        
    Returns:
        Array with shape (N, 5) containing [cx, cy, a, b, theta] (ParG format)
    """
    if ellipse_params.numel() == 0:
        return np.empty((0, 5))
    
    # Convert from [a, b, cx, cy, theta] to [cx, cy, a, b, theta]
    ParG = ellipse_params[:, [2, 3, 0, 1, 4]].detach().cpu().numpy()
    return ParG


def calculate_ellipse_errors(pred_ellipses: torch.Tensor, 
                           target_ellipses: torch.Tensor) -> Dict[str, List[float]]:
    """
    Calculate various ellipse errors between predictions and targets.
    
    Args:
        pred_ellipses: Predicted ellipse parameters
        target_ellipses: Ground truth ellipse parameters
        
    Returns:
        Dictionary containing different error metrics
    """
    errors = {
        'geometric_center': [],
        'geometric_angle': [],
        'geometric_major': [],
        'geometric_minor': [],
        'geometric_area': [],
        'algebraic': []
    }
    
    if pred_ellipses.numel() == 0 or target_ellipses.numel() == 0:
        return errors
    
    # Convert to ParG format
    pred_ParG = ellipse_params_to_ParG(pred_ellipses)
    target_ParG = ellipse_params_to_ParG(target_ellipses)
    
    # For each prediction, find the closest target ellipse
    for pred in pred_ParG:
        if len(target_ParG) == 0:
            continue
            
        # Calculate geometric errors with all targets
        all_errors = []
        for target in target_ParG:
            geom_errors = EllipseParGErrors(target, pred)
            all_errors.append(geom_errors)
        
        # Find the target with minimum center error
        center_errors = [err[0] for err in all_errors]
        min_idx = np.argmin(center_errors)
        best_errors = all_errors[min_idx]
        
        # Store geometric errors
        errors['geometric_center'].append(best_errors[0])
        errors['geometric_angle'].append(best_errors[1])
        errors['geometric_major'].append(best_errors[2])
        errors['geometric_minor'].append(best_errors[3])
        errors['geometric_area'].append(best_errors[4])
        
        # Calculate algebraic error
        try:
            pred_ParA = GtoA(pred, 1)  # code=1 for ellipse
            target_ParA = GtoA(target_ParG[min_idx], 1)
            alg_error = EllipseAlgebraicError(target_ParA, pred_ParA)
            errors['algebraic'].append(alg_error)
        except:
            errors['algebraic'].append(float('inf'))
    
    return errors


def run_single_prediction(model: torch.nn.Module, 
                        image_tensor: torch.Tensor,
                        device: torch.device,
                        min_score: float,
                        use_tta: bool = False) -> torch.Tensor:
    """
    Run prediction on a single image with or without TTA.
    
    Args:
        model: The ellipse detection model
        image_tensor: Input image tensor
        device: Device to run inference on
        min_score: Minimum confidence threshold
        use_tta: Whether to use TTA
        
    Returns:
        Predicted ellipse parameters
    """
    if use_tta:
        tta_predictions = tta_predict(
            model=model,
            image_tensor=image_tensor,
            device=device,
            min_score=min_score,
            consensuate=True,
            visualize=False
        )
        prediction = tta_predictions[0]
    else:
        with torch.no_grad():
            standard_pred = model([image_tensor.to(device)])
        prediction = standard_pred[0]
    
    # Apply score filtering
    if prediction["ellipse_params"].numel() > 0:
        score_mask = prediction["scores"] > min_score
        ellipses = prediction["ellipse_params"][score_mask]
    else:
        ellipses = torch.empty(0, 5)
    
    return ellipses


@app.command()
def evaluate_tta(
    model_path_or_repo: str = typer.Argument(
        ..., help="Path to the model weights or HF repo."
    ),
    data_path: str = typer.Argument(..., help="Path to the dataset directory."),
    min_score: float = typer.Option(
        0.6, help="Minimum score threshold for predictions."
    ),
    dataset: str = typer.Option("FDDB", help="Dataset type: FDDB or Craters"),
    batch_size: int = typer.Option(20, help="Number of images to evaluate"),
    output_dir: str = typer.Option("tta_evaluation_results", help="Output directory for results"),
    device_name: str = typer.Option("cpu", help="Device to use: cpu or cuda")
) -> None:
    """
    Evaluate TTA performance by comparing predictions with and without TTA.
    """
    typer.echo("=== TTA Performance Evaluation ===")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device(device_name)
    typer.echo(f"Using device: {device}")
    
    # Load dataset
    typer.echo(f"Loading {dataset} dataset from {data_path}...")
    match dataset:
        case "FDDB": 
            ds = FDDB(data_path)
        case "Craters": 
            ds = CraterEllipseDataset(data_path, group="validation")
        case _: 
            raise ValueError(f"Unknown dataset: {dataset}")
    
    # Load model
    typer.echo(f"Loading model from {model_path_or_repo}...")
    model = EllipseRCNN.from_pretrained(model_path_or_repo)
    model.eval().to(device)
    
    # Sample random images
    indices = np.random.choice(len(ds), min(batch_size, len(ds)), replace=False)
    typer.echo(f"Evaluating on {len(indices)} random images...")
    
    # Storage for results
    results = {
        'image_idx': [],
        'method': [],
        'num_predictions': [],
        'geometric_center_mean': [],
        'geometric_angle_mean': [],
        'geometric_major_mean': [],
        'geometric_minor_mean': [],
        'geometric_area_mean': [],
        'algebraic_mean': [],
        'geometric_center_std': [],
        'geometric_angle_std': [],
        'geometric_major_std': [],
        'geometric_minor_std': [],
        'geometric_area_std': [],
        'algebraic_std': []
    }
    
    # Evaluate each image
    for idx in tqdm(indices, desc="Processing images"):
        image_tensor, target = ds[idx]
        target_ellipses = target["ellipse_params"]
        
        # Skip images without ground truth ellipses
        if target_ellipses.numel() == 0:
            continue
        
        # Standard prediction (no TTA)
        pred_standard = run_single_prediction(
            model, image_tensor, device, min_score, use_tta=False
        )
        
        # TTA prediction
        pred_tta = run_single_prediction(
            model, image_tensor, device, min_score, use_tta=True
        )
        
        # Calculate errors for both methods
        for method, predictions in [("Standard", pred_standard), ("TTA", pred_tta)]:
            errors = calculate_ellipse_errors(predictions, target_ellipses)
            
            # Store results
            results['image_idx'].append(idx)
            results['method'].append(method)
            results['num_predictions'].append(len(predictions))
            
            # Calculate statistics for each error type
            for error_type in ['geometric_center', 'geometric_angle', 'geometric_major', 
                             'geometric_minor', 'geometric_area', 'algebraic']:
                values = errors[error_type]
                if values:
                    results[f'{error_type}_mean'].append(np.mean(values))
                    results[f'{error_type}_std'].append(np.std(values))
                else:
                    results[f'{error_type}_mean'].append(np.nan)
                    results[f'{error_type}_std'].append(np.nan)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save detailed results
    csv_path = output_path / "detailed_results.csv"
    df.to_csv(csv_path, index=False)
    typer.echo(f"Detailed results saved to {csv_path}")
    
    # Calculate summary statistics
    summary_stats = df.groupby('method').agg({
        'num_predictions': ['mean', 'std'],
        'geometric_center_mean': ['mean', 'std'],
        'geometric_angle_mean': ['mean', 'std'],
        'geometric_major_mean': ['mean', 'std'],
        'geometric_minor_mean': ['mean', 'std'],
        'geometric_area_mean': ['mean', 'std'],
        'algebraic_mean': ['mean', 'std']
    }).round(4)
    
    # Print summary
    typer.echo("\n=== SUMMARY STATISTICS ===")
    print(summary_stats)
    
    # Save summary
    summary_path = output_path / "summary_statistics.csv"
    summary_stats.to_csv(summary_path)
    typer.echo(f"Summary statistics saved to {summary_path}")
    
    # Create comparison plots
    create_comparison_plots(df, output_path)
    
    typer.echo(f"\nEvaluation complete! Results saved to {output_path}")


def create_comparison_plots(df: pd.DataFrame, output_path: Path) -> None:
    """Create comparison plots between TTA and standard predictions."""
    
    # Set plotting style
    sns.set_theme(style="whitegrid")
    
    # Error metrics to plot
    error_metrics = [
        'geometric_center_mean', 'geometric_angle_mean', 
        'geometric_major_mean', 'geometric_minor_mean', 
        'geometric_area_mean', 'algebraic_mean'
    ]
    
    metric_labels = [
        'Center Error', 'Angle Error', 
        'Major Axis Error', 'Minor Axis Error',
        'Area Error', 'Algebraic Error'
    ]
    
    # Create box plots comparing methods
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, label) in enumerate(zip(error_metrics, metric_labels)):
        ax = axes[i]
        
        # Filter out NaN values
        plot_data = df.dropna(subset=[metric])
        
        if not plot_data.empty:
            sns.boxplot(data=plot_data, x='method', y=metric, ax=ax)
            ax.set_title(f'{label} Comparison')
            ax.set_xlabel('Method')
            ax.set_ylabel('Error')
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} Comparison')
    
    plt.tight_layout()
    plt.savefig(output_path / "error_comparison_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create bar plot of mean errors
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Calculate mean errors for each method
    mean_errors = df.groupby('method')[error_metrics].mean()
    
    # Create grouped bar plot
    x = np.arange(len(metric_labels))
    width = 0.35
    
    if 'Standard' in mean_errors.index:
        standard_means = mean_errors.loc['Standard', error_metrics].values
        ax.bar(x - width/2, standard_means, width, label='Standard', alpha=0.8)
    
    if 'TTA' in mean_errors.index:
        tta_means = mean_errors.loc['TTA', error_metrics].values
        ax.bar(x + width/2, tta_means, width, label='TTA', alpha=0.8)
    
    ax.set_xlabel('Error Metrics')
    ax.set_ylabel('Mean Error')
    ax.set_title('Mean Error Comparison: TTA vs Standard')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / "mean_error_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create number of predictions comparison
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.boxplot(data=df, x='method', y='num_predictions', ax=ax)
    ax.set_title('Number of Predictions Comparison')
    ax.set_xlabel('Method')
    ax.set_ylabel('Number of Predictions per Image')
    
    plt.tight_layout()
    plt.savefig(output_path / "predictions_count_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    typer.echo("Comparison plots saved to output directory")


if __name__ == "__main__":
    app()
