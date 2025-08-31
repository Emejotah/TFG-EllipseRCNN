import torch
import typer
from matplotlib import pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

from ellipse_rcnn.data.craters import CraterEllipseDataset
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.utils.viz import plot_ellipses, plot_bboxes
from ellipse_rcnn.tta.tta_transforms import tta_predict

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def predict(
    model_path_or_repo: str = typer.Argument(
        ..., help="Path to the model weights or HF repo."
    ),
    data_path: str = typer.Argument(..., help="Path to the dataset directory."),
    min_score: float = typer.Option(
        0.8, help="Minimum score threshold for predictions."
    ),
    dataset: str = "FDDB",
    plot_centers: bool = typer.Option(False, help="Whether to plot ellipse centers."),
    use_tta: bool = typer.Option(True, help="Whether to use Test Time Augmentation."),
) -> None:
    """
    Load a pretrained model, predict ellipses on the given dataset using TTA, and visualize results.
    """
    # Load dataset based on the specified type
    match dataset:
        case "FDDB": ds = FDDB(data_path)
        case "Craters": ds = CraterEllipseDataset(data_path, group="validation")
        case _: raise ValueError(f"Unknown dataset: {dataset}")

    # Load the pretrained model
    typer.echo(f"Loading model from {model_path_or_repo}...")
    model = EllipseRCNN.from_pretrained(model_path_or_repo)
    model.eval().cpu()
    device = torch.device("cpu")

    typer.echo(f"Using {'Test Time Augmentation (TTA)' if use_tta else 'standard prediction'}")
    typer.echo(f"Minimum score threshold: {min_score}")

    # Configure plot styling
    sns.set_theme(style="whitegrid")
    # Create a 6x6 grid of subplots for visualization
    fig, axs = plt.subplots(6, 6, figsize=(16, 16))
    axs = axs.flatten()

    # Randomly sample images from the dataset
    indices = random.sample(range(len(ds)), len(axs))
    
    # Process each sampled image with progress bar
    for ax, idx in tqdm(zip(axs, indices), total=len(indices), desc="Processing images", unit="img"):
        # Load image and ground truth annotations
        image_tensor, target = ds[idx]
        
        if use_tta:
            # Apply Test Time Augmentation using the tta_predict function
            tta_predictions = tta_predict(
                model=model,
                image_tensor=image_tensor,
                device=device,
                min_score=min_score,
                consensuate=True,  # Use consensuation for better results
                visualize=False    # Don't show individual TTA visualizations for batch processing
            )
            
            # Extract the consensuated predictions
            prediction = tta_predictions[0]
        else:
            # Standard prediction without TTA
            with torch.no_grad():
                standard_pred = model([image_tensor])
            prediction = standard_pred[0]
        
        # Apply score threshold filtering
        if prediction["ellipse_params"].numel() > 0:
            score_mask = prediction["scores"] > min_score
            ellipses = prediction["ellipse_params"][score_mask].view(-1, 5)
            boxes = prediction["boxes"][score_mask].view(-1, 4)
        else:
            ellipses = torch.empty(0, 5)
            boxes = torch.empty(0, 4)

        # Configure subplot appearance
        ax.set_aspect("equal"), ax.axis("off")
        # Display the original image
        image_tensor = image_tensor.permute(1, 2, 0) if image_tensor.ndim == 3 else image_tensor
        ax.imshow(image_tensor, cmap="grey")
        # Plot ground truth ellipses in blue
        plot_ellipses(target["ellipse_params"], ax=ax, rim_color="b")
        # Plot consensus predicted ellipse and box in red
        plot_ellipses(ellipses, ax=ax, plot_centers=plot_centers, alpha=0.7)
        # Plot bounding boxes
        plot_bboxes(target["boxes"],box_type="xyxy",ax=ax,rim_color="b",alpha=1)
        plot_bboxes(boxes, box_type="xyxy", ax=ax)

    # Adjust layout and display the plot
    plt.suptitle(f"Ellipse Predictions {'with TTA' if use_tta else 'without TTA'} (min_score={min_score})", 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()