import torch
import typer
from matplotlib import pyplot as plt
import seaborn as sns
import random
import torchvision.transforms as T
from torchvision.ops import nms

from ellipse_rcnn.data.craters import CraterEllipseDataset
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.utils.viz import plot_ellipses, plot_bboxes
from ellipse_rcnn.tta_transforms import get_tta_augmented_images, transform_predictions_back

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def predict(
    model_path_or_repo: str = typer.Argument(
        ..., help="Path to the model weights or HF repo."
    ),
    data_path: str = typer.Argument(..., help="Path to the dataset directory."),
    min_score: float = typer.Option(
        0.6, help="Minimum score threshold for predictions."
    ),
    dataset: str = "FDDB",
    plot_centers: bool = typer.Option(False, help="Whether to plot ellipse centers."),
) -> None:
    """
    Load a pretrained model, predict ellipses on the given dataset using TTA, and visualize results.
    """
    # Load dataset based on the specified type
    match dataset:
        case "FDDB": ds = FDDB(data_path)
        case "Craters": ds = CraterEllipseDataset(data_path, group="validation")
        case _: raise ValueError(f"Unknown dataset: {dataset}")

    # Set up device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load pretrained model and set to evaluation mode
    model = EllipseRCNN.from_pretrained(model_path_or_repo).eval().to(device)

    # Configure plot styling
    sns.set_theme(style="whitegrid")
    # Create a 6x6 grid of subplots for visualization
    fig, axs = plt.subplots(6, 6, figsize=(16, 16))
    axs = axs.flatten()

    # Randomly sample images from the dataset
    indices = random.sample(range(len(ds)), len(axs))
    
    # Process each sampled image
    for ax, idx in zip(axs, indices):
        # Load image and ground truth annotations
        image_tensor, target = ds[idx]
        # Convert tensor to PIL image for TTA processing
        image_pil = T.ToPILImage()(image_tensor.cpu())
        original_size = image_pil.size

        # Apply Test Time Augmentation - generate multiple augmented versions
        augmented_tensors, transforms_info = get_tta_augmented_images(image_pil)
        all_tta_predictions = []

        # Run inference on all augmented images
        with torch.no_grad():
            for aug_img_tensor, transform_info in zip(augmented_tensors, transforms_info):
                # Add batch dimension and move to device
                aug_img_batch = aug_img_tensor.unsqueeze(0).to(device)
                # Get model predictions
                prediction = model(aug_img_batch)[0]
                # Move predictions back to CPU
                prediction = {k: v.cpu() for k, v in prediction.items()}
                # Store prediction with its corresponding transform info
                all_tta_predictions.append((prediction, transform_info))

        # Aggregate predictions from all TTA augmentations
        final_ellipses, final_boxes, final_scores = [], [], []
        for pred, transform_info in all_tta_predictions:
            # Only process predictions that have detections
            if len(pred["scores"]) > 0:
                # Transform predictions back to original image coordinates
                inv_ellipses, inv_boxes = transform_predictions_back(pred, transform_info, original_size)
                final_ellipses.append(inv_ellipses)
                final_boxes.append(inv_boxes)
                final_scores.append(pred["scores"])
        
        # Handle case where no predictions were made
        if not final_boxes:
            pred_agg = {"ellipse_params": torch.empty(0, 5), "boxes": torch.empty(0, 4)}
        else:
            final_ellipses = torch.cat(final_ellipses, dim=0)
            final_boxes = torch.cat(final_boxes, dim=0)
            final_scores = torch.cat(final_scores, dim=0)
            
            # Group predictions by proximity to handle multiple faces
            # Use simple distance-based clustering on ellipse centers
            if len(final_ellipses) > 0:
                centers = final_ellipses[:, :2]  # Extract x, y coordinates
                distance_threshold = 50.0  # Pixels - adjust based on your data
                
                # Simple clustering: group ellipses within distance threshold
                clusters = []
                used = torch.zeros(len(centers), dtype=torch.bool)
                
                for i in range(len(centers)):
                    if used[i]:
                        continue
                    
                    # Start new cluster with current point
                    cluster_indices = [i]
                    used[i] = True
                    
                    # Find all unused points within threshold distance
                    distances = torch.norm(centers - centers[i], dim=1)
                    nearby_mask = (distances < distance_threshold) & (~used)
                    nearby_indices = torch.where(nearby_mask)[0]
                    
                    # Add nearby points to cluster and mark as used
                    for idx in nearby_indices:
                        cluster_indices.append(idx.item())
                        used[idx] = True
                    
                    clusters.append(cluster_indices)
                
                # Compute consensus for each cluster
                consensus_ellipses = []
                consensus_boxes = []
                
                for cluster in clusters:
                    cluster_ellipses = final_ellipses[cluster]
                    cluster_boxes = final_boxes[cluster]
                    cluster_scores = final_scores[cluster]
                    
                    # Alternative aggregation methods for each cluster:
                    # - Mean: torch.mean(cluster_ellipses, dim=0)
                    # - Median: torch.median(cluster_ellipses, dim=0)[0]
                    # - Mode: torch.mode(cluster_ellipses, dim=0)[0] (for discrete values)
                    # - Geometric mean: torch.exp(torch.mean(torch.log(torch.abs(cluster_ellipses) + 1e-8), dim=0))
                    consensus_ellipse = torch.sum(cluster_ellipses * cluster_scores.unsqueeze(1), dim=0) / torch.sum(cluster_scores)
                    consensus_box = torch.sum(cluster_boxes * cluster_scores.unsqueeze(1), dim=0) / torch.sum(cluster_scores)
                    
                    consensus_ellipses.append(consensus_ellipse)
                    consensus_boxes.append(consensus_box)
                
                pred_agg = {
                    "ellipse_params": torch.stack(consensus_ellipses) if consensus_ellipses else torch.empty(0, 5),
                    "boxes": torch.stack(consensus_boxes) if consensus_boxes else torch.empty(0, 4),
                }
            else:
                pred_agg = {"ellipse_params": torch.empty(0, 5), "boxes": torch.empty(0, 4)}

        # Configure subplot appearance
        ax.set_aspect("equal"), ax.axis("off")
        # Display the original image
        ax.imshow(image_tensor.permute(1, 2, 0), cmap="gray")
        # Plot ground truth ellipses in blue
        plot_ellipses(target["ellipse_params"], ax=ax, rim_color="b")
        # Plot consensus predicted ellipse and box in red
        plot_ellipses(pred_agg["ellipse_params"], ax=ax, rim_color="r")
        plot_bboxes(pred_agg["boxes"], ax=ax, rim_color="r")

    # Adjust layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app()