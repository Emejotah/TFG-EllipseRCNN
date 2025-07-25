import torchvision.transforms as T
from PIL import Image
import torch


def get_tta_augmented_images(image: Image.Image) -> list[torch.Tensor]:
    """
    Generates a list of augmented images for Test Time Augmentation (TTA).

    Args:
        image: The input PIL Image to be augmented.

    Returns:
        A tuple containing:
        - A list of torch.Tensor, where each is an augmented version of the image.

    """

    # Base transformation: Convert to tensor and normalize
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    tta_transforms = []

   # 1. Original (Identity)
    tta_transforms.append(T.Compose([T.ToTensor(), normalize]))

    # 2. Horizontal Flip
    tta_transforms.append(
        T.Compose([T.RandomHorizontalFlip(p=1.0), T.ToTensor(), normalize])
    )

    # 3. Small Rotations
    for angle in [-10, 10]:
        tta_transforms.append(
            T.Compose(
                [
                    T.RandomRotation(degrees=(angle, angle), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    normalize,
                ]
            )
        )

    # 4. Brightness/Contrast Adjustments
    tta_transforms.append(
        T.Compose(
            [
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0),
                T.ToTensor(),
                normalize,
            ]
        )
    )

    augmented_images = [transform(image) for transform in tta_transforms]

    return augmented_images



def transform_predictions_back(predictions, transform_info, original_size):
    """
    Transforms predicted ellipses and boxes back to the original image coordinates.

    Args:
        predictions (dict): Dictionary containing prediction tensors with keys:
            - "ellipse_params": torch.Tensor of shape (N, 5) representing ellipse parameters
            - "boxes": torch.Tensor of shape (N, 4) representing bounding boxes
        transform_info (dict): Dictionary describing the applied transform, must contain key "type" (str),
            and for rotations, an "angle" (int or float) in degrees.
        original_size (tuple): (width, height) of the original image.

    Returns:
        tuple: (ellipses, boxes) transformed back to original coordinates.
    """
    w, h = original_size
    ellipses = predictions["ellipse_params"].clone()
    boxes = predictions["boxes"].clone()
    transform_type = transform_info["type"]

    if transform_type == "identity":
        return ellipses, boxes

    if transform_type == "hflip":
        # Flip ellipse center x-coordinate: new_x = width - old_x
        ellipses[:, 0] = w - ellipses[:, 0]
        # Flip ellipse angle: adjust for horizontal reflection
        ellipses[:, 4] = -ellipses[:, 4] % (2 * torch.pi)
        # Flip bounding box coordinates
        x1, _, x2, _ = boxes.T
        boxes[:, 0] = w - x2
        boxes[:, 2] = w - x1
        return ellipses, boxes

    if transform_type == "rotation":
        angle_deg = transform_info["angle"]
        # Convert to radians and negate to reverse the rotation
        angle_rad = -torch.deg2rad(torch.tensor(angle_deg, dtype=torch.float32))
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        rot_matrix = torch.tensor([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # Rotate ellipse centers around image center
        center_coords = ellipses[:, :2] - torch.tensor([w / 2, h / 2])
        ellipses[:, :2] = (rot_matrix @ center_coords.T).T + torch.tensor([w / 2, h / 2])
        # Adjust ellipse angle by subtracting the rotation angle
        ellipses[:, 4] = (ellipses[:, 4] - angle_rad) % (2 * torch.pi)
        
        # Transform bounding box corners and recompute box
        if len(boxes) > 0:
            x1, y1, x2, y2 = boxes.T
            corners = torch.stack([
                torch.stack([x1, y1], dim=1), torch.stack([x2, y1], dim=1),
                torch.stack([x1, y2], dim=1), torch.stack([x2, y2], dim=1)
            ], dim=0)  # Shape: (4, N, 2)
            
            # Rotate each corner around image center
            for i in range(4):
                corner_coords = corners[i] - torch.tensor([w / 2, h / 2])
                corners[i] = (rot_matrix @ corner_coords.T).T + torch.tensor([w / 2, h / 2])
            
            # Find new bounding box from rotated corners
            all_x = corners[:, :, 0]  # Shape: (4, N)
            all_y = corners[:, :, 1]  # Shape: (4, N)
            boxes = torch.stack([
                torch.min(all_x, dim=0)[0],  # x1
                torch.min(all_y, dim=0)[0],  # y1
                torch.max(all_x, dim=0)[0],  # x2
                torch.max(all_y, dim=0)[0]   # y2
            ], dim=1)
        
        return ellipses, boxes
        
    if transform_type == "colorjitter":
        return ellipses, boxes

    return ellipses, boxes