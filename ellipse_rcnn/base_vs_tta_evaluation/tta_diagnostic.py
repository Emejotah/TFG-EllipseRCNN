"""
Diagnostic script to understand why TTA is detecting more ellipses than expected.
This will trace through the TTA process step by step.
"""

import torch
from ellipse_rcnn.hf import EllipseRCNN
from ellipse_rcnn.data.fddb import FDDB
from ellipse_rcnn.tta.tta_transforms import tta_predict, TTA_CONFIG, TTA_TRANSFORMS

def diagnose_tta_over_detection():
    """Diagnose why TTA is detecting more ellipses than the base model."""
    
    print("🔍 TTA Over-Detection Diagnostic")
    print("="*60)
    
    # Load model and data
    model = EllipseRCNN.from_pretrained("MJGT/ellipse-rcnn-FDDB")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    
    dataset = FDDB("../../data/FDDB")
    
    # Test on a few images
    num_test_images = 5
    min_score = 0.5  # Same as in evaluation
    
    print(f"Testing on {num_test_images} images with min_score={min_score}")
    print(f"TTA Config: {TTA_CONFIG}")
    print(f"Number of TTA transforms: {len(TTA_TRANSFORMS)}")
    print()
    
    total_base_detections = 0
    total_tta_detections = 0
    total_targets = 0
    
    for idx in range(min(num_test_images, len(dataset))):
        image_tensor, target = dataset[idx]
        image_tensor = image_tensor.to(device)
        target_ellipses = target["ellipse_params"]
        
        if target_ellipses.numel() == 0:
            continue
            
        num_targets = len(target_ellipses)
        total_targets += num_targets
        
        print(f"\n--- Image {idx+1} ---")
        print(f"Ground truth ellipses: {num_targets}")
        
        # Base model prediction
        with torch.no_grad():
            base_pred = model([image_tensor])
            base_prediction = base_pred[0]
        
        # Filter by score
        if base_prediction["ellipse_params"].numel() > 0:
            base_score_mask = base_prediction["scores"] > min_score
            base_ellipses = base_prediction["ellipse_params"][base_score_mask]
            base_scores = base_prediction["scores"][base_score_mask]
        else:
            base_ellipses = torch.empty(0, 5)
            base_scores = torch.empty(0)
        
        num_base = len(base_ellipses)
        total_base_detections += num_base
        print(f"Base model detections: {num_base}")
        if num_base > 0:
            print(f"  Base scores: {base_scores.cpu().numpy()}")
        
        # TTA prediction WITHOUT consensuation first
        print(f"\n  TTA Analysis (individual transforms):")
        
        # Get raw TTA predictions from each transform
        all_transform_predictions = []
        transform_details = []
        
        for i, transform in enumerate(TTA_TRANSFORMS):
            # Apply forward transform
            augmented_img = transform['forward'][0](image_tensor, **transform['forward'][1])
            
            # Get predictions on augmented image
            with torch.no_grad():
                aug_pred = model([augmented_img])
                aug_prediction = aug_pred[0]
            
            # Reverse transform predictions
            if aug_prediction["ellipse_params"].numel() > 0:
                # Apply reverse transform to predictions
                reversed_pred = {
                    'boxes': transform['reverse'][0](aug_prediction['boxes'], image_tensor.shape[-2:], **transform['reverse'][1]),
                    'ellipse_params': transform['reverse'][0](aug_prediction['ellipse_params'], image_tensor.shape[-2:], **transform['reverse'][1]),
                    'labels': aug_prediction['labels'],
                    'scores': aug_prediction['scores']
                }
                
                # Filter by score
                score_mask = reversed_pred["scores"] > min_score
                filtered_ellipses = reversed_pred["ellipse_params"][score_mask]
                filtered_scores = reversed_pred["scores"][score_mask]
            else:
                filtered_ellipses = torch.empty(0, 5)
                filtered_scores = torch.empty(0)
            
            num_transform_detections = len(filtered_ellipses)
            all_transform_predictions.append({
                'ellipse_params': filtered_ellipses,
                'scores': filtered_scores,
                'transform_name': transform['name']
            })
            
            print(f"    {transform['name']}: {num_transform_detections} detections")
            if num_transform_detections > 0:
                print(f"      Scores: {filtered_scores.cpu().numpy()}")
        
        # Total individual predictions (before consensuation)
        total_individual = sum(len(pred['ellipse_params']) for pred in all_transform_predictions)
        print(f"  Total individual predictions: {total_individual}")
        
        # Now TTA with consensuation
        tta_predictions = tta_predict(
            model=model,
            image_tensor=image_tensor,
            device=device,
            min_score=min_score,
            consensuate=True,
            visualize=False,
            consensuation_method='quality'
        )
        tta_prediction = tta_predictions[0]
        
        # Filter TTA predictions by score
        if tta_prediction["ellipse_params"].numel() > 0:
            tta_score_mask = tta_prediction["scores"] > min_score
            tta_ellipses = tta_prediction["ellipse_params"][tta_score_mask]
            tta_scores = tta_prediction["scores"][tta_score_mask]
        else:
            tta_ellipses = torch.empty(0, 5)
            tta_scores = torch.empty(0)
        
        num_tta = len(tta_ellipses)
        total_tta_detections += num_tta
        print(f"TTA consensuated detections: {num_tta}")
        if num_tta > 0:
            print(f"  TTA scores: {tta_scores.cpu().numpy()}")
        
        # Analysis
        print(f"\n  Summary for Image {idx+1}:")
        print(f"    Targets: {num_targets}")
        print(f"    Base: {num_base} ({'more' if num_base > num_targets else 'fewer' if num_base < num_targets else 'same'})")
        print(f"    TTA: {num_tta} ({'more' if num_tta > num_targets else 'fewer' if num_tta < num_targets else 'same'})")
        print(f"    Individual transforms total: {total_individual}")
        print(f"    Consensuation effect: {total_individual} → {num_tta} ({((num_tta - total_individual) / max(total_individual, 1)) * 100:+.1f}%)")
    
    # Final summary
    print(f"\n" + "="*60)
    print(f"DIAGNOSTIC SUMMARY ({num_test_images} images)")
    print(f"="*60)
    print(f"Total targets: {total_targets}")
    print(f"Total base detections: {total_base_detections}")
    print(f"Total TTA detections: {total_tta_detections}")
    print(f"")
    print(f"Average per image:")
    print(f"  Targets: {total_targets / num_test_images:.2f}")
    print(f"  Base: {total_base_detections / num_test_images:.2f}")
    print(f"  TTA: {total_tta_detections / num_test_images:.2f}")
    print(f"")
    print(f"TTA vs Base: {((total_tta_detections - total_base_detections) / max(total_base_detections, 1)) * 100:+.1f}%")
    print(f"TTA vs Targets: {((total_tta_detections - total_targets) / max(total_targets, 1)) * 100:+.1f}%")
    
    # Potential issues to investigate
    print(f"\n🔍 POTENTIAL ISSUES TO INVESTIGATE:")
    if total_tta_detections > total_targets:
        print(f"1. ✓ TTA is over-detecting ({total_tta_detections} vs {total_targets} targets)")
    if total_tta_detections > total_base_detections:
        print(f"2. ✓ TTA detects more than base model ({total_tta_detections} vs {total_base_detections})")
    
    print(f"3. Check if consensuation is working correctly")
    print(f"4. Check if NMS is being applied properly")
    print(f"5. Check if score thresholds are consistent")
    print(f"6. Check for duplicate detections from different transforms")

if __name__ == "__main__":
    diagnose_tta_over_detection()
