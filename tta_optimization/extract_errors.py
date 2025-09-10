#!/usr/bin/env python3
"""
Quick script to extract average pixel errors from TTA analysis results.
"""

import json
import numpy as np

def main():
    # Load the analysis results
    with open('tta_analysis_results\\tta_analysis_results.json', 'r') as f: # tta_analysis_results.json
        data = json.load(f)

    print('=== AVERAGE ERRORS FOR EACH TRANSFORMATION ===')
    print('(Based on {} images analyzed)\n'.format(data['metadata']['actual_images_analyzed']))

    # Get per_transform_errors
    per_transform_errors = data.get('error_analysis', {}).get('per_transform_errors', {})

    if per_transform_errors:
        # Calculate averages for each transformation
        results = []
        for transform_name, errors in per_transform_errors.items():
            center_errors = errors.get('center_errors', [])
            angle_errors = errors.get('angle_errors', [])
            area_errors = errors.get('area_errors', [])
            
            if center_errors:
                avg_center_error = np.mean(center_errors)
                avg_angle_error = np.mean(angle_errors) if angle_errors else 0.0
                avg_area_error = np.mean(area_errors) if area_errors else 0.0
                results.append((transform_name, avg_center_error, avg_angle_error, avg_area_error))
            else:
                results.append((transform_name, float('inf'), float('inf'), float('inf')))  # No valid predictions
        
        # Sort by average center error (best to worst)
        results.sort(key=lambda x: x[1])
        
        print("Transformation           | Center Error | Angle Error | Area Error")
        print("                         |   (pixels)   |  (degrees)  |  (ratio)")
        print("-" * 75)
        
        for transform_name, avg_center, avg_angle, avg_area in results:
            if avg_center == float('inf'):
                print(f"{transform_name:<24} | No valid predictions")
            else:
                print(f"{transform_name:<24} | {avg_center:>10.2f} | {avg_angle:>9.2f} | {avg_area:>8.3f}")
                
        # Find best performing for each error type
        valid_results = [(name, center, angle, area) for name, center, angle, area in results 
                        if center != float('inf')]
        
        if valid_results:
            best_center = min(valid_results, key=lambda x: x[1])
            best_angle = min(valid_results, key=lambda x: x[2])
            best_area = min(valid_results, key=lambda x: x[3])
            
            print(f"\n=== BEST PERFORMING TRANSFORMATIONS ===")
            print(f"Lowest Center Error:  {best_center[0]} ({best_center[1]:.2f} pixels)")
            print(f"Lowest Angle Error:   {best_angle[0]} ({best_angle[2]:.2f} degrees)")
            print(f"Lowest Area Error:    {best_area[0]} ({best_area[3]:.3f} ratio)")
        
    else:
        print('No per-transform error data found in file')

if __name__ == '__main__':
    main()
