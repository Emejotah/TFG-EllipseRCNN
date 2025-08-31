#!/usr/bin/env python3
"""
Quick script to extract average pixel errors from TTA analysis results.
"""

import json
import numpy as np

def main():
    # Load the analysis results
    with open('tta_analysis_results\\tta_analysis_results.json', 'r') as f:
        data = json.load(f)

    print('=== AVERAGE PIXEL ERRORS FOR EACH TRANSFORMATION ===')
    print('(Based on {} images analyzed)\n'.format(data['metadata']['actual_images_analyzed']))

    # Get per_transform_errors
    per_transform_errors = data.get('error_analysis', {}).get('per_transform_errors', {})

    if per_transform_errors:
        # Calculate averages for each transformation
        results = []
        for transform_name, errors in per_transform_errors.items():
            center_errors = errors.get('center_errors', [])
            if center_errors:
                avg_center_error = np.mean(center_errors)
                results.append((transform_name, avg_center_error))
            else:
                results.append((transform_name, float('inf')))  # No valid predictions
        
        # Sort by average error (best to worst)
        results.sort(key=lambda x: x[1])
        
        print("Transformation           | Avg Pixel Error")
        print("-" * 45)
        
        for transform_name, avg_error in results:
            if avg_error == float('inf'):
                print(f"{transform_name:<20} | No valid predictions")
            else:
                print(f"{transform_name:<20} | {avg_error:.2f} pixels")
                
        print(f"\nBest performing (lowest error): {results[0][0]} ({results[0][1]:.2f} pixels)")
        print(f"Worst performing (highest error): {results[-2][0]} ({results[-2][1]:.2f} pixels)")  # Skip inf values
        
    else:
        print('No per-transform error data found in file')

if __name__ == '__main__':
    main()
