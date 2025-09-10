"""
Example usage of the Python ellipse error utility functions.

This script demonstrates how to use the translated MATLAB functions
for ellipse parameter conversions and error calculations.
"""

import numpy as np
import matplotlib.pyplot as plt


from ellipse_conversions import *
from ellipse_errors import *
from ellipse_utils import *
from distinguishable_colors import distinguishable_colors


def example_ellipse_conversions():
    """Example of ellipse parameter conversions."""
    print("=== Ellipse Parameter Conversions ===")
    
    # Define ellipse geometric parameters
    # [Xcenter, Ycenter, a(major), b(minor), AngleOfTilt]
    ParG_true = np.array([50, 30, 25, 15, np.pi/6])
    print(f"True geometric parameters: {ParG_true}")
    
    # Convert to algebraic parameters
    ParA = GtoA(ParG_true, 1)  # code=1 for ellipse
    print(f"Algebraic parameters: {ParA}")
    
    # Convert back to geometric parameters
    ParG_recovered, code = AtoG(ParA)
    print(f"Recovered geometric parameters: {ParG_recovered}")
    print(f"Conic code: {code} (1=ellipse)")
    
    # Convert to natural parameters
    ParN = GtoN(ParG_true)
    print(f"Natural parameters (foci + sum): {ParN}")
    
    print()


def example_error_calculations():
    """Example of ellipse error calculations."""
    print("=== Ellipse Error Calculations ===")
    
    # True ellipse parameters
    ParG_true = np.array([0, 0, 20, 10, np.pi/4])
    ParA_true = GtoA(ParG_true, 1)
    ParN_true = GtoN(ParG_true)
    
    # Estimated ellipse parameters (with some error)
    ParG_est = ParG_true + np.array([1, 0.5, 2, 1, 0.1])
    ParA_est = GtoA(ParG_est, 1)
    ParN_est = GtoN(ParG_est)
    
    # Calculate errors
    alg_error = EllipseAlgebraicError(ParA_true, ParA_est)
    nat_error = EllipseNaturalError(ParN_true, ParN_est)
    geom_errors = EllipseParGErrors(ParG_true, ParG_est)
    
    print(f"Algebraic error: {alg_error:.4f}")
    print(f"Natural error: {nat_error:.4f}")
    print(f"Geometric errors [center, angle, major, minor, area]: {geom_errors}")
    
    print()


def example_point_projection():
    """Example of projecting points onto an ellipse."""
    print("=== Point Projection onto Ellipse ===")
    
    # Define ellipse
    ParG = np.array([0, 0, 30, 20, np.pi/6])
    
    # Generate some random points
    np.random.seed(42)
    points = np.random.randn(10, 2) * 50
    
    # Project points onto ellipse
    proj_points, params = ProjectPointsOntoEllipse(points, ParG)
    
    print(f"Original points shape: {points.shape}")
    print(f"Projected points shape: {proj_points.shape}")
    print(f"Parameter values shape: {params.shape}")
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot ellipse
    PlotEllipseG(ParG, 'blue', 2)
    
    # Plot original points
    plt.scatter(points[:, 0], points[:, 1], c='red', s=50, alpha=0.7, label='Original points')
    
    # Plot projected points
    plt.scatter(proj_points[0, :], proj_points[1, :], c='green', s=50, alpha=0.7, label='Projected points')
    
    # Draw lines connecting original to projected points
    for i in range(len(points)):
        plt.plot([points[i, 0], proj_points[0, i]], 
                [points[i, 1], proj_points[1, i]], 'k--', alpha=0.5)
    
    plt.axis('equal')
    plt.legend()
    plt.title('Point Projection onto Ellipse')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print()


def example_random_ellipse_generation():
    """Example of generating random ellipse with noisy samples."""
    print("=== Random Ellipse Generation ===")
    
    # Generate random ellipse with samples
    TestSamples, TrainingSamples, ParA, ParG, ParN = GenerateRandomTestTrainingEllipse(
        NumTestSamples=100,
        NumTrainSamples=50,
        NoiseLevel=0.05,
        OutlierProbability=0.1,
        OcclusionLevel=0.3,
        scale=100
    )
    
    print(f"Test samples shape: {TestSamples.shape}")
    print(f"Training samples shape: {TrainingSamples.shape}")
    print(f"Generated ellipse parameters (ParG): {ParG}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot true ellipse
    PlotEllipseG(ParG, 'blue', 2)
    
    # Plot test samples (full ellipse)
    plt.scatter(TestSamples[0, :], TestSamples[1, :], c='green', s=20, alpha=0.6, label='Test samples')
    
    # Plot training samples (partial ellipse with noise and outliers)
    plt.scatter(TrainingSamples[0, :], TrainingSamples[1, :], c='red', s=30, alpha=0.8, label='Training samples')
    
    plt.axis('equal')
    plt.legend()
    plt.title('Random Ellipse with Noisy Samples')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print()


def example_distinguishable_colors():
    """Example of generating distinguishable colors."""
    print("=== Distinguishable Colors ===")
    
    # Generate some distinguishable colors
    n_colors = 12
    colors = distinguishable_colors(n_colors, bg='w')
    
    print(f"Generated {n_colors} distinguishable colors")
    
    # Plot the colors
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Show colors as patches
    ax1.imshow(colors.reshape(1, n_colors, 3))
    ax1.set_title('Generated Colors')
    ax1.set_xticks(range(n_colors))
    ax1.set_yticks([])
    
    # Use colors in a plot
    angles = np.linspace(0, 2*np.pi, 100)
    for i, color in enumerate(colors):
        radius = 1 + i * 0.1
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        ax2.plot(x, y, color=color, linewidth=2, label=f'Line {i+1}')
    
    ax2.set_aspect('equal')
    ax2.set_title('Using Distinguishable Colors')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    print()


if __name__ == "__main__":
    print("Python Ellipse Error Utilities - Examples")
    print("="*50)
    
    example_ellipse_conversions()
    example_error_calculations()
    example_point_projection()
    example_random_ellipse_generation()
    example_distinguishable_colors()
    
    print("All examples completed successfully!")
