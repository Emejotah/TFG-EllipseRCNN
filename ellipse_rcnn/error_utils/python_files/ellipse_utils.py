"""
Ellipse utility functions.

Translated from MATLAB code by Nikolai Chernov.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

from .ellipse_conversions import GtoA, GtoN


def ellipse_params(u: np.ndarray, show: bool = False) -> Tuple[np.ndarray, float, float, float, int]:
    """
    Get ellipse parameters from algebraic equation.
    
    Args:
        u: Coefficients of algebraic equation [A, B, C, D, E, F]
           representing: A*x^2 + B*x*y + C*y^2 + D*x + E*y + F = 0
        show: If True, plot figure if error
        
    Returns:
        Tuple of (z, a, b, alpha, err) where:
        - z: Ellipse center [x, y]
        - a, b: Semi-major and semi-minor axes
        - alpha: Angle of rotation
        - err: Error flag (0 if success, 1 if not an ellipse)
    """
    err = 0
    
    if u[0] < 0:
        u = -u
    
    A = np.array([[u[0], u[1]/2], [u[1]/2, u[2]]])
    bb = np.array([u[3], u[4]])
    c = u[5]
    
    eigenvals, eigenvecs = np.linalg.eig(A)
    det = eigenvals[0] * eigenvals[1]
    
    if det <= 0:
        err = 1
        if show:
            print("Not an ellipse - determinant <= 0")
        z = np.array([0, 0])
        a = b = 1
        alpha = 0
    else:
        bs = eigenvecs.T @ bb
        alpha = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
        zs = -(2 * np.diag(eigenvals)) @ bs / (2 * eigenvals)
        z = eigenvecs @ zs
        h = -np.dot(bs, zs) / 2 - c
        a = np.sqrt(h / eigenvals[0])
        b = np.sqrt(h / eigenvals[1])
    
    return z, a, b, alpha, err


def ProjectPointsOntoEllipse(XY: np.ndarray, ParG: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project a given set of points onto an ellipse and compute distances.
    
    Args:
        XY: Array of coordinates (n x 2) where XY[i] = [x[i], y[i]]
        ParG: Vector of ellipse parameters [Center_x, Center_y, a, b, Angle]
        
    Returns:
        Tuple of (XYproj, ParameterProj) where:
        - XYproj: Array of projected point coordinates (2 x n)
        - ParameterProj: Array of parameters t such that XYproj = (a*cos(t), b*sin(t))
    """
    Center = ParG[:2]
    a = ParG[2]
    b = ParG[3]
    Angle = ParG[4]
    
    if a <= 0 or b <= 0:
        raise ValueError("Axes of the ellipse must be positive")
    
    if a < b:
        raise ValueError("Major axis of the ellipse cannot be smaller than its minor axis")
    
    n = XY.shape[0]
    XYproj = np.zeros((n, 2))
    
    aa = a**2
    bb = b**2
    D = (a - b) * (a + b)  # "distortion measure"
    
    # Rotation matrix for canonical coordinates
    Q = np.array([[np.cos(Angle), -np.sin(Angle)], 
                  [np.sin(Angle), np.cos(Angle)]])
    
    # Transform points to canonical coordinates
    XY0 = (XY - Center) @ Q
    
    for i in range(n):
        u = abs(XY0[i, 0])
        v = abs(XY0[i, 1])
        T = max(a*u - D, b*v)  # initial value of T variable
        
        if T <= 0 and D <= 0:  # circle (a=b) and point at center
            XYproj[i, 0] = 0
            XYproj[i, 1] = b
            continue
        
        if T <= 0:  # true ellipse (a>b) and point on major axis near center
            XYproj[i, 0] = aa * XY0[i, 0] / D
            XYproj[i, 1] = b * np.sqrt(max(1 - (a*u/D)**2, 0))
            continue
        
        # Main case - Newton's iterations
        iterMax = 100
        
        for iteration in range(iterMax):
            F = (a*u/(T+D))**2 + (b*v/T)**2 - 1
            if F <= 0:  # gone too far, emergency stop
                break
            
            Fder = -2 * ((a*u/(T+D))**2/(T+D) + (b*v/T)**2/T)
            Step = F / Fder
            
            if T == T - Step:  # no progress
                break
            
            T = T - Step
        
        # Compute projection candidates
        xprojx = aa * u / (T + D)
        yprojx = b * np.sqrt(max(1 - (xprojx/a)**2, 0))
        yprojy = bb * v / T
        xprojy = a * np.sqrt(max(1 - (yprojy/b)**2, 0))
        
        # Choose better candidate
        Fx = (xprojx - u)**2 + (yprojx - v)**2
        Fy = (xprojy - u)**2 + (yprojy - v)**2
        
        if Fx < Fy:
            XYproj[i, 0] = xprojx
            XYproj[i, 1] = yprojx
        else:
            XYproj[i, 0] = xprojy
            XYproj[i, 1] = yprojy
        
        # Adjust signs for proper quadrant
        if XY0[i, 0] < 0:
            XYproj[i, 0] = -XYproj[i, 0]
        if XY0[i, 1] < 0:
            XYproj[i, 1] = -XYproj[i, 1]
    
    # Compute parameter values
    ParameterProj = np.arctan2(XYproj[:, 1]/b, XYproj[:, 0]/a)
    
    # Transform back to original coordinate system
    XYproj = XYproj @ Q.T + Center
    
    return XYproj.T, ParameterProj


def PlotEllipseG(ParG: np.ndarray, Color: str = 'b', linesize: float = 1) -> List:
    """
    Plot an ellipse given its geometric parameters.
    
    Args:
        ParG: Geometric parameters [Center_x, Center_y, a, b, Angle]
        Color: Color for plotting
        linesize: Line width
        
    Returns:
        List of plot handles
    """
    Center = ParG[:2]
    a = ParG[2]
    b = ParG[3]
    phi = -ParG[4]
    
    # Generate foci
    CenterToFocusDistance = np.sqrt(a**2 - b**2)
    Focus1 = np.array([-CenterToFocusDistance, 0])
    Focus2 = np.array([CenterToFocusDistance, 0])
    
    RotationMatrix = np.array([[np.cos(-phi), -np.sin(-phi)],
                              [np.sin(-phi), np.cos(-phi)]])
    
    Focus1 = RotationMatrix @ Focus1 + Center
    Focus2 = RotationMatrix @ Focus2 + Center
    
    # Generate ellipse points
    NumPoints = 1000
    angles = np.linspace(0, 2*np.pi, NumPoints)
    x1 = a * np.cos(angles)
    y1 = b * np.sin(angles)
    
    # Rotate points
    points = np.column_stack([x1, y1])
    rotated_points = points @ np.array([[np.cos(phi), -np.sin(phi)],
                                       [np.sin(phi), np.cos(phi)]])
    
    # Shift points
    x1 = rotated_points[:, 0] + Center[0]
    y1 = rotated_points[:, 1] + Center[1]
    
    # Plot ellipse
    handle1 = plt.plot(x1, y1, color=Color, linewidth=linesize)[0]
    
    # Plot foci
    handle2 = plt.plot([Focus1[0], Focus2[0]], [Focus1[1], Focus2[1]], 
                       'o', color=Color, linewidth=linesize)[0]
    
    return [handle1, handle2]


def GenerateRandomTestTrainingEllipse(
    NumTestSamples: int,
    NumTrainSamples: int, 
    NoiseLevel: float,
    OutlierProbability: float,
    OcclusionLevel: Optional[float] = None,
    scale: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a random ellipse and draw some noisy samples from it.
    
    Args:
        NumTestSamples: Number of test samples to generate
        NumTrainSamples: Number of training samples to generate
        NoiseLevel: Standard deviation of Gaussian noise
        OutlierProbability: Probability of outliers
        OcclusionLevel: Level of occlusion (None for random)
        scale: Scale factor for ellipse size
        
    Returns:
        Tuple of (TestSamples, TrainingSamples, ParA, ParG, ParN)
    """
    # Choose center in scaled unit square
    Center = scale * np.random.rand(2)
    
    # Choose major and minor axes
    a = scale * (0.2 + 0.8 * np.random.rand())
    b = scale * (0.1 + 0.9 * np.random.rand())
    
    # Ensure major axis is bigger than minor axis
    if a < b:
        a, b = b, a
    
    # Choose tilt angle
    phi = np.pi * np.random.rand() - np.pi/2
    
    # Choose arc parameters
    done = False
    while not done:
        StartingParameter = 2*np.pi*np.random.rand() - np.pi
        if OcclusionLevel is None:  # Random occlusion
            EndingParameter = 2*np.pi*np.random.rand() - np.pi
        else:
            EndingParameter = StartingParameter + 2*np.pi*(1 - OcclusionLevel)
        
        if StartingParameter > EndingParameter:
            StartingParameter, EndingParameter = EndingParameter, StartingParameter
        
        done = (EndingParameter - StartingParameter) > 1  # Arc length > 1 radian
    
    # Generate training samples in canonical coordinates
    Samples = np.zeros((2, NumTrainSamples))
    for i in range(NumTrainSamples):
        MyAngle = StartingParameter + (EndingParameter - StartingParameter) * np.random.rand()
        Samples[0, i] = a * np.cos(MyAngle)
        Samples[1, i] = b * np.sin(MyAngle)
    
    # Add Gaussian noise
    TrainingSamples = Samples + NoiseLevel * scale * np.random.randn(2, NumTrainSamples)
    
    # Add outliers
    OutlierIndices = np.where(np.random.rand(NumTrainSamples) < OutlierProbability)[0]
    TrainingSamples[:, OutlierIndices] = -1 + 2 * np.random.rand(2, len(OutlierIndices))
    
    # Generate test samples
    TestSamples = np.zeros((2, NumTestSamples))
    for i in range(NumTestSamples):
        MyAngle = -np.pi + 2*np.pi*np.random.rand()
        TestSamples[0, i] = a * np.cos(MyAngle)
        TestSamples[1, i] = b * np.sin(MyAngle)
    
    # Rotation matrix
    RotationMatrix = np.array([[np.cos(phi), -np.sin(phi)],
                              [np.sin(phi), np.cos(phi)]])
    
    # Rotate and shift samples
    TrainingSamples = RotationMatrix @ TrainingSamples + Center.reshape(-1, 1)
    TestSamples = RotationMatrix @ TestSamples + Center.reshape(-1, 1)
    
    # Collect parameters
    ParG = np.array([Center[0], Center[1], a, b, phi])
    ParA = GtoA(ParG, 1)
    ParN = GtoN(ParG)
    
    return TestSamples, TrainingSamples, ParA, ParG, ParN
