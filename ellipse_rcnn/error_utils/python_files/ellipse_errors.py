"""
Ellipse error calculation functions.

Translated from MATLAB code by Nikolai Chernov.
"""

import numpy as np
from typing import List


def EllipseAlgebraicError(TrueParA: np.ndarray, ParA: np.ndarray) -> float:
    """
    Compute the error in the estimation of an ellipse using algebraic parameters.
    
    Args:
        TrueParA: True algebraic parameters [A,B,C,D,E,F]
        ParA: Estimated algebraic parameters [A,B,C,D,E,F]
        
    Returns:
        Error: Algebraic error between true and estimated parameters
    """
    # Parameter normalization
    TrueParA = TrueParA / np.linalg.norm(TrueParA)
    ParA = ParA / np.linalg.norm(ParA)
    
    # Error computation
    Error = np.linalg.norm(TrueParA - ParA)
    
    return Error


def EllipseNaturalError(TrueParN: np.ndarray, ParN: np.ndarray) -> float:
    """
    Compute the error in the estimation of an ellipse using natural parameters.
    
    Args:
        TrueParN: True natural parameters [Focus1x, Focus1y, Focus2x, Focus2y, SumDists]
        ParN: Estimated natural parameters [Focus1x, Focus1y, Focus2x, Focus2y, SumDists]
        
    Returns:
        Error: Natural error between true and estimated parameters
    """
    Error1 = np.linalg.norm(TrueParN - ParN)
    # Try swapping the foci order
    Error2 = np.linalg.norm(TrueParN[[2, 3, 0, 1, 4]] - ParN)
    Error = min(Error1, Error2)
    
    return Error


def EllipseParGErrors(TrueParG: np.ndarray, ParG: np.ndarray) -> List[float]:
    """
    Compute the errors in the estimation of an ellipse using geometric parameters.
    
    Args:
        TrueParG: True geometric parameters [Xcenter, Ycenter, a, b, AngleOfTilt]
        ParG: Estimated geometric parameters [Xcenter, Ycenter, a, b, AngleOfTilt]
        
    Returns:
        Errors: [CenterError, AngleError, MajorSemiaxisError, MinorSemiaxisError, AreaError]
    """
    # Make a copy to avoid modifying the original
    ParG_copy = ParG.copy()
    TrueParG_copy = TrueParG.copy()
    
    CenterError = np.linalg.norm(TrueParG_copy[:2] - ParG_copy[:2])
    
    # Making sure that ParG(3) is major, ParG(4) is minor (for true ellipse is sure)
    if ParG_copy[2] < ParG_copy[3]:
        ParG_copy[2], ParG_copy[3] = ParG_copy[3], ParG_copy[2]
        ParG_copy[4] = ParG_copy[4] - np.pi/2
        if ParG_copy[4] < 0:
            ParG_copy[4] = ParG_copy[4] + np.pi
    
    MajorSemiaxisError = abs(TrueParG_copy[2] - ParG_copy[2])
    MinorSemiaxisError = abs(TrueParG_copy[3] - ParG_copy[3])
    
    # Making sure that both angles are in first and second quadrants
    if TrueParG_copy[4] > np.pi:
        TrueParG_copy[4] = TrueParG_copy[4] - np.pi
    if ParG_copy[4] > np.pi:
        ParG_copy[4] = ParG_copy[4] - np.pi
    
    AngleError = abs(TrueParG_copy[4] - ParG_copy[4])
    
    AreaError = abs(np.pi * (TrueParG_copy[2]*TrueParG_copy[3] - ParG_copy[2]*ParG_copy[3]))
    
    Errors = [CenterError, AngleError, MajorSemiaxisError, MinorSemiaxisError, AreaError]
    
    return Errors
