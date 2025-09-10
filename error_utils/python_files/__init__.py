"""
Python translations of MATLAB ellipse error utility functions.

This module provides functions for ellipse parameter conversions and error calculations,
translated from the original MATLAB code by Nikolai Chernov.
"""

from .ellipse_conversions import AtoG, GtoA, GtoN
from .ellipse_errors import (
    EllipseAlgebraicError, 
    EllipseNaturalError, 
    EllipseParGErrors
)
from .ellipse_utils import (
    ellipse_params,
    ProjectPointsOntoEllipse,
    PlotEllipseG,
    GenerateRandomTestTrainingEllipse
)
from .distinguishable_colors import distinguishable_colors

__all__ = [
    'AtoG', 'GtoA', 'GtoN',
    'EllipseAlgebraicError', 'EllipseNaturalError', 'EllipseParGErrors',
    'ellipse_params', 'ProjectPointsOntoEllipse', 'PlotEllipseG',
    'GenerateRandomTestTrainingEllipse', 'distinguishable_colors'
]
