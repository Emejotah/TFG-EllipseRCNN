"""
Generate distinguishable colors for plotting.

Translated from MATLAB code by Timothy E. Holy.
"""

import numpy as np
from typing import List, Union, Optional, Callable


def distinguishable_colors(
    n_colors: int, 
    bg: Optional[Union[List, np.ndarray, str]] = None, 
    func: Optional[Callable] = None
) -> np.ndarray:
    """
    Pick colors that are maximally perceptually distinct.
    
    When plotting a set of lines, you may want to distinguish them by color.
    This function generates a set of colors which are distinguishable
    by reference to the "Lab" color space, which more closely matches
    human color perception than RGB.
    
    Args:
        n_colors: Number of colors to generate
        bg: Background color(s) to avoid. Can be:
            - String color specification ('w', 'k', etc.)
            - RGB array (n x 3)
            - List of color specifications
        func: Optional color conversion function (RGB to Lab)
        
    Returns:
        colors: n_colors x 3 RGB array of distinguishable colors
    """
    # Parse background colors
    if bg is None:
        bg = np.array([[1, 1, 1]])  # default white background
    elif isinstance(bg, (list, tuple)):
        # User specified a list of colors
        bgc = []
        for color in bg:
            bgc.append(parsecolor(color))
        bg = np.vstack(bgc)
    else:
        # User specified a numeric array or single color
        bg = parsecolor(bg)
        if bg.ndim == 1:
            bg = bg.reshape(1, -1)
    
    # Generate RGB space grid
    n_grid = 30  # number of grid divisions along each axis
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x, indexing='ij')
    rgb = np.column_stack([R.ravel(), G.ravel(), B.ravel()])
    
    if n_colors > rgb.shape[0] // 3:
        raise ValueError("You can't readily distinguish that many colors")
    
    # Convert to Lab color space
    if func is not None:
        lab = func(rgb)
        bglab = func(bg)
    else:
        # Simple RGB to Lab approximation (for systems without color conversion)
        lab = rgb_to_lab_simple(rgb)
        bglab = rgb_to_lab_simple(bg)
    
    # Compute distances from candidate colors to background colors
    mindist2 = np.full(rgb.shape[0], np.inf)
    for i in range(bglab.shape[0]):
        dX = lab - bglab[i:i+1, :]  # displacement from bg
        dist2 = np.sum(dX**2, axis=1)  # square distance
        mindist2 = np.minimum(dist2, mindist2)
    
    # Iteratively pick colors that maximize distance to nearest chosen color
    colors = np.zeros((n_colors, 3))
    lastlab = bglab[-1, :]  # initialize with last background color
    
    for i in range(n_colors):
        dX = lab - lastlab.reshape(1, -1)  # displacement from last color
        dist2 = np.sum(dX**2, axis=1)  # square distance
        mindist2 = np.minimum(dist2, mindist2)  # update minimum distances
        index = np.argmax(mindist2)  # find farthest color
        colors[i, :] = rgb[index, :]  # save color
        lastlab = lab[index, :]  # prepare for next iteration
    
    return colors


def parsecolor(s: Union[str, np.ndarray]) -> np.ndarray:
    """Parse color specification to RGB array."""
    if isinstance(s, str):
        return colorstr2rgb(s)
    elif isinstance(s, np.ndarray) and s.shape[-1] == 3:
        return s
    else:
        raise ValueError('Color specification cannot be parsed.')


def colorstr2rgb(c: str) -> np.ndarray:
    """Convert a color string to an RGB value."""
    rgbspec = np.array([
        [1, 0, 0],  # r
        [0, 1, 0],  # g  
        [0, 0, 1],  # b
        [1, 1, 1],  # w
        [0, 1, 1],  # c
        [1, 0, 1],  # m
        [1, 1, 0],  # y
        [0, 0, 0]   # k
    ])
    cspec = 'rgbwcmyk'
    
    try:
        k = cspec.index(c[0].lower())
    except (ValueError, IndexError):
        raise ValueError('Unknown color string.')
    
    if k != 3 or len(c) == 1:  # not 'b' or single character
        return rgbspec[k, :]
    elif len(c) > 2:
        if c[:3].lower() == 'bla':
            return np.array([0, 0, 0])
        elif c[:3].lower() == 'blu':
            return np.array([0, 0, 1])
        else:
            raise ValueError('Unknown color string.')
    else:
        return rgbspec[k, :]


def rgb_to_lab_simple(rgb: np.ndarray) -> np.ndarray:
    """
    Simple RGB to Lab conversion approximation.
    
    This is a simplified version for systems without proper color conversion.
    For better results, use a proper color conversion library.
    """
    # Simple transformation that preserves perceptual distances better than RGB
    # This is not true Lab space but provides better color separation
    
    # Apply gamma correction
    rgb_linear = np.where(rgb > 0.04045, 
                         ((rgb + 0.055) / 1.055) ** 2.4,
                         rgb / 12.92)
    
    # Convert to XYZ (simplified)
    xyz = rgb_linear @ np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722], 
        [0.0193, 0.1192, 0.9505]
    ]).T
    
    # Normalize by D65 white point
    xyz /= np.array([0.95047, 1.0, 1.08883])
    
    # Convert to Lab (simplified)
    xyz = np.where(xyz > 0.008856, xyz**(1/3), (7.787 * xyz + 16/116))
    
    L = 116 * xyz[:, 1] - 16
    a = 500 * (xyz[:, 0] - xyz[:, 1])
    b = 200 * (xyz[:, 1] - xyz[:, 2])
    
    return np.column_stack([L, a, b])
