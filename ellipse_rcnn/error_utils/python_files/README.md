# Python Ellipse Error Utilities

This folder contains Python translations of MATLAB ellipse error utility functions originally written by Nikolai Chernov and Timothy E. Holy.

## Original MATLAB Source
The original MATLAB code can be found at: http://people.cas.uab.edu/~mosya/cl/MATLABconics.html

## Files

### Core Modules

1. **`ellipse_conversions.py`** - Ellipse parameter conversion functions
   - `AtoG()` - Convert Algebraic parameters to Geometric parameters  
   - `GtoA()` - Convert Geometric parameters to Algebraic parameters
   - `GtoN()` - Convert Geometric parameters to Natural parameters

2. **`ellipse_errors.py`** - Ellipse error calculation functions
   - `EllipseAlgebraicError()` - Compute algebraic parameter errors
   - `EllipseNaturalError()` - Compute natural parameter errors  
   - `EllipseParGErrors()` - Compute geometric parameter errors

3. **`ellipse_utils.py`** - Ellipse utility functions
   - `ellipse_params()` - Extract ellipse parameters from algebraic equation
   - `ProjectPointsOntoEllipse()` - Project points onto ellipse surface
   - `PlotEllipseG()` - Plot ellipse from geometric parameters
   - `GenerateRandomTestTrainingEllipse()` - Generate random ellipse with samples

4. **`distinguishable_colors.py`** - Color generation utilities
   - `distinguishable_colors()` - Generate perceptually distinguishable colors

### Example and Documentation

5. **`example_usage.py`** - Complete examples demonstrating all functions
6. **`__init__.py`** - Package initialization with imports

## Parameter Formats

### Algebraic Parameters (ParA)
Vector `[A, B, C, D, E, F]` representing the conic equation:
```
A*x² + 2*B*x*y + C*y² + 2*D*x + 2*E*y + F = 0
```

### Geometric Parameters (ParG)  
Vector `[Xcenter, Ycenter, a, b, AngleOfTilt]` where:
- `Xcenter, Ycenter` - Center coordinates
- `a` - Semi-major axis (≥ b)
- `b` - Semi-minor axis  
- `AngleOfTilt` - Rotation angle in radians

### Natural Parameters (ParN)
Vector `[Focus1x, Focus1y, Focus2x, Focus2y, SumDists]` where:
- `Focus1, Focus2` - Coordinates of the two foci
- `SumDists` - Sum of distances to both foci (= 2*a)

## Usage Examples

```python
import numpy as np
from ellipse_conversions import AtoG, GtoA, GtoN
from ellipse_errors import EllipseParGErrors
from ellipse_utils import PlotEllipseG

# Define an ellipse
ParG = np.array([0, 0, 30, 20, np.pi/6])  # center=(0,0), a=30, b=20, angle=π/6

# Convert to algebraic form
ParA = GtoA(ParG, 1)  # code=1 for ellipse

# Convert back and check
ParG_recovered, code = AtoG(ParA)

# Calculate errors between original and recovered
errors = EllipseParGErrors(ParG, ParG_recovered)
print(f"Errors: {errors}")

# Plot the ellipse
import matplotlib.pyplot as plt
plt.figure()
PlotEllipseG(ParG, 'blue', 2)
plt.axis('equal')
plt.show()
```

## Dependencies

- `numpy` - For numerical computations
- `matplotlib` - For plotting functions
- `typing` - For type hints

## Conic Type Codes

The `AtoG()` function returns a code indicating the conic type:

- 1: Ellipse
- 2: Hyperbola  
- 3: Parabola
- 4: Intersecting lines
- 5: Parallel lines
- 6: Coincident lines
- 7: Single line
- 8: Single point
- 9: Imaginary ellipse
- 10: Imaginary parallel lines
- 11: "Impossible" equation

## Notes

- All functions use NumPy arrays for vector/matrix operations
- Angles are in radians
- The code assumes ellipses have `a ≥ b` (major axis ≥ minor axis)
- Error calculations handle the ambiguity in ellipse orientation by testing both possible focus orderings

## Testing

Run the example file to test all functions:

```python
python example_usage.py
```

This will demonstrate ellipse parameter conversions, error calculations, point projections, and color generation.
