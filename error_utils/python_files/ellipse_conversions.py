"""
Ellipse parameter conversion functions.

Translated from MATLAB code by Nikolai Chernov.
http://people.cas.uab.edu/~mosya/cl/MATLABconics.html
"""

import numpy as np
from typing import Tuple, Union


def AtoG(ParA: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Conversion of Algebraic parameters of a conic to its Geometric parameters.
    
    Algebraic parameters are coefficients A,B,C,D,E,F in the algebraic
    equation: Ax^2 + 2Bxy + Cy^2 + 2Dx + 2Ey + F = 0
    
    Args:
        ParA: Vector of Algebraic parameters [A,B,C,D,E,F]
        
    Returns:
        Tuple of (ParG, code) where:
        - ParG: Vector of Geometric parameters (see below)
        - code: Code of the conic type (see below)
        
    Codes:
        1 - ellipse, 2 - hyperbola, 3 - parabola
        4 - intersecting lines, 5 - parallel lines
        6 - coincident lines, 7 - single line
        8 - single point, 9 - imaginary ellipse
        10 - imaginary parallel lines
        11 - "impossible" equation, 1=0 or -1=0 (no solutions)
        
    Geometric parameters for ellipses (code=1):
        ParG = [Xcenter, Ycenter, a, b, AngleOfTilt]
        where a >= b are the major and minor semi-axes
    """
    ParA = ParA / np.linalg.norm(ParA)  # normalize
    
    ParG = -1  # default return for imaginary or degenerate conics
    
    # Check for no quadratic part
    if (np.abs(ParA[0]) < 1e-14 and np.abs(ParA[1]) < 1e-14 and 
        np.abs(ParA[2]) < 1e-14):
        if np.abs(ParA[3]) < 1e-14 and np.abs(ParA[4]) < 1e-14:
            code = 11  # the "pole", extreme singularity
        else:
            code = 7   # single line
        return ParG, code
    
    # Construct matrices
    M33 = np.array([
        [ParA[0], ParA[1], ParA[3]],
        [ParA[1], ParA[2], ParA[4]],
        [ParA[3], ParA[4], ParA[5]]
    ])
    
    M22 = np.array([
        [ParA[0], ParA[1]],
        [ParA[1], ParA[2]]
    ])
    
    det3x3 = np.linalg.det(M33)
    det2x2 = np.linalg.det(M22)
    
    # Check if big matrix is singular
    if np.abs(det3x3) < 1e-14:
        if np.abs(det2x2) < 1e-14:
            dettwo = ParA[0]*ParA[5] - ParA[3]**2 + ParA[2]*ParA[5] - ParA[4]**2
            if dettwo > 0:
                code = 10  # imaginary parallel lines
            elif dettwo < 0:
                code = 5   # parallel lines
            else:
                code = 6   # coincident lines
            return ParG, code
        
        if det2x2 > 0:
            code = 8  # single point
        else:
            code = 4  # intersecting lines
            # Compute intersection lines parameters
            eigenvals, eigenvecs = np.linalg.eig(M33)
            max_idx = np.argmax(eigenvals)
            min_idx = np.argmin(eigenvals)
            
            Qmax = eigenvecs[:, max_idx] * np.sqrt(np.abs(eigenvals[max_idx]))
            Qmin = eigenvecs[:, min_idx] * np.sqrt(np.abs(eigenvals[min_idx]))
            
            Q1 = Qmax + Qmin
            Q2 = Qmax - Qmin
            
            theta1 = np.arctan2(Q1[1], Q1[0])
            d1 = Q1[2] / np.linalg.norm(Q1[:2])
            theta2 = np.arctan2(Q2[1], Q2[0])
            d2 = Q2[2] / np.linalg.norm(Q2[:2])
            
            ParG = np.array([theta1, d1, theta2, d2])
        
        return ParG, code
    
    # Non-degenerate conics
    eigenvals, eigenvecs = np.linalg.eig(M22)
    U = eigenvecs.T @ np.array([ParA[3], ParA[4]])
    
    # Check for parabola
    if np.abs(eigenvals[0]) < 1e-14 or np.abs(eigenvals[1]) < 1e-14:
        code = 3  # parabola
        if np.abs(eigenvals[0]) > np.abs(eigenvals[1]):
            Uc1 = -U[0] / eigenvals[0]
            Uc2 = -(U[0]*Uc1 + ParA[5]) / (2*U[1])
            Center = eigenvecs @ np.array([Uc1, Uc2])
            p = -U[1] / eigenvals[0]
            Angle = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
            ParG = np.array([Center[0], Center[1], p, Angle])
        else:
            Uc2 = -U[1] / eigenvals[1]
            Uc1 = -(U[1]*Uc2 + ParA[5]) / (2*U[0])
            Center = eigenvecs @ np.array([Uc1, Uc2])
            p = -U[0] / eigenvals[1]
            Angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
            ParG = np.array([Center[0], Center[1], p, Angle])
        
        return ParG, code
    
    # Ellipse or hyperbola
    Uc = -U / eigenvals
    Center = eigenvecs @ Uc
    H = -np.dot(U, Uc) - ParA[5]
    
    if eigenvals[0] * eigenvals[1] < 0:
        code = 2  # hyperbola
        if eigenvals[0] * H > 0:
            a = np.sqrt(H / eigenvals[0])
            b = np.sqrt(-H / eigenvals[1])
            Angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
            if Angle < 0:
                Angle += np.pi
        else:
            a = np.sqrt(H / eigenvals[1])
            b = np.sqrt(-H / eigenvals[0])
            Angle = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
            if Angle < 0:
                Angle += np.pi
        ParG = np.array([Center[0], Center[1], a, b, Angle])
    else:
        if H * eigenvals[0] <= 0:
            code = 9  # imaginary ellipse
        else:
            code = 1  # ellipse
            a = np.sqrt(H / eigenvals[0])
            b = np.sqrt(H / eigenvals[1])
            Angle = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])
            if Angle < 0:
                Angle += np.pi
            
            # Ensure a is major, b is minor
            if a < b:
                a, b = b, a
                Angle -= np.pi/2
                if Angle < 0:
                    Angle += np.pi
            
            ParG = np.array([Center[0], Center[1], a, b, Angle])
    
    return ParG, code


def GtoA(ParG: np.ndarray, code: int) -> np.ndarray:
    """
    Conversion of Geometric parameters of a conic to its Algebraic parameters.
    
    Args:
        ParG: Vector of Geometric parameters
        code: Code of the conic type
        
    Returns:
        ParA: Vector of Algebraic parameters [A,B,C,D,E,F]
    """
    if code == 1:  # ellipse
        c = np.cos(ParG[4])
        s = np.sin(ParG[4])
        a = ParG[2]
        b = ParG[3]
        Xc = ParG[0]
        Yc = ParG[1]
        
        P = (c/a)**2 + (s/b)**2
        Q = (s/a)**2 + (c/b)**2
        R = c*s*(1/a**2 - 1/b**2)
        
        ParA = np.array([
            P,
            R,
            Q,
            -P*Xc - R*Yc,
            -Q*Yc - R*Xc,
            P*Xc**2 + Q*Yc**2 + 2*R*Xc*Yc - 1
        ])
        
        ParA = ParA / np.linalg.norm(ParA)
        
    elif code == 2:  # hyperbola
        c = np.cos(ParG[4])
        s = np.sin(ParG[4])
        a = ParG[2]
        b = ParG[3]
        Xc = ParG[0]
        Yc = ParG[1]
        
        P = (c/a)**2 - (s/b)**2
        Q = (s/a)**2 - (c/b)**2
        R = c*s*(1/a**2 + 1/b**2)
        
        ParA = np.array([
            P,
            R,
            Q,
            -P*Xc - R*Yc,
            -Q*Yc - R*Xc,
            P*Xc**2 + Q*Yc**2 + 2*R*Xc*Yc - 1
        ])
        
        ParA = ParA / np.linalg.norm(ParA)
        
    elif code == 3:  # parabola
        c = np.cos(ParG[3])
        s = np.sin(ParG[3])
        p = ParG[2]
        Xc = ParG[0]
        Yc = ParG[1]
        
        R = Xc*s - Yc*c
        
        ParA = np.array([
            s**2,
            -c*s,
            c**2,
            -R*s - p*c,
            R*c - p*s,
            R**2 + 2*p*(Xc*c + Yc*s)
        ])
        
        ParA = ParA / np.linalg.norm(ParA)
        
    elif code == 4:  # intersecting lines
        c1 = np.cos(ParG[0])
        s1 = np.sin(ParG[0])
        c2 = np.cos(ParG[2])
        s2 = np.sin(ParG[2])
        
        ParA = np.array([
            c1*c2,
            (c1*s2 + c2*s1)/2,
            s1*s2,
            (c1*ParG[3] + c2*ParG[1])/2,
            (s1*ParG[3] + s2*ParG[1])/2,
            ParG[1]*ParG[3]
        ])
        
        ParA = ParA / np.linalg.norm(ParA)
    else:
        raise ValueError(f"Unsupported conic code: {code}")
    
    return ParA


def GtoN(ParG: np.ndarray) -> np.ndarray:
    """
    Compute the natural parameters of an ellipse from the geometric parameters.
    
    Args:
        ParG: [Xcenter, Ycenter, a, b, AngleOfTilt] geometric parameters
        
    Returns:
        ParN: [Focus1x, Focus1y, Focus2x, Focus2y, SumDists] natural parameters
    """
    Center = ParG[:2]
    a = ParG[2]
    b = ParG[3]
    phi = -ParG[4]
    
    CenterToFocusDistance = np.sqrt(a**2 - b**2)
    Focus1 = np.array([-CenterToFocusDistance, 0])
    Focus2 = np.array([CenterToFocusDistance, 0])
    
    RotationMatrix = np.array([
        [np.cos(-phi), -np.sin(-phi)],
        [np.sin(-phi), np.cos(-phi)]
    ])
    
    Focus1 = RotationMatrix @ Focus1 + Center
    Focus2 = RotationMatrix @ Focus2 + Center
    
    ParN = np.array([Focus1[0], Focus1[1], Focus2[0], Focus2[1], 2*a])
    
    return ParN
