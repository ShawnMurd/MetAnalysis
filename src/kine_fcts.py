"""
Kinematic CM1 Functions

Shawn Murdzek
sfm5282@psu.edu
Date Created: 8/10/2020
Environment: local_py (Python 3.6)
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr
from numba import jit


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

@jit(nopython=True)
def circ(u, v, x1d, y1d, r, nazimuths=72):
    """
    Compute the Eulerian circulation with radius r at each gridpoint. It is assumed that there is 
    constant grid spacing in the x and y dimensions (z dimension can be stretched). Code here is 
    adapted from Paul Markowski's circdist.f code.
    Inputs:
        u = U wind component, shape (nz, ny, nx) (m/s)
        v = V wind component, shape (nz, ny, nx) (m/s)
        x1d = 1D array of x coordinates (km)
        y1d = 1D array of y coordinates (km)
        r = Radius of Eulerian circuits (km)
    Outputs:
        circ = Array of circulation values (m^2/s)
    Keywords:
        nazimuths = Number of points used to define the circle around which C is computed (larger 
            values are more accurate, but also increase the runtime)
    """
    
    # Determine grid spacing
    
    dx = x1d[1] - x1d[0]
    dy = y1d[1] - y1d[0]
    nx, ny = x1d.size, y1d.size
    
    # Determine starting and ending indices for loop (must start with x and y > r)
    
    istart = int(np.ceil(r / dx))
    jstart = int(np.ceil(r / dy))
    
    # Loop over each grid point
    
    circ = np.ones(u.shape) * np.nan
    for i in range(istart, nx - istart - 1):
        for j in range(jstart, ny - jstart - 1):
            
            sumVt = np.zeros(u[:, 0, 0].shape)
            for angle in np.arange(0, 2 * np.pi, 2 * np.pi / nazimuths):
                xtmp = x1d[i] + r * np.cos(angle)
                ytmp = y1d[j] + r * np.sin(angle)
                
                # Bilinearly interpolate winds to (xtmp, ytmp)
                
                sx = int((xtmp - x1d[0]) / dx)
                sy = int((ytmp - y1d[0]) / dy)
                
                a = (xtmp - x1d[sx]) / dx
                b = (ytmp - y1d[sy]) / dy
                
                utmp = ((1 - a) * (1 - b) * u[:, sy, sx] +
                        a * (1 - b) * u[:, sy, sx+1] +
                        (1 - a) * b * u[:, sy+1, sx] +
                        a * b * u[:, sy, sx])
                
                vtmp = ((1 - a) * (1 - b) * v[:, sy, sx] +
                        a * (1 - b) * v[:, sy, sx+1] +
                        (1 - a) * b * v[:, sy+1, sx] +
                        a * b * v[:, sy, sx])
                
                # Compute V (dot) dl
                
                sumVt = sumVt + (utmp * np.cos(0.5 * np.pi + angle) + 
                                 vtmp * np.sin(0.5 * np.pi + angle))
                
            # Compute circulation
            
            dl_mag = 2. * np.pi * r * 1.0e3 / nazimuths
            circ[:, j, i] = sumVt * dl_mag
            
    return circ   


"""
End cm1_kine_fcts.py
"""    