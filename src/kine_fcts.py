"""
Kinematic Functions

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
from numba import jit


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def circ(u, v, x1d, y1d, r, nazimuths=72):
    """
    Compute the Eulerian circulation with radius r at each gridpoint. 
    
    It is assumed that there is constant grid spacing in the x and y directions (the z dimension 
    can be stretched). 
    
    Parameters
    ----------
    u : array
        Zonal wind component, shape (nz, ny, nx) (m/s)
    v : array
        Meridional wind component, shape (nz, ny, nx) (m/s)
    x1d : array
        1D array of x coordinates (km)
    y1d : array 
        1D array of y coordinates (km)
    r : float
        Radius of Eulerian circuits (km)
    nzazimuths : float, optional
        Number of points used to define the circle around which circulation is computed (larger 
            values are more accurate, but also increase the runtime)
        
    Returns
    -------
    circ : array 
        Eulerian circulation (m^2/s)
    
    Notes
    -----
    Code adapted from Paul Markowski's circdist.f program
    
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
                
                # Compute Vt (dot) dl, where Vt is the wind component tangential to the circuit
                # Note: -sin(angle) = cos(pi/2 + angle) and cos(angle) = sin(pi/2 + angle)
                
                sumVt = sumVt + (utmp * np.cos(0.5 * np.pi + angle) + 
                                 vtmp * np.sin(0.5 * np.pi + angle))
                
            # Compute circulation (1e3 is needed to convert r from km to m)
            
            dl_mag = 2. * np.pi * r * 1.0e3 / nazimuths
            circ[:, j, i] = sumVt * dl_mag
            
    return circ


"""
End kine_fcts.py
"""    