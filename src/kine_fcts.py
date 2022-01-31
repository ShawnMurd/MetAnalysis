"""
Kinematic Functions

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import skimage.morphology as sm
from numba import jit
from scipy.signal import convolve2d


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


def avg_var(var, x1d, y1d, r):
    """
    Compute the the average of a variable within a user-defined radius
    
    It is assumed that there is constant grid spacing in the x and y directions (the z dimension 
    can be stretched). 
    
    Parameters
    ----------
    var : array
        Variable to average, shape (nt, nz, ny, nx)
    x1d : array
        1D array of x coordinates (km)
    y1d : array 
        1D array of y coordinates (km)
    r : float
        Radius of used to compute the average (km)
        
    Returns
    -------
    avg : array 
        var averaged within circles with a radius of r
    
    """
    
    # Determine grid spacing
    
    dx = 0.001*int((x1d[1] - x1d[0])*1000)
    dy = 0.001*int((y1d[1] - y1d[0])*1000)
    
    # Create kernel for 2D convolution
    
    krx = int(r / dx)
    kry = int(r / dy)
    kx1d = np.arange(-krx*dx, krx*dx+(0.1*dx), dx)
    ky1d = np.arange(-kry*dy, kry*dy+(0.1*dy), dy)
    kx2d, ky2d = np.meshgrid(kx1d, ky1d)
    dist2 = kx2d*kx2d + ky2d*ky2d
    kernel = np.zeros([2*krx+1, 2*kry+1])
    inds = dist2 <= (r*r)
    kernel[inds] = 1
    
    # Use convolve2d to compute average of var within a disk with radius r
    
    avg = np.zeros(var.shape)
    for i in range(avg.shape[0]):
        for j in range(avg.shape[1]):
            avg[i, j, :, :] = (convolve2d(var[i, j, :, :], kernel, mode='same', fillvalue=np.nan) 
                               / kernel.sum())
            
    return avg


@jit(nopython=True, cache=True)
def azprof(var, xctr, yctr, x1d, y1d, radii, step='flexible', avg=True):
    """
    Compute the azimuthal average or sum of a field at several radii. 
    
    It is assumed that there is constant grid spacing in the x and y directions (the z dimension 
    can be stretched). 
    
    Parameters
    ----------
    var : array
        Variable to take azimuthal average of, shape (nt, nz, ny, nx)
    xctr : array
        x location of centroid, shape (nt) (km)
    yctr : array
        y location of centroid, shape (nt) (km)
    x1d : array
        1D array of x coordinates (km)
    y1d : array 
        1D array of y coordinates (km)
    radii : array
        Radii at which to compute azimuthal averages (km)
    step: float, optional
        Distance between interpolation points along a circuit (km)
            If 'flexible is selected, step = min(dx, dy)
    avg : boolean, optional
        Option to compute the azimuthal average (if set to False, the azimuthal sum is returned)
        
    Returns
    -------
    prof : array 
        Azimuthal profiles, shape (nt, nz, nr)
    
    Notes
    -----
    Code loosely based on Paul Markowski's circdist.f program
    
    """
    
    # Determine grid spacing
    
    dx = 0.001*int((x1d[1] - x1d[0])*1000)
    dy = 0.001*int((y1d[1] - y1d[0])*1000)
    nt = var.shape[0]
    nz = var.shape[1]
    if step == 'flexible':
        step = min(dx, dy)
    
    # Loop over each time and radii
    
    prof = np.zeros((nt, nz, radii.size))
    for i in range(nt):
        for j, r in enumerate(radii):
            nazimuths = int(2*np.pi*r/step)
            for angle in np.arange(0, 2 * np.pi, 2 * np.pi / nazimuths):
                xtmp = xctr[i] + r * np.cos(angle)
                ytmp = yctr[i] + r * np.sin(angle)
                
                # Bilinearly interpolate var to (xtmp, ytmp)
                    
                sx = int((xtmp - x1d[0]) / dx)
                sy = int((ytmp - y1d[0]) / dy)
            
                a = (xtmp - x1d[sx]) / dx
                b = (ytmp - y1d[sy]) / dy
                
                prof[i, :, j] = prof[i, :, j] + ((1 - a) * (1 - b) * var[i, :, sy, sx] +
                                                 a * (1 - b) * var[i, :, sy, sx+1] +
                                                 (1 - a) * b * var[i, :, sy+1, sx] +
                                                 a * b * var[i, :, sy, sx])

            if avg:
                prof[i, :, j] = float(prof[i, :, j] / nazimuths)               
            
    return prof


"""
End kine_fcts.py
"""    
