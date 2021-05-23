"""
Functions Related to the Largest Contiguous Area within a 2D Array

Shawn Murdzek
sfm5282@psu.edu
Date Created: April 5, 2020
Environment: local_py (Python 3.6)
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
from scipy import ndimage


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def largestArea(x):
    """
    Find the largest continguous area where x == True

    Parameters
    ----------
        x : array 
            2D array of boolean values
            
    Returns
    -------
        result : integer 
            Number of grid points in the largest area within x
        iinds_f : array
            j indices corresponding to the grid points within the largest contiguous area in x
        jinds_f : array
            j indices corresponding to the grid points within the largest contiguous area in x

    Notes
    -----
    This script is partially based on Alex Schueth's SVC detection code (see Alex's MS thesis)

    """

    # Note that anything labeled with a '0' is the background (i.e., x == False)
    # This is why the first element in label_num (= 0) is ignored

    labels, _ = ndimage.label(x)
    label_num, ct = np.unique(labels, return_counts=True)
    
    result = np.amax(ct[1:])
    iinds_f, jinds_f = np.where(labels == label_num[1 + np.argmax(ct[1:])])

    return result, iinds_f, jinds_f


def weighted_avg_ctr(A, wgts, thres):
    """
    Find the weighted average of A using the weights in wgts. 

    Before computing the weighted average, all weights outside the largest, contiguous area where 
    wgts > thres are set to 0. In essence, this function finds the center (using a weighted average) 
    of the largest, contiguous area where wgts > thres.

    Parameters
    ----------
        A : array 
            2D array to take average of
        wgts : array 
            2D array of weights
        thres : float 
            Only compute weighted average in the largest, contiguous area where wgts > thres

    Returns
    -------
        Weighted-average from A

    """

    _, iind, jind = largestArea(wgts >= thres)

    mask = np.zeros(wgts.shape)
    mask[iind, jind] = 1
    wgts = mask * wgts

    return np.average(A, weights=wgts)


def supercell_cent(x2d, y2d, uh, uh_thres=50, coord=[np.nan, np.nan], max_dist=10):
    """
    Determine the centroid of a supercell using center of the largest, contiguous area where 
    uh >= uh_thres. It is assumed that uh is the 2-5 km updraft helicity, but other fields could
    also be used.
    
    If coord is not None, then the supercell centroid is forced to be within max_dist of coord.
    This means that the supercell centroid could actually be the second, third, etc. largest area
    with uh >= uh_thres.
    
    Inputs:
        x2d = Grid of x-coordinates (km)
        y2d = Grid of y-coordinates (km)
        uh = 2-5 km updraft helicity
    Outputs:
        xcent = X-coordinate of supercell centroid (km)
        ycent = Y-coordinate of supercell centroid (km)
    Keywords:
        uh_thres = Supercell centroid is defined as the center of the largest area with 
            uh >= uh_thres
        coord = Previous supercell centroid coordinate
        max_dist = Maximum distance from coord that the supercell centroid can be (km)
    """
    
    xcent = np.nan
    d2 = max_dist**2
    
    while np.isnan(xcent):
        
        # Find weighted center of largest area with uh >= uh_thres
        
        _, iind, jind = largestArea(uh >= uh_thres)
        mask = np.zeros(uh.shape)
        mask[iind, jind] = 1
        wgts = mask * uh
        xcent = np.average(x2d, weights=wgts)
        ycent = np.average(y2d, weights=wgts)
        
        if np.isnan(coord[0]):
            break
        elif ((xcent - coord[0])**2 + (ycent - coord[1])**2) > d2:
            uh = (1 - mask) * uh
            xcent = np.nan
            
    return xcent, ycent


"""
End largest_area.py
""" 
