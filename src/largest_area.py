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
    This script is partially based on Alex Schueth's SVC detection code (see Alex's MS thesis)
    Inputs:
        x = 2D array of boolean values
    Outputs:
        result = Number of grid points in largest area in x
        iinds_f, jinds_f = Indices corresponding to the grid points within the largest contiguous 
            area in x
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
    Find the weighted average of A using the weights in wgts. Before computing the weighted average,
    all weights outside the largest, contiguous area where wgts > thres are set to 0. In essence,
    this function finds the center (using a weighted average) of the largest, contiguous area where 
    wgts > thres.
    Inputs:
        A = 2D array to take average of
        wgts = 2D array of weights
        thres = Only compute weighted average in the largest, contiguous area where wgts > thres
    Outputs:
        Weighted-average from A
    """

    _, iind, jind = largestArea(wgts >= thres)

    mask = np.zeros(wgts.shape)
    mask[iind, jind] = 1
    wgts = mask * wgts

    return np.average(A, weights=wgts)


"""
End largest_area.py
""" 