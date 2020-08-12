"""
Tests for largest_area.py

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import MetAnalysis.src.largest_area as la
import numpy as np


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

def test_largestArea():

    X = np.array([[1, 4, 5, 3, 2], 
                  [-2, 0, 6, 3, 0],
                  [3, 4, 1, 4, 5],
                  [7, 0, 0, -4, 1]])

    size, iind, jind = la.largestArea(X >= 3)

    assert size == 7
    assert iind == [0, 0, 0, 1, 1, 2, 2]
    assert jind == [1, 2, 3, 3, 2, 3, 4]


def test_weighted_avg_ctr():

    wgts = np.array([[1, 4, 5, 3, 2], 
                     [-2, 0, 6, 3, 0],
                     [3, 4, 1, 4, 5],
                     [7, 0, 0, -4, 1]])

    A = np.zeros(wgts.shape)
    for i in range(4):
        A[i, :] = np.arange(5)

    avg = la.weighted_avg_ctr(A, wgts, 3)
    true_avg = np.average(np.array([1, 2, 3, 2, 3, 3, 4]), 
                          weights=np.array([4, 5, 3, 6, 3, 4, 5]))

    assert np.isclose(avg, true_avg)
