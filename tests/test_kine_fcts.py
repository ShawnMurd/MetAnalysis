"""
Tests for Kinematic Functions

To run tests, simply enter the command `pytest` in the Anaconda shell

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import xarray as xr
import MetAnalysis.src.kine_fcts as kf


#---------------------------------------------------------------------------------------------------
# Test Functions
#---------------------------------------------------------------------------------------------------

def test_circ():
    
    # Read in reference circulation values

    try:    
        cm1_ds = xr.open_dataset(r'..\sample_data\cm1_winds.nc')
    except:
        cm1_ds = xr.open_dataset(r'../sample_data/cm1_winds.nc')

    circ_ref = cm1_ds['circ_1km'].values
    u = cm1_ds['uinterp'].values
    v = cm1_ds['vinterp'].values
    x = cm1_ds['ni'].values
    y = cm1_ds['nj'].values
    
    cm1_ds.close()
    
    # Compute circulation and compare to reference value
    
    circ = kf.circ(u, v, x, y, 1, nazimuths=72)
    rmsd = np.sqrt(np.nanmean((circ - circ_ref)**2))
    
    assert rmsd <= 0.5


def test_avg_var():
    
    # Create test data
    
    x1d = np.arange(5)
    y1d = np.arange(5)
    A = np.array([[[[1, 1, 1, 1, 1],
                    [1, 2, 3, 4, 5],
                    [3, 5, 7, 9, 11],
                    [2, 4, 6, 8, 10],
                    [2, 1, 0, 1, 2 ]]]])
    r = 1.5
    avg_ref = np.array([[[[np.nan, np.nan, np.nan, np.nan, np.nan],
                          [np.nan, 2.6667, 3.6667, 4.6667, np.nan],
                          [np.nan, 3.6667, 5.3333, 7.0000, np.nan],
                          [np.nan, 3.3333, 4.5556, 6.0000, np.nan],
                          [np.nan, np.nan, np.nan, np.nan, np.nan]]]])
    
    # Compute average variable and compare to reference value
    
    avg = kf.avg_var(A, x1d, y1d, r)
    rmsd = np.sqrt(np.nanmean((avg - avg_ref)**2))
    
    assert rmsd <= 0.001


"""
End test_kine_fcts.py
"""
