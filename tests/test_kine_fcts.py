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


"""
End test_kine_fcts.py
"""
