"""
Tests for getcape

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import pandas as pd
import MetAnalysis.src.getcape as gc
import pytest


#---------------------------------------------------------------------------------------------------
# Tests
#---------------------------------------------------------------------------------------------------

def test_getcape():
    
    # Read in CM1 Weisman-Klemp sounding

    wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')    

    T = wk_df['theta (K)'].values * wk_df['pi'].values
    qv = wk_df['qv (kg/kg)'].values
    p = wk_df['prs (Pa)'].values
    
    # "True" values from fortran version (run on PSU-ICDS Roar)
    
    truth = {}
    tlab = ['sb_pseudo', 'sb_rev', 'sb_pseudo_ice', 'mu_pseudo', 'ml_pseudo']
    
    truth['sb_pseudo']     = (1884.456, 45.003, 958.656, 1549.249, 11432.290)
    truth['sb_rev']        = (1256.796, 49.647, 958.656, 1697.166, 11083.252)
    truth['sb_pseudo_ice'] = (2022.359, 45.003, 958.656, 1549.249, 11674.379)
    truth['mu_pseudo']     = (2384.172, 0.7342, 1232.674, 1306.146, 11967.149)
    truth['ml_pseudo']     = (1974.274, 31.535, 1008.741, 1502.865, 11548.418)
    
    for (src, adbt, t) in zip(['sfc', 'sfc', 'sfc', 'mu', 'ml'], [1, 2, 3, 1, 1], tlab):
        gc_out = gc.getcape(p, T, qv, source=src, adiabat=adbt)
        assert gc_out[0] == pytest.approx(truth[t][0], rel=0.005)
        assert gc_out[1] == pytest.approx(truth[t][1], rel=0.01)
        assert gc_out[2] == pytest.approx(truth[t][2], rel=0.005)
        assert gc_out[3] == pytest.approx(truth[t][3], rel=0.005)
        assert gc_out[4] == pytest.approx(truth[t][4], rel=0.005)


"""
End test_getcape.py
"""