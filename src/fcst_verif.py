"""
Forecast Verification Functions

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def ctable(fcst, obs):
    """
    Computes the elements of a 2 X 2 contingency table.

    Parameters
    ----------
    fcst : array, bool or int
        Forecasts (1 = event forecasted, 0 = event not forecasted)
    obs : array, bool or int
        Observations (1 = event occurred, 0 = event did not occur)

    Returns
    -------
    cont : array, int
        2 X 2 contingency table

    Notes
    -----
    See Doswell et al. (1990, WAF) Table 1

    """
    
    cont = np.empty([2, 2])
    
    cont[0, 0] = np.logical_and(fcst, obs).sum()
    cont[0, 1] = np.logical_and(fcst, obs == 0).sum()
    cont[1, 0] = np.logical_and(fcst == 0, obs).sum()
    cont[1, 1] = np.logical_and(fcst == 0, obs == 0).sum()
    
    return cont


def fcst_metrics(fcst, obs):
    """
    Computes various forecast verification statistics.
    
    Parameters
    ----------
    fcst : array, bool or int
        Forecasts (1 = event forecasted, 0 = event not forecasted)
    obs : array, bool or int
        Observations (1 = event occurred, 0 = event did not occur)

    Returns
    -------
    pod : float
        Probability of detection
    far : float
        False alarm ratio
    pofd : float
        Probability of false detection
    dfr : float
        Detection failure ratio
    csi : float
        Critical success index
    tss : float
        True skill statistic
    S : float
        Heidke skill score

    Notes
    -----
    See Doswell et al. (1990, WAF)
    
    """
    
    cont = ctable(fcst, obs)
    x = cont[0, 0]
    y = cont[1, 0]
    z = cont[0, 1]
    w = cont[1, 1]
    
    pod  = x / (x + y)
    far  = z / (x + z)
    pofd = z / (z + w)
    dfr  = y / (y + w)
    
    csi  = x / (x + y + z)
    tss = pod - pofd
    S   = (2. * (x*w - y*z)) / (y*y + z*z + 2.*x*w + (y + z)*(x + w))
    
    return pod, far, pofd, dfr, csi, tss, S


"""
End fcst_verif.py
"""