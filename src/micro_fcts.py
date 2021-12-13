"""
Microphysical Functions

Shawn Murdzek
sfm5282@psu.edu
Date Created: 13 December 2021
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import scipy.special as spec


#---------------------------------------------------------------------------------------------------
# Functions for Bulk Microphysics Parameterizations with Gamma PSDs
#---------------------------------------------------------------------------------------------------

# Rain m-D and V-D parameters from the Morrison et al. (2005, 2009) 2-moment scheme

rhow = 997.
Ar = 841.99667
Br = 0.8
Cr = (np.pi / 6.) * rhow
Dr = 3.

def lamda(N, q, mu=0, c=Cr, d=Dr):
    """
    Compute PSD slope parameter
    
    Parameters
    ----------
    N : array or float
        Total number mixing ratio (# / kg)
    q : array or float
        Mass mixing ratio (kg / kg)
    mu : float, optional
        Shape parameter (unitless)
    c : float, optiona
        Mass-diameter coefficient
    d : float, optional
        Mass-diameter exponent (unitless)
    
    Returns
    -------
    lmda : array or float
        PSD slope parameter
    
    """
    
    lmda = ((c*N*spec.gamma(d+mu+1.)) / (q*spec.gamma(mu+1.))) ** (1./d)
    
    return lmda


def N0(N, q, mu=0, c=Cr, d=Dr):
    """
    Compute PSD intercept parameter
    
    Parameters
    ----------
    N : array or float
        Total number mixing ratio (# / kg)
    q : array or float
        Mass mixing ratio (kg / kg)
    mu : float, optional
        Shape parameter (unitless)
    c : float, optiona
        Mass-diameter coefficient
    d : float, optional
        Mass-diameter exponent (unitless)
    
    Returns
    -------
    n0 : array or float
        PSD intercept parameter
    
    """
    
    lmda = lamda(N, q, mu=mu, c=Cr, d=Dr)
    n0 = (N*(lmda**(mu+1.))) / spec.gamma(mu+1.)
    
    return n0
    

"""
End micro_fcts.py
"""