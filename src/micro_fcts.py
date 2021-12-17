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
    c : float, optional
        Mass-diameter coefficient
    d : float, optional
        Mass-diameter exponent (unitless)
    
    Returns
    -------
    n0 : array or float
        PSD intercept parameter
    
    """
    
    lmda = lamda(N, q, mu=mu, c=c, d=d)
    n0 = (N*(lmda**(mu+1.))) / spec.gamma(mu+1.)
    
    return n0


def invtaur(qr, nr, rho, T, p, mur=0, a=Ar, b=Br, c=Cr, d=Dr):
    """
    Compute the inverse phase relaxation time for rain

    Parameters
    ----------
    qr : array
        Rain mass mixing ratio (kg / kg)
    nr : aray
        Rain number mixing ratio (# / kg)
    rho : array
        Air density (kg / m^3)
    T : array
        Air temperature (K)
    p : array
        Air pressure (Pa)
    mur : float, optional
        Rain DSD shape parameter
    a : float, optional
        Rain velocity-diameter relationship coefficient
    b : float, optional
        Rain velocity-diameter relationship exponent
    c : float, optional
        Rain mass-diameter relationship coefficient
    d : float, optional
        Rain mass-diameter relationship exponent

    Returns
    -------
    itaur : array
        Inverse phase relaxation time for rain (/s)

    """

    # Define constants

    R = 287.04
    rhow = 997.
    f1 = 0.78
    f2 = 0.308
    cnst1 = 2.5 + b/2. + mur
    cnst2 = spec.gamma(cnst1)

    # Environmental Parameters

    mu = 1.496e-6*T**1.5/(T+120.)
    Dv = 8.794e-5*T**1.81/p
    Sc = mu / (rho*Dv)

    # Add density correction to rain fall speed

    rhosu = 85000./(R*273.15)
    ARN = a*((rhosu / rho)**0.54)

    # Compute DSD parameters

    lmda = lamda(nr, qr, mu=mur, c=c, d=d)
    n0 = N0(nr, qr, mu=mur, c=c, d=d)

    itaur = 2.*np.pi*rho*Dv*n0*((f1/(lmda**2.)) +
                                (f2*(((ARN*rho)/mu)**0.5)*(Sc**(1./3.))*cnst2/(lmda**cnst1)))

    return itaur
   

"""
End micro_fcts.py
"""
