"""
Idealized Soundings Functions

The functions used herein compute CAPE, CIN, LCL, etc. using a the formulation found in getcape.F
within George Bryan's CM1 (cm1r19.8).

A note about Numba JIT: Although the compiled versions of the functions in this module run faster
than non-compiled versions, the overhead associated with compiling the functions can take 
considerable time and compilation is done the first time a JIT-compiled function is encountered in  
each script. This overhead can be reduced by caching the compiled function (cache=True) so that
compilation only occurs when this module is altered. 

Shawn Murdzek
sfm5282@psu.edu
Date Created: October 10, 2019
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import scipy.optimize as so
import scipy.integrate as si
from numba import jit


#---------------------------------------------------------------------------------------------------
# Basic Thermodynamic Functions
#---------------------------------------------------------------------------------------------------

def exner(p):
    """
    Compute the Exner function (nondimensional pressure)

    Parameters
    ----------
    p : array or float 
        Pressure (Pa)

    Returns
    -------
    pi: array or float
        Exner function (unitless)
    
    Notes
    -----
    See Markowski and Richardson (2010) Chpt 2, footnote 8 (pg 20)
    
    """

    rd = 287.04
    cp = 1005.7
    p00 = 100000.0

    pi = (p / p00) ** (rd / cp)

    return pi


def theta(T, p):
    """
    Compute potential temperature

    Parameters
    ----------
    T : array or float 
        Temperature (K)
    p : array or float 
        Pressure (Pa)
            
    Returns
    -------
    theta : array or float 
        Potential temperature  (K)
            
    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.7)
    
    """
    
    return T / exner(p)


def DALR(T0, p):
    """
    Compute the temperature at a series of pressure levels assuming a dry adiabatic process

    Parameters
    ----------
    T0 : float
        Initial parcel temperature (K)
    p : array
        Pressure levels to compute parcel temperatures (Pa)

    Returns
    -------
    T_prof : array
        Parcel temperatures corresponding to the pressure levels in p (K)
        
    Notes
    -----
    Methdology follows that used in dry_lapse from MetPy

    """
    
    rd = 287.04
    cp = 1005.7
    
    return T0 * (p / p[0]) ** (rd / cp)


def getTfromTheta(theta, p):
    """
    Compute the temperature using the potential temperature and pressure
    
    Parameters
    ----------
    theta : array or float 
        Potential temperature (K)
    p : array or float 
        Pressure (Pa)
    
    Returns
    -------
    T : array or float 
        Temperature (K)
            
    """
    
    return theta * exner(p)


@jit(nopython=True, cache=True)
def get_es(T, sfc='l'):
    """
    Compute equilibrium vapor pressure (over liquid water or ice)
    
    Parameters
    ----------
    T : array or float 
        Temperature (K)
    sfc : string, optional
        Surface over which to compute equilibrium vapor pressure ('l' = liquid, 'i' = ice)
    
    Returns
    -------
    e_s : array or float 
        Equilibrium vapor pressure over liquid water (Pa)

    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.16)
    
    """
  
    T = T - 273.15
    if sfc == 'l':
        e_s = 611.2 * np.exp(17.67 * T / (T + 243.5))
    elif sfc == 'i':
        e_s = 611.2 * np.exp(21.8745584 * T / (T + 265.49))

    return e_s


@jit(nopython=True, cache=True)
def get_qvs(T, p, sfc='l'):
    """
    Compute equilibrium water vapor mass mixing ratio (over liquid water or ice)
    
    Parameters
    ----------
    T : array or float 
        Temperature (K)
    p : array or float 
        Pressure (Pa)
    sfc : string, optional
        Surface over which to compute equilibrium vapor pressure ('l' = liquid, 'i' = ice)
    
    Returns
    -------
    qvs : array or float
        Equilibrium water vapor mass mixing ratio (kg / kg)
    
    """

    Rv = 461.5
    Rd = 287.04
    eps = Rd / Rv
    es = get_es(T, sfc=sfc)
    qvs = eps * es / (p - es)

    return qvs


def getTv(T, qv):
    """
    Compute virtual temperature
    
    Parameters
    ----------
    T : array or float 
        Temperature (K)
    qv : array or float 
        Water vapor mass mixing ratio (kg / kg)
    
    Returns
    -------
    Tv : array or float 
        Virtual potential temperature (K)
            
    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.19)i
    
    """
    
    Rv = 461.5
    Rd = 287.04
    
    Tv = T * (1.0 + ((Rv / Rd) * qv)) / (1.0 + qv)
    
    return Tv


def thetav(T, p, qv):
    """
    Compute virtual potential temperature
    
    Parameters
    ----------
    T : array or float 
        Temperature (K)
    p : array or float
        Pressure (Pa)
    qv : array or float 
        Water vapor mass mixing ratio (kg / kg)
    
    Returns
    -------
    thetav : array or float 
        Virtual potential temperature (K)
            
    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.20)
    
    """
    
    return getTv(theta(T, p), qv)


def getTfromTv(Tv, qv):
    """
    Compute the temperature given a virtual temperature and water vapor mass mixing ratio
    
    Parameters
    ----------
    Tv : array or float
        Virtual temperature (K)
    qv : array or float 
        Water vapor mass mixing ratio (kg / kg)
            
    Returns
    -------
    T : array or float 
        Temperature (K)
            
    """
    
    Rv = 461.5
    Rd = 287.04
    
    T = Tv * (1.0 + qv) / (1.0 + ((Rv / Rd) * qv))
    
    return T


def getqv(RH, T, p):
    """
    Compute water vapor mass mixing ratio from relative humidity
    
    Parameters
    ----------
    RH : array or float 
        Relative humidity (decimal)
    T : array or float 
        Temperature (K)
    p : array or float
        Pressure (Pa)
            
    Returns
    -------
    qv : array or float 
        Water vapor mass mixing ratio (kg / kg)
    
    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.14)
    
    """

    return RH * get_qvs(T, p)


def getRH(T, p, qv):
    """
    Compute relative humidity from water vapor mass mixing ratio
    
    Parameters
    ----------
    RH : array or float
        Relative humidity (decimal)
    T : array or float
        Temperature (K)
    p : array or float
        Pressure (Pa)
    
    Returns
    -------
    qv : array or float
        Water vapor mass mixing ratio (kg / kg)
        
    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.14)
    
    """

    return qv / get_qvs(T, p)


@jit(nopython=True, cache=True)
def getTd(T, p, qv):
    """
    Compute dewpoint by inverting the Clausius-Clapeyron equation

    Parameters
    ----------
    T : array or float
        Temperature (K)
    p : array or float
        Pressure (Pa)
    qv : array or float
        Water vapor mass mixing ratio (kg / kg)

    Returns
    -------
    Td : array or float
        Dewpoint (K)
    
    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.16)

    """
    
    Rv = 461.5
    Rd = 287.04
    eps = Rd / Rv
    
    ln_e = np.log(((qv / eps) * p) / (1 + (qv / eps)))
    Td = 273.15 + (1562.1558 - 243.4 * ln_e) / (ln_e - 24.08542)
    
    return Td


def buoy(T_p, p_p, qv_p, T_env, p_env, qv_env):
    """
    Compute buoyancy
    
    Parameters
    ----------
    T_p : array or float
        Parcel temperature (K)
    p_p : array or float 
        Parcel pressure (Pa)
    qv_p : array or float
        Parcel water vapor mass mixing ratio (kg / kg)
    T_env : array or float 
        Environmental temperature (K)
    p_env : array or float
        Environmental pressure (Pa)
    qv_env : array or float 
        Environmental water vapor mass mixing ratio (kg / kg)
    
    Returns
    -------
    B : array or float 
        Buoyancy (m / s^2)
    
    Notes
    -----
    See Markowski and Richardson (2010) eqn (2.78)
    
    """
    
    g = 9.81
    
    thetav_p = thetav(T_p, p_p, qv_p)
    thetav_env = thetav(T_env, p_env, qv_env)
    
    B = g * (thetav_p - thetav_env) / thetav_env
    
    return B


@jit(nopython=True, cache=True)
def getthe(T, p, qv):
    """
    Compute pseudoequivalent potential temperature following the approximation of Bolton (1980, MWR)

    Parameters
    ----------
    T : array or float
        Temperature (K)
    p : array or float
        Pressure (Pa)
    qv : array or float
        Water vapor mass mixing ratio (kg / kg)

    Returns
    -------
    the : array or float
        Pseudoequivalent potential temperature (K)

    """
    
    # Compute LCL temperature
    Td = getTd(T, p, qv)
    Tlcl = 56. + 1. / (1. / (Td - 56.0) + 0.00125 * np.log(T / Td))
        
    # Compute theta-ep
    the = (T * (100000. / p) ** (0.2854 * (1. - 0.28 * qv)) * 
           np.exp(((3376. / Tlcl) - 2.54) * qv * (1. + 0.81 * qv)))
           
    return the


def getLCL(T, p, qv):
    """
    Compute the lifting condensation level pressure using the condition qv = qvs

    Parameters
    ----------
    T : float
        Initial parcel temperature (K)
    p : float
        Initial parcel pressure (Pa)
    qv : float
        Initial parcel water vapor mass mixing ratio (kg/kg)

    Returns
    -------
    plcl : float
        LCL pressure (Pa)

    Notes
    -----
    LCL initial guess comes from Kerry Emanuel's calcsound program

    """
    
    def funct(p, T0=300., p0=100000., qv0=0.001):
        return qv0 - get_qvs(DALR(T, np.array([p0, p]))[-1], p)
    
    rh = getRH(T, p, qv)
    if rh < 1.:
        p0 = p * (rh ** (T / (1669. - 122.*rh - T)))
        plcl = so.root(funct, p0, args=(T, p, qv), tol=0.001).x[0]
    else:
        plcl = p
    
    return plcl


def MALR(T0, p):
    """
    Compute pseudo-moist adiabats

    Parameters
    ----------
    T0 : float
        Starting temperature (K)
    p : array
        Pressure levels to compute parcel temperature at (Pa)

    Returns
    -------
    T_prof : array
        Parcel temperatures following a pseudo-moist adiabat (K)
        
    Notes
    -----
    This function follows MetPy's moist_lapse(), which is based on Bakhshaii (2013, JAMC)

    """

    def funct(p, T):
        rd  = 287.04
        rv  = 461.5
        xlv = 2501000.
        cp  = 1005.7
        eps = rd / rv
        qvs = get_qvs(T, p)
        return (rd*T + xlv*qvs) / (p * (cp + (xlv*xlv*qvs*eps)/(rd*T*T)))
    
    T_prof = si.solve_ivp(funct, (p[0], p[-1]), np.array([T0]), t_eval=p).y
    
    return T_prof.reshape(p.size)


def getTwb(T, p, qv):
    """
    Compute wet-bulb temperature by lifting parcel to LCL, then following a moist adiabat down
    
    Parameters
    ----------
    T : float, scalar
        Temperature (K)
    p : float, scalar
        Pressure (Pa)
    qv : float, scalar
        Water vapor mass mixing ratio (kg / kg)
        
    Returns
    -------
    Twb : float, scalar
        Wet-bulb temperature
        
    Notes
    -----
    See Markowski and Richardson (2010) sect 2.6 and the MetPy function wet_bulbe_temperature()
    
    """
       
    plcl = getLCL(T, p, qv)
    Tlcl = DALR(T, np.array([p, plcl]))[-1]
    Twb = MALR(Tlcl, np.array([plcl, p]))[-1]
    
    return Twb


#---------------------------------------------------------------------------------------------------
# Composite Parameters
#---------------------------------------------------------------------------------------------------

def stp(cape, srh, bwd, lcl, cin):
    """
    Compute significant tornado parameter (STP)

    Parameters
    ----------
    cape : float
        Convective available potential energy (J / kg)
    srh : float
        Storm-relative helicity (m^2 / s^2)
    bwd : float
        Bulk wind difference (m / s)
    lcl : float
        Lifting condensation level (m)
    cin : float
        Convective inhibition (positive number, J / kg)

    Returns
    -------
    stp : float
        Significant tornado parameter (unitless)

    Notes
    -----
    See Thompson et al. (2012, WAF)
    
    """

    if type(cape) == float:
        cape = np.array([cape])
        srh = np.array([srh])
        bwd = np.array([bwd])
        lcl = np.array([lcl])
        cin = np.array([cin])
    
    stp = ((cape/1500.) * (srh/150.) * np.minimum((bwd/20.), 1.5) * 
           np.minimum(((2000.-lcl)/1000.), 1.) * np.minimum(((200.-cin)/150.), 1.))
    
    stp[np.where(bwd < 12.5)] = 0.
    stp[np.where(lcl > 2000.)] = 0.
    stp[np.where(cin > 200.)] = 0.
    stp[np.where(srh < 0.)] = 0.
        
    return stp


#---------------------------------------------------------------------------------------------------
# Functions to Compute Sounding Parameters Following getcape.F from CM1
#---------------------------------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def sounding_pressure(z, th, qv, p0):
    """
    Computes the pressure profile using an upward integration of the hydrostatic balance equation
    
    Parameters
    ----------
    z : array 
        Sounding heights (m)
    th : array
        Sounding potential temperatures (K)
    qv : array 
        Sounding water vapor mass mixing ratios (kg / kg)
    p0 : float 
        Pressure corresponding to z[0] (Pa)
        
    Returns
    -------
    p : array 
        Sounding pressure (Pa)
        
    """

    # Define constants
    reps = 461.5 / 287.04
    rd = 287.04
    cp = 1005.7
    p00 = 100000.0
    g = 9.81

    # Compute Exner function and virtual potential temperature
    pi = np.zeros(z.shape)
    pi[0] = (p0 / p00) ** (rd / cp)
    thv = th * (1.0 + (reps * qv)) / (1.0 + qv)

    # Integrate hydrostatic equation upward from surface
    for i in range(1, z.size):
        pi[i] = pi[i-1] - g * (z[i] - z[i-1]) / (cp * 0.5 * (thv[i] + thv[i-1]))

    p = p00 * (pi ** (cp / rd))

    return p


@jit(nopython=True, cache=True)
def sounding_height(p, th, qv, z0):
    """
    Computes the height profile using an upward integration of the hydrostatic balance equation
    
    Parameters
    ----------
    p : array 
        Sounding pressures (Pa)
    th : array 
        Sounding potential temperatures (K)
    qv : array 
        Sounding water vapor mass mixing ratios (kg / kg)
    z0 : float
        Height corresponding to p[0] (m)
        
    Returns
    -------
    z : array 
        Sounding heights (m)
        
    """

    # Define constants
    reps = 461.5 / 287.04
    rd = 287.04
    cp = 1005.7
    g = 9.81
    cpdg = cp / g

    # Compute Exner function and virtual potential temperature
    pi = (p / 100000.0) ** (rd / cp)
    thv = th * (1.0 + (reps * qv)) / (1.0 + qv)

    # Integrate hydrostatic equation upward from surface
    z = np.zeros(p.shape)
    z[0] = z0
    for i in range(1, p.size):
        z[i] = z[i-1] - cpdg * 0.5 * (thv[i] + thv[i-1]) * (pi[i] - pi[i-1])

    return z


@jit(nopython=True, cache=True)
def _lift_parcel(p, T, qv, source='sfc', adiabat=1, ml_depth=500.0, pinc=10.0, z0=0.0):
    """
    Compute various sounding parameters

    Parameters
    ----------
    p : array
        Pressure profile (Pa)
    T : array
        Temperature profile (K)
    qv : array
        Water vapor mass mixing ratio profile (kg / kg)
    source : string, optional
        Parcel to lift. Options:
            'sfc' = Surface-based
            'mu' = Most-unstable (max theta-e)
            'ml' = Mixed-layer
    adiabat : integer, optional
        Adiabat to follow for parcel ascent. Options:
            1 = Pseudoadiabatic, liquid only
            2 = Reversible, liquid only
            3 = Pseudoadiabatic, with ice
            4 = Reversible, with ice
    ml_depth : float, optional
        Mixed-layer depth for source = 'ml' (m)
    pinc : float, optional
        Pressure increment for integration of hydrostatic equation (Pa)
    z0 : float, optional
        Height corresponding to the first pressure level (m AGL)

    Returns
    -------
    cape : float
        Convective available potential energy (J / kg)
    cin : float
        Convective inhibition (J / kg)
    zlcl : float
        Lifting condensation level (m)
    zlfc : float
        Level of free convetcion (m)
    zel : float
        Equilibrium level (m)
    B_all : array
        Buoyancy profile following the lifted parcel (m / s^2)
    thv_all : array
        Virtual potential temperature profile following the lifted parcel (K)
    qtot_all : array
        Total water mass mixing ratio profile following the lifted parcel (kg/kg)
    
    Notes
    -----
    Credit to George Bryan (NCAR) for writing the original getcape.F subroutine [which is 
    distributed as part of Cloud Model 1 (CM1)]

    """

    # Define constants
    g     = 9.81
    p00   = 100000.0
    cp    = 1005.7
    rd    = 287.04
    rv    = 461.5
    xlv   = 2501000.0
    xls   = 2836017.0
    t0    = 273.15
    cpv   = 1875.0
    cpl   = 4190.0
    cpi   = 2118.636
    lv1   = xlv+(cpl-cpv)*t0
    lv2   = cpl-cpv
    ls1   = xls+(cpi-cpv)*t0
    ls2   = cpi-cpv

    rp00  = 1.0/p00
    reps  = rv/rd
    rddcp = rd/cp
    cpdg  = cp/g

    converge = 0.0002
    nlvl = p.shape[0]

    # Compute derived quantities
    pi = (p*rp00) ** rddcp
    th = T / pi
    thv = th * (1. + reps*qv) / (1. + qv)
    z = sounding_height(p, th, qv, z0)

    # Determine initial parcel location

    if source == 'sfc':
        kmax = 0

    elif source == 'mu':
        idxmax = (p >= 50000.0).sum()
        thetae = getthe(T[:idxmax], p[:idxmax], qv[:idxmax])
        kmax = np.argmax(thetae)

    elif source == 'ml':
        
        if z[1] > ml_depth:
            avgth = th[0]
            avgqv = qv[0]
            kmax = 0
        elif z[-1] < ml_depth:
            avgth = th[-1]
            avgqv = qv[-1]
            kmax = th.size - 1
        else:
            
            # Compute the average theta and qv weighted by the distance between two sounding levels
            
            ktop = np.where(z <= ml_depth)[0][-1]
            ml_th = 0.5 * (th[:ktop] + th[1:(ktop+1)])
            ml_qv = 0.5 * (qv[:ktop] + qv[1:(ktop+1)])
            depths = z[1:(ktop+1)] - z[:ktop]
            
            avgth = np.sum(ml_th * depths) / np.sum(depths)
            avgqv = np.sum(ml_qv * depths) / np.sum(depths)
            kmax = 0

    else:
        kmax = 0
        
    # Define initial parcel properties

    th2  = th[kmax]
    pi2  = pi[kmax]
    p2   = p[kmax]
    T2   = T[kmax]
    thv2 = thv[kmax]
    qv2  = qv[kmax]
    B2   = 0.0

    if source == 'ml':
        th2  = avgth
        qv2  = avgqv
        T2   = th2 * pi2
        thv2 = th2 * (1. + reps*qv2) / (1. + qv2)
        B2   = g * (thv2 - thv[kmax]) / thv[kmax]
        
    # Initialize variables for parcel ascent

    narea = 0.0

    ql2 = 0.0
    qi2 = 0.0
    qt  = qv2

    cape = 0.0
    cin  = 0.0

    cloud = False
    if (adiabat == 1 or adiabat == 2):
        ice = False
    else:
        ice = True

    zlcl       = -1.0
    zlfc       = -1.0
    zel        = -1.0
    B_all      = np.zeros(nlvl - kmax)
    thv_all    = np.zeros(nlvl - kmax)
    qtot_all     = np.zeros(nlvl - kmax)
    B_all[0]   = B2
    thv_all[0] = thv2
    qtot_all[0]  = qv2

    # Parcel ascent: Loop over each vertical level in sounding

    for k in range(kmax+1, nlvl):

        B1 =  B2
        dp = p[k-1] - p[k]

        # Substep dp in increments equal to pinc

        nloop = 1 + int(dp/pinc)
        dp = dp / float(nloop)

        for n in range(nloop):

            p1   =  p2
            T1   =  T2
            th1  = th2
            qv1  = qv2
            ql1  = ql2
            qi1  = qi2

            p2 = p2 - dp
            pi2 = (p2*rp00)**rddcp

            thlast = th1
            i = 0
            not_converged = True

            while not_converged:
                i = i + 1
                T2 = thlast * pi2
                
                if ice:
                    fliq = max(min((T2 - 233.15) / (273.15 - 233.15), 1.0), 0.0)
                    fice = 1.0 - fliq
                else:
                    fliq = 1.0
                    fice = 0.0
                    
                qv2 = min(qt, fliq * get_qvs(T2, p2) + fice * get_qvs(T2, p2, sfc='i'))
                qi2 = max(fice * (qt - qv2), 0.0)
                ql2 = max(qt - qv2 - qi2, 0.0)

                Tbar  = 0.5*(T1 + T2)
                qvbar = 0.5*(qv1 + qv2)
                qlbar = 0.5*(ql1 + ql2)
                qibar = 0.5*(qi1 + qi2)

                lhv = lv1 - lv2*Tbar
                lhs = ls1 - ls2*Tbar

                rm  = rd + rv*qvbar
                cpm = cp + cpv*qvbar + cpl*qlbar + cpi*qibar
                th2 = th1 * np.exp(lhv*(ql2 - ql1) / (cpm*Tbar)  
                                   + lhs*(qi2 - qi1) / (cpm*Tbar) 
                                   + (rm/cpm - rd/cp) * np.log(p2/p1))

                if i > 100:
                    raise RuntimeError('Max number of iterations (100) reached')
                             
                if abs(th2 - thlast) > converge:
                    thlast = thlast + 0.3*(th2 - thlast)
                else:
                    not_converged = False
            
            # Latest pressure increment is complete.  Calculate some important stuff:

            if ql2 >= 1.0e-10: 
                cloud = True
            if (cloud and zlcl < 0.0):
                zlcl = z[k-1] + (z[k]-z[k-1]) * float(n) / float(nloop)

            if (adiabat == 1 or adiabat == 3):
                # pseudoadiabat
                qt  = qv2
                ql2 = 0.0
                qi2 = 0.0            

        thv2 = th2 * (1. + reps*qv2) / (1. + qv2 + ql2 + qi2)
        B2 = g * (thv2 - thv[k]) / thv[k]
        dz = -cpdg * 0.5 * (thv[k] + thv[k-1]) * (pi[k] - pi[k-1])
        B_all[k-kmax] = B2
        thv_all[k-kmax] = thv2
        qtot_all[k-kmax] = qv2 + ql2 + qi2

        #if (zlcl > 0.0 and zlfc < 0.0 and B2 > 0.0):
        #    if B1 > 0.0:
        #        zlfc = zlcl
        #    else:
        #        zlfc = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2 - B1)

        if (zlfc >= 0.0 and zel < 0.0 and B2 < 0.0):
            zel = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2-B1)

        # Get contributions to CAPE and CIN:           

        if (B2 >= 0.0 and B1 <= 0.0):
            # first trip into positive area
            frac = B2 / (B2 - B1)
            parea =  0.5*B2*dz*frac
            narea = narea - 0.5*B1*dz*(1.-frac)
            cin  = cin  + narea
            narea = 0.0
            zlfc = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2 - B1)
        elif (B2 < 0.0 and B1 >= 0.0):
            # first trip into neg area
            frac = B1 / (B1 - B2)
            parea =  0.5*B1*dz*frac
            narea = -0.5*B2*dz*(1.0-frac)
        elif B2 < 0.0:
            # still collecting negative buoyancy
            parea =  0.0
            narea = narea - 0.5*dz*(B1+B2)
        else:
            # still collecting positive buoyancy
            parea =  0.5*dz*(B1+B2)
            narea =  0.0

        cape = cape + max(0.0, parea)

        if (p[k] <= 10000. and B2 <= 0.):
            break

    return cape, cin, zlcl, zlfc, zel, B_all, thv_all, qtot_all


@jit(nopython=True, cache=True)
def _parcel_descent(p, T, qv, idx, adiabat=1, pinc=10.0, z0=0.0):
    """
    Track a parcel as it descends. Currently, only liquid processes are considered 

    Parameters
    ----------
    p : array
        Pressure profile (Pa)
    T : array
        Temperature profile (K)
    qv : array
        Water vapor mass mixing ratio profile (kg / kg)
    idx : integer
        Index corresponding to initial parcel location
    adiabat : integer, optional
        Adiabat to follow for parcel ascent. Options:
            1 = Pseudoadiabatic, liquid only
            2 = Reversible, liquid only
    pinc : float, optional
        Pressure increment for integration of hydrostatic equation (Pa)
    z0 : float, optional
        Height corresponding to the first pressure level (m AGL)

    Returns
    -------
    dcape_tot : float
        Downdraft convective available potential energy from source to surface (J / kg)
    dcape_max : float
        Same as dcape_tot, but only include negative B contributions (J / kg)
    B_all : array
        Buoyancy profile following the descending parcel (m / s^2)
    thv_all : array
        Virtual potential temperature profile following the descending parcel (K)
    qtot_all : array
        Total water mass mixing ratio profile following the descending parcel (kg/kg)
    
    Notes
    -----
    Credit to George Bryan (NCAR) for writing the original getcape.F subroutine [which is 
    distributed as part of Cloud Model 1 (CM1)]
    
    Definition of DCAPE follows Markowski and Richardson (2010) sect 2.6

    """

    # Define constants
    g     = 9.81
    p00   = 100000.0
    cp    = 1005.7
    rd    = 287.04
    rv    = 461.5
    xlv   = 2501000.0
    xls   = 2836017.0
    t0    = 273.15
    cpv   = 1875.0
    cpl   = 4190.0
    cpi   = 2118.636
    lv1   = xlv+(cpl-cpv)*t0
    lv2   = cpl-cpv
    ls1   = xls+(cpi-cpv)*t0
    ls2   = cpi-cpv

    rp00  = 1.0/p00
    reps  = rv/rd
    rddcp = rd/cp
    cpdg  = cp/g

    converge = 0.0002
    nlvl = p.shape[0]

    # Compute derived quantities
    pi = (p*rp00) ** rddcp
    th = T / pi
    thv = th * (1. + reps*qv) / (1. + qv)
    z = sounding_height(p, th, qv, z0)    
    
    # Compute initial parcel properties (saturated with T = wet-bulb temperature)
    p2 = p[idx]
    T2 = getTwb(T[idx], p2, qv[idx])
    qv2 = get_qvs(T2, p2)
    th2  = theta(T2, p2)
    pi2  = exner(p2)
    thv2 = thetav(T2, p2, qv2)
    B2 = buoy(T2, p2, qv2, T[idx], p[idx], qv[idx])
        
    # Initialize variables for parcel descent

    narea = 0.0

    ql2 = 0.0
    qi2 = 0.0
    qt  = qv2

    dcape_tot = 0.0
    dcape_max = 0.0

    cloud = False

    zlcl       = -1.0
    zlfc       = -1.0
    zel        = -1.0
    B_all      = np.zeros(nlvl - kmax)
    thv_all    = np.zeros(nlvl - kmax)
    qtot_all     = np.zeros(nlvl - kmax)
    B_all[0]   = B2
    thv_all[0] = thv2
    qtot_all[0]  = qv2

    # Parcel ascent: Loop over each vertical level in sounding

    for k in range(kmax+1, nlvl):

        B1 =  B2
        dp = p[k-1] - p[k]

        # Substep dp in increments equal to pinc

        nloop = 1 + int(dp/pinc)
        dp = dp / float(nloop)

        for n in range(nloop):

            p1   =  p2
            T1   =  T2
            th1  = th2
            qv1  = qv2
            ql1  = ql2
            qi1  = qi2

            p2 = p2 - dp
            pi2 = (p2*rp00)**rddcp

            thlast = th1
            i = 0
            not_converged = True

            while not_converged:
                i = i + 1
                T2 = thlast * pi2
                
                if ice:
                    fliq = max(min((T2 - 233.15) / (273.15 - 233.15), 1.0), 0.0)
                    fice = 1.0 - fliq
                else:
                    fliq = 1.0
                    fice = 0.0
                    
                qv2 = min(qt, fliq * get_qvs(T2, p2) + fice * get_qvs(T2, p2, sfc='i'))
                qi2 = max(fice * (qt - qv2), 0.0)
                ql2 = max(qt - qv2 - qi2, 0.0)

                Tbar  = 0.5*(T1 + T2)
                qvbar = 0.5*(qv1 + qv2)
                qlbar = 0.5*(ql1 + ql2)
                qibar = 0.5*(qi1 + qi2)

                lhv = lv1 - lv2*Tbar
                lhs = ls1 - ls2*Tbar

                rm  = rd + rv*qvbar
                cpm = cp + cpv*qvbar + cpl*qlbar + cpi*qibar
                th2 = th1 * np.exp(lhv*(ql2 - ql1) / (cpm*Tbar)  
                                   + lhs*(qi2 - qi1) / (cpm*Tbar) 
                                   + (rm/cpm - rd/cp) * np.log(p2/p1))

                if i > 100:
                    raise RuntimeError('Max number of iterations (100) reached')
                             
                if abs(th2 - thlast) > converge:
                    thlast = thlast + 0.3*(th2 - thlast)
                else:
                    not_converged = False
            
            # Latest pressure increment is complete.  Calculate some important stuff:

            if ql2 >= 1.0e-10: 
                cloud = True
            if (cloud and zlcl < 0.0):
                zlcl = z[k-1] + (z[k]-z[k-1]) * float(n) / float(nloop)

            if (adiabat == 1 or adiabat == 3):
                # pseudoadiabat
                qt  = qv2
                ql2 = 0.0
                qi2 = 0.0            

        thv2 = th2 * (1. + reps*qv2) / (1. + qv2 + ql2 + qi2)
        B2 = g * (thv2 - thv[k]) / thv[k]
        dz = -cpdg * 0.5 * (thv[k] + thv[k-1]) * (pi[k] - pi[k-1])
        B_all[k-kmax] = B2
        thv_all[k-kmax] = thv2
        qtot_all[k-kmax] = qv2 + ql2 + qi2

        #if (zlcl > 0.0 and zlfc < 0.0 and B2 > 0.0):
        #    if B1 > 0.0:
        #        zlfc = zlcl
        #    else:
        #        zlfc = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2 - B1)

        if (zlfc >= 0.0 and zel < 0.0 and B2 < 0.0):
            zel = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2-B1)

        # Get contributions to CAPE and CIN:           

        if (B2 >= 0.0 and B1 <= 0.0):
            # first trip into positive area
            frac = B2 / (B2 - B1)
            parea =  0.5*B2*dz*frac
            narea = narea - 0.5*B1*dz*(1.-frac)
            cin  = cin  + narea
            narea = 0.0
            zlfc = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2 - B1)
        elif (B2 < 0.0 and B1 >= 0.0):
            # first trip into neg area
            frac = B1 / (B1 - B2)
            parea =  0.5*B1*dz*frac
            narea = -0.5*B2*dz*(1.0-frac)
        elif B2 < 0.0:
            # still collecting negative buoyancy
            parea =  0.0
            narea = narea - 0.5*dz*(B1+B2)
        else:
            # still collecting positive buoyancy
            parea =  0.5*dz*(B1+B2)
            narea =  0.0

        cape = cape + max(0.0, parea)

        if (p[k] <= 10000. and B2 <= 0.):
            break

    return cape, cin, zlcl, zlfc, zel, B_all, thv_all, qtot_all


def getcape(p, T, qv, source='sfc', adiabat=1, ml_depth=500.0, pinc=10.0, returnB=False, 
            returnTHV=False, returnQTOT=False, z0=0.0):
    """
    Compute various sounding parameters

    Parameters
    ----------
    p : array
        Pressure profile (Pa)
    T : array
        Temperature profile (K)
    qv : array
        Water vapor mass mixing ratio profile (kg / kg)
    source : string, optional
        Parcel to lift. Options:
            'sfc' = Surface-based
            'mu' = Most-unstable (max theta-e)
            'ml' = Mixed-layer
    adiabat : integer, optional
        Adiabat to follow for parcel ascent. Options:
            1 = Pseudoadiabatic, liquid only
            2 = Reversible, liquid only
            3 = Pseudoadiabatic, with ice
            4 = Reversible, with ice
    ml_depth : float, optional
        Mixed-layer depth for source = 'ml' (m)
    pinc : float, optional
        Pressure increment for integration of hydrostatic equation (Pa)
    returnB : boolean, optional
        Option to return buoyancy profile
    returnTHV : boolean, optional
        Option to return virtual potential temperature profile
    returnQTOT : boolean, optional
        Option to return total water mass mixing ratio profile
    z0 : float, optional
        Height corresponding to the first pressure level (m AGL)

    Returns
    -------
    cape : float
        Convective available potential energy (J / kg)
    cin : float
        Convective inhibition (J / kg)
    zlcl : float
        Lifting condensation level (m)
    zlfc : float
        Level of free convetcion (m)
    zel : float
        Equilibrium level (m)
    B : array
        Buoyancy profile following the lifted parcel (m / s^2)
    thv : array
        Virtual potential temperature profile following the lifted parcel (K)
    qtot : array
        Total Water mass mixing ratio profile following the lifted parcel (kg/kg)
    
    Notes
    -----
    Timing tests using the WK82 sample sounding from CM1:
        No JIT       ~ 0.6 s   (on fujita)
        With JIT     ~ 0.004 s (on fujita, not including compilation time)
        Fortran code ~ 0.006 s (on Roar interactive node)
    
    """
    
    out = _lift_parcel(p, T, qv, source=source, adiabat=adiabat, ml_depth=ml_depth, pinc=pinc, 
                       z0=z0)
    out = list(out)
    
    if not returnQTOT:
        del out[7]
    
    if not returnTHV:
        del out[6]
    
    if not returnB:
        del out[5]
    
    return tuple(out)


#---------------------------------------------------------------------------------------------------
# Function to Print Environmental Parameters
#---------------------------------------------------------------------------------------------------

def print_env_param(T, p, qv, print_results=True, adiabat=1):
    """
    Print various thermodynamic environmental parameters (e.g., CAPE, CIN, LFC) using getcape
    
    Parameters
    ----------
    T : array 
        Environmental temperature profile (K)
    p : array 
        Environmental pressure profile (Pa)
    qv : array 
        Environmental water vapor mixing ratio profile (kg / kg)
    print_results : boolean, optional
        Option to print parameters to the screen
    adiabat : integer, optional
        Parcel adiabat for getcape
            1: Pseudoadiabatic, liquid only
            2: Reversible, liquid only
            3: Pseudoadiabatic, with ice
            4: Reversible, with ice
        
    Returns
    -------
    param_dict : dictionary 
        Various environmental parameters

    """
    
    param_dict = {}
    params = ['CAPE', 'CIN', 'LCL', 'LFC', 'EL']
    units = ['J / kg', 'J / kg', 'm', 'm', 'm']

    # Compute sounding parameters using getcape    
    for src in ['sfc', 'mu', 'ml']:
        out = getcape(p, T, qv, source=src, adiabat=adiabat)
        for j, param in enumerate(params):
            param_dict[src + param] = out[j]
    
    # Print results to screen    
    if print_results:
        for src in ['sfc', 'mu', 'ml']:
            for k, (param, u) in enumerate(zip(params, units)):
                print('%s%s = %.2f %s' % (src, param, param_dict[src + param], u))
            print()
        
    return param_dict


#---------------------------------------------------------------------------------------------------
# Functions Related to Vertical Profiles of Sounding Parameters
#---------------------------------------------------------------------------------------------------

def cm1_snd_helper(cm1_sounding):
    """
    Extracts the temperature, water vapor mass mixing ratio, pressure, height, and wind profiles 
    from a CM1 input sounding file
    
    Parameters
    ----------
    cm1_sounding : string 
        CM1 input sounding file
    
    Returns
    -------
    out_df : pd.DataFrame
        CM1 input sounding. Includes:
            T : temperature (K)
            th : potential temperature (K)
            qv : water vapor mass mixing ratio (kg / kg)
            p : pressure (Pa)
            z : height (m)
            u : u wind component (m / s)
            v : v wind component (m / s)
            
    """

    # Extract data from cm1_sounding
    sounding = pd.read_csv(cm1_sounding, delim_whitespace=True, header=None,
                           names=['z', 'theta', 'qv', 'u', 'v'], skiprows=1)

    fptr = open(cm1_sounding)
    line1 = fptr.readline().split()

    p0 = float(line1[0]) * 100.0
    z = np.concatenate(([0], sounding['z'].values))
    th = np.concatenate(([float(line1[1])], sounding['theta'].values))
    qv = np.concatenate(([float(line1[2])], sounding['qv'].values)) / 1000.0
    u = np.concatenate(([np.nan], sounding['u'].values))
    v = np.concatenate(([np.nan], sounding['v'].values))

    fptr.close()

    # Compute pressure and temperature profile
    p = sounding_pressure(z, th, qv, p0)
    T = getTfromTheta(th, p)

    # Create output DataFrame
    out_df = pd.DataFrame({'T':T, 'th':th, 'qv':qv, 'p':p, 'z':z, 'u':u, 'v':v})

    return out_df


def create_calcsound_input(p, T, qv, fname):
    """
    Create input .dat file for Kerry Emanuel's calcsound program
    
    Parameters
    ----------
    p : array 
        Pressure (Pa)
    T : array 
        Temperature (K)
    qv : array 
        Water vapor mass mixing ratio (kg / kg)
    fname : string 
        File to save calcsound input file to (including path)
        
    Returns
    --------
        None, creates input file in the directory specified by fname
        
    """

    # Put thermodynamic variables in correct units
    p = p * 0.01
    T = T - 273.15
    qv = qv * 1000

    # Write results to input file
    fptr = open(fname, 'w')
    fptr.write('  N= %d\n' % len(p))
    fptr.write('   Pressure (mb)     Temperature (C)     Mixing Ratio (g/kg)\n')
    fptr.write('   -------------     ---------------     -------------------\n')

    for i in range(len(p)):
        fptr.write('%.1f     %.1f     %.2f\n' % (p[i], T[i], qv[i]))

    fptr.close()
    return None


def calcsound_out_to_df(out):
    """
    Read in a .out file from Kerry Emanuel's calcsound program as a pandas DataFrame    
    
    Parameters
    ----------
    out : string 
        Name of an output file from Kerry Emanuel's calcsound program
        
    Returns
    -------
    out_df : pd.DataFrame 
        DataFrame with information from out
        
    """

    # Read in output file

    fptr = open(out, 'r')
    lines = fptr.readlines()
    fptr.close()

    # Parse output file contents

    N = int((len(lines) - 11) / 2)
    out_dict = {}
    field1 = ['p', 'Tv (rev)', 'Trho (rev)', 'Tv (pseudo)', 'Trho (pseudo)']
    field2 = ['Rev. PA', 'P.A. PA', 'Rev. NA', 'P.A. NA', 'Rev. CAPE', 'P.A. CAPE', 'DCAPE']
    for f in field1:
        out_dict[f] = []
    for f in field2:
        out_dict[f] = []

    for i in range(3, N+3):
        for f, val in zip(field1, lines[i].strip().split()):
            out_dict[f].append(float(val))

    for i in range(N+11, 2 * N + 11):
        for f, val in zip(field2, lines[i].strip().split()[1:]):
            out_dict[f].append(float(val))

    out_df = pd.DataFrame(out_dict)

    return out_df


def effect_inflow(p, T, qv, min_cape=100, max_cin=250, adiabat=1):
    """
    Compute the effective inflow layer (EIL)
    
    Parameters
    ----------
    p : array 
        Pressure (Pa)
    T : array 
        Temperature (K)
    qv : array 
        Water vapor mass mixing ratio (kg / kg)
    min_cape : float, optional
        CAPE threshold for EIL
    max_cin : float, optional
        CIN threshold for EIL
    adiabat : integer, optional
        Parcel adiabat for getcape:
            1: Pseudoadiabatic, liquid only
            2: Reversible, liquid only
            3: Pseudoadiabatic, with ice
            4: Reversible, with ice
            
    Returns
    -------
    p_top : float 
        Pressure at top of EIL (Pa)
    p_bot : float 
        Pressure at bottom of EIL (Pa)
    i_top : integer 
        Index corresponding to EIL top
    i_bot : integer
        Index corresponding to EIL bottom
            
    Notes
    -----
    This algorithm stops searching for an EIL if the EIL base is above 500 hPa
    
    EIL definition comes from Thompson et al. (2007, WAF)
    
    """
    
    # Set stopping criteria
    istop = np.where(p < 50000.)[0][0]
    
    # Determine bottom of effective inflow layer
    i = 0
    cape, cin, _, _, _ = getcape(p[i:], T[i:], qv[i:], adiabat=adiabat)
    while (cape < min_cape) or (cin > max_cin):
        cape, cin, _, _, _ = getcape(p[i:], T[i:], qv[i:], adiabat=adiabat)
        i = i + 1
        if i > istop:
            return np.nan, np.nan, np.nan, np.nan
    i_bot = i
    p_bot = p[i_bot]
    
    # Determine top of effective inflow layer
    while (cape > min_cape) and (cin < max_cin):
        cape, cin, _, _, _ = getcape(p[i:], T[i:], qv[i:], adiabat=adiabat)
        i = i + 1
    i_top = i
    p_top = p[i_top]
    
    return p_top, p_bot, i_top, i_bot


def param_vprof(p, T, qv, pbot, ptop, adiabat=1, ric=0, rjc=0, zc=1.5, bhrad=10.0, bvrad=1.5,
                bptpert=0.0, maintain_rh=False, xloc=0.0, yloc=0.0, z0=0):
    """
    Compute vertical profiles of sounding parameters and vertical profiles of parcel buoyancy
    
    Parameters
    ----------
    p : array 
        Pressure (Pa)
    T : array 
        Temperature (K)
    qv : array 
        Water vapor mass mixing ratio (kg / kg)
    pbot : float 
        Bottom of layer to compute sounding parameters (Pa)
    ptop : float 
        Top of layer to compute sounding parameters (Pa)
    adiabat : integer, optional 
        Parcel adiabat for getcape:
            1: Pseudoadiabatic, liquid only
            2: Reversible, liquid only
            3: Pseudoadiabatic, with ice
            4: Reversible, with ice
    ric, rjc, zc : floats, optional 
        Center of initiating warm bubble (km)
    bhrad, bvrad : floats, optional 
        Horizontal and vertical radius of initiating warm bubble (K)
    bptpert : float, optional 
        Warm buble perturbation (set to 0 to plot parameters without warm bubble) (K)
    maintain_rh : boolean, optional 
        Keep constant RH in initiating warm bubble
    xloc, yloc : floats, optional 
        Horizontal location of vertical profile (km)
    z0 : float, optional 
        Height of p[0] (only needed if bptpert > 0) (m)
        
    Returns
    -------
    param_df : pd.DataFrame 
        Sounding parameters dictionary
    B : array 
        2D array of parcel buoyancies (m / s^2)

    """
   
    # Add initiating warm bubble

    if not np.isclose(bptpert, 0):

        th = theta(T, p)
        pi = exner(p)
        z = sounding_height(p, th, qv, 0)

        beta = np.sqrt(((xloc - ric) / bhrad) ** 2.0 +
                       ((yloc - rjc) / bhrad) ** 2.0 +
                       (((z / 1000) - zc) / bvrad) ** 2.0)

        thpert = np.zeros(th.shape)
        inds = np.where(beta < 1.0)[0]
        thpert[inds] = bptpert * (np.cos(0.5 * np.pi * beta[inds]) ** 2.0)

        if maintain_rh:
            rh = qv / get_qvs(th * pi, p)
            qv = rh * get_qvs((th + thpert) * pi, p)

        th = th + thpert
        T = th * pi   
 
    # Determine indices for pbot and ptop

    ibot = np.argmin(np.abs(p - pbot))
    itop = np.argmin(np.abs(p - ptop))
    
    # Initialize output dictionary
    
    nlvls = len(p[ibot:itop+1])

    param_dict = {}
    param_dict['p'] = p[ibot:itop+1]
    params = ['CAPE', 'CIN', 'zlcl', 'zlfc', 'zel']    
    for s in params:
        param_dict[s] = np.zeros(nlvls)
    
    B = np.empty((nlvls, p.size))
    B[:, :] = np.nan
    
    # Loop through each vertical level
    
    for i in range(ibot, itop+1):
        out = getcape(p[i:], T[i:], qv[i:], source='sfc', adiabat=adiabat, returnB=True)
        for j, s in enumerate(params):
            param_dict[s][i] = out[j]
        B[i, :len(out[5])] = out[5]
    
    param_df = pd.DataFrame.from_dict(param_dict)
    
    return param_df, B


#---------------------------------------------------------------------------------------------------
# Weisman-Klemp Analytic Thermodynamic Sounding
#---------------------------------------------------------------------------------------------------

def weisman_klemp(z, qv0=0.014, theta0=300.0, p0=100000.0, z_tr=12000.0, theta_tr=343.0, T_tr=213.0, 
                  cm1_out=None):
    """
    Create an analytic thermodynamic profile following the methodology of Weisman and Klemp 
    (1982, MWR) [hereafter WK82]
    
    Parameters
    ----------
    z : array 
        Vertical levels used to compute thermodynamic profile (m)
    qv0 : float, optional
        Surface water vapor mass mixing ratio (kg / kg)
    theta0 : float, optional 
        Surface potential temperature (K)
    p0 : float, optional 
        Surface pressure (Pa)
    z_tr : float, optional 
        Tropopause height (m)
    theta_tr : float, optional 
        Tropopause potential temperature (K)
    T_tr : float, optional 
        Tropopause temperature (K)
    cm1_out : string, optional 
        CM1 output text file to save sounding to (set to None to not create an output file)
        
    Returns
    -------
    snd_df : pd.DataFrame
        WK82 sounding
            T : temperature (K)
            qv : water vapor mass mixing ratio (kg / kg)
            p : pressure (Pa)
            z : height (m)
        
    Notes
    -----
    Implementation follows base.f from George Bryan's CM1r20.1
    
    """

    # Define constants
    
    g = 9.81
    cp = 1005.7

    # Set z[0] equal to 0 if not already

    if not np.isclose(z[0], 0):
        z = np.concatenate((np.array([0.0]), z))

    # Compute theta profile (eqn 1 from WK82)

    theta = theta0 + (theta_tr - theta0) * ((z / z_tr) ** 1.25)
    theta[z > z_tr] = theta_tr * np.exp(g * (z[z > z_tr] - z_tr) / (cp * T_tr))

    # Compute relative humidity profile (eqn 2 from WK82)

    RH = 1.0 - 0.75 * ((z / z_tr) ** 1.25)
    RH[z > z_tr] = 0.25

    # Obtain p and qv iteratively by integrating the hydrostatic balance equation and adjusting
    # qv accordingly

    qv = np.zeros(RH.size)
    p = sounding_pressure(z, theta, qv, p0)

    max_p_dif = 100
    i = 0
    while max_p_dif > 0.01:
        qv = getqv(RH, getTfromTheta(theta, p), p)
        qv[np.where(qv > qv0)] = qv0
        p_new = sounding_pressure(z, theta, qv, p0)
        max_p_dif = np.amax(np.abs(p - p_new))
        p = p_new
        i = i + 1
        if i > 20:
            print('max number of iterations reached')
            break

    # Create output DataFrame

    snd_df = pd.DataFrame({'T':getTfromTheta(theta, p), 'qv':qv, 'p':p, 'z':z})

    # Create CM1 sounding input text file, if desired

    if cm1_out != None:
        fptr = open(cm1_out, 'w')
        fptr.write('%7.2f    %6.2f    %6.3f\n' % (p0 * 0.01, theta0, qv0 * 1e3))
        for i in range(1, z.size):
            fptr.write('%7.1f    %6.2f    %6.4f    0.00    0.00\n' % (z[i], theta[i], qv[i] * 1e3))
        fptr.close()  

    return snd_df


#---------------------------------------------------------------------------------------------------
# Functions for McCaul-Weisman Analytic Sounding
#---------------------------------------------------------------------------------------------------

def getqv_from_thetae(T, p, the):
    """
    Compute the water vapor mass mixing ratio for a given T, p, and theta-e

    Parameters
    ----------
    T : float
        Temperature (K)
    p : float
        Pressure (Pa)
    the : float
        Equivalent potential temperature (K)

    Returns
    -------
    qv : float
        Water vapor mass mixing ratio (kg/kg)

    """
    
    # Define function to find root of
    
    def funct(qv, T=300., p=1.0e5, the=335.):
        return getthe(T, p, qv) - the
    
    # Initial guess for qv (assume relative humidity of 50%)
    
    qv0 = getqv(0.5, T, p)
    qv = so.root(funct, qv0, args=(T, p, the), tol=0.001).x[0]
    
    return qv  


def _mw_pbl(thetae, T_sfc, p_sfc, dz, lapse_rate=0.0085, depth=None, lr=0.0001):
    """
    Construct the PBL of an atmospheric sounding using a constant theta-e value and hydrostatic
    balance. Above the LCL, a constant theta-e layer with a lapse rate slightly less than the moist
    adiabatic lapse rate is used.
    
    Parameters
    ----------
    thetae : float 
        Constant equivalent potential temperature values in the PBL (K)
    T_sfc : float 
        Surface temperature (K)
    p_sfc : float 
        Surface pressure (Pa)
    dz : float 
        Vertical grid spacing (m)
    lapse_rate : float, optional 
        PBL lapse rate (K / m)
    depth : float, optional 
        Height of PBL (m). If None, PBL is terminated at LCL
    lr : float, optional 
        Lapse rate in LFC-LCL layer is equal to the MALR - lr (K / m)
    
    Returns
    -------
    z_prof : array 
        PBL heights (m)
    p_prof : array 
        PBL pressures (Pa)
    T_prof : array 
        PBL temperatures (K)
    qv_prof : array 
        PBL water vapor mass mixing ratios (kg / kg)
        
    Notes
    -----
    General methodology here loosely follows McCaul and Cohen (2002, MWR)
    
    """

    # Define constants

    Rd = 287.04
    g = 9.81

    # Determine surface conditions

    qv_sfc = getqv_from_thetae(T_sfc, p_sfc, thetae)
    Tv = getTv(T_sfc, qv_sfc)

    # Determine LCL pressure

    p_lcl = getLCL(T_sfc, p_sfc, qv_sfc)

    # Create sub-LCL profile

    p_prof = [p_sfc]
    T_prof = [T_sfc]
    qv_prof = [qv_sfc]
    z_prof = [0]

    i = 0
    while p_prof[i] > p_lcl:

        i = i + 1

        # Compute p and T at next level using hydrostatic balance

        p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz) / (Rd * Tv)))
        T_prof.append(T_prof[i-1] - (lapse_rate * dz))
        z_prof.append(z_prof[i-1] + dz)

        # Update qv by forcing thetae to be constant

        qv_prof.append(getqv_from_thetae(T_prof[i], p_prof[i], thetae))
        Tv = getTv(T_prof[i], qv_prof[i])
    
    # Create constant theta-e layer above LCL using an upward integration of the hydrostatic
    # balance eqn, if desired

    if depth != None:
        
        T_adjust = lr * dz

        while z_prof[i] < depth:

            i = i + 1
            p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz) / (Rd * Tv)))

            # Have T decrease at MALR - lr
            
            T_prof.append(MALR(T_prof[i-1], np.array(p_prof[i-1:]))[-1] + T_adjust)
            qv_prof.append(getqv_from_thetae(T_prof[i], p_prof[i], thetae))
            
            # Update Tv

            Tv = getTv(T_prof[i], qv_prof[i])

            z_prof.append(z_prof[i-1] + dz)

    # Turn lists into array

    z_prof = np.array(z_prof)
    p_prof = np.array(p_prof)
    T_prof = np.array(T_prof)
    qv_prof = np.array(qv_prof)

    return z_prof, p_prof, T_prof, qv_prof


def mccaul_weisman(z, E=2000.0, m=2.2, H=12500.0, z_trop=12000.0, RH_min=0.1, p_sfc=1e5,
                   T_sfc=300.0, thetae_pbl=335.0, pbl_lapse=0.009, crit_lapse=0.0095, 
                   pbl_depth=None, lr=0.0001):
    """
    Create an analytic thermodynamic profile following the method in the appendix of McCaul and 
    Weisman (2001, MWR)
    
    Parameters
    ----------
    z : array 
        Sounding vertical levels (m AGL)
    E : float, optional
        Tv-corrected CAPE value (J / kg)
    m : float, optional 
        Buoyancy profile compression parameter
    H : float, optional 
        Vertical scale (m)
    z_trop : float, optional 
        Height of tropopause (m)
    RH_min : float, optional 
        Minimum RH at tropopause (decimal)
    p_sfc : float, optional 
        Pressure at the surface (Pa)
    T_sfc : float, optional 
        Temperature at the surface (K)
    thetae_pbl : float, optional 
        Constant theta-e in PBL (K)
    pbl_lapse : float, optional 
        Lapse rate below LCL (K / m)
    crit_lapse : float, optional 
        Lapse rates greater than crit_lapse are set to the PBL lapse rate (K / m)
    pbl_depth : float, optional
        PBL depth (m). Set to None to use the LCL as the PBL top
    lr : float, optional 
        Lapse rate in LCL-LFC layer is the MALR - lr (K / m)
        
    Returns
    -------
    thermo_prof : pd.DataFrame
        MW01 sounding
            z : height (m)
            prs : pressure (Pa)
            qv : water vapor mixing ratio (kg / kg)
            T : temperature (K) 
            Td : dewpoint (K)
        
    Notes
    -----
    Algorithm loosely follows Paul Markowski's saminit.f with modifications to the original
    methodology as outlined in Warren et al. (2017, MWR)
    
    """
    
    # Define constants
    
    Rd = 287.04
    g = 9.81
    
    # Initialize arrays
    
    Tv_env_prof = np.zeros(z.shape)
    p_prof = np.zeros(z.shape)
    qv_prof = np.zeros(z.shape)
    
    # Determine LCL height and create sub-LCL thermodynamic profile
    
    dz = z[1] - z[0]
    z_pbl, p_pbl, T_pbl, qv_pbl = _mw_pbl(thetae_pbl, T_sfc, p_sfc, dz, lapse_rate=pbl_lapse, 
                                          depth=pbl_depth, lr=lr)
    
    pbl_top_ind = z_pbl.size
    pbl_z = z_pbl[-1]
    
    Tv_env_prof[:pbl_top_ind] = getTv(T_pbl, qv_pbl)
    p_prof[:pbl_top_ind] = p_pbl
    qv_prof[:pbl_top_ind] = qv_pbl
    
    # Extract surface water vapor mixing ratio and LCL relative humidity for later
    
    qv_sfc = qv_prof[0]
    pbl_top_rh = getRH(T_pbl[-1], p_pbl[-1], qv_pbl[-1])
    
    # Determine virtual temperature profile

    T_trop = -999.0
    
    p_prof[pbl_top_ind] = p_prof[pbl_top_ind-1] - ((p_prof[pbl_top_ind-1] * g * dz) / 
                                                   (Rd * Tv_env_prof[pbl_top_ind-1]))

    for i in range(pbl_top_ind, z.size):

        if (z[i] <= z_trop):

            # Determine the parcel virtual temperature

            _, _, _, _, _, thv_parcel = getcape(np.array([p_sfc, p_prof[i]]), 
                                                np.array([T_sfc, Tv_env_prof[i-1]]),
                                                np.array([qv_sfc, qv_prof[i-1]]), 
                                                returnTHV=True)
            Tv_parcel = getTfromTheta(thv_parcel[-1], p_prof[i])

            # Determine environmental temperature using buoyancy profile (eqn A1 from McCaul and
            # Weisman 2001). Use the critical lapse rate if environmental lapse rate exceeds 
            # crit_lapse

            B = (E * ((m / H) ** 2) * (z[i] - pbl_z) * np.exp(-(m / H) * (z[i] - pbl_z)))
            
            Tv_env = Tv_parcel / (1.0 + (B / g))
            T_env = getTfromTv(Tv_env, qv_prof[i-1])
            T_env_prev = getTfromTv(Tv_env_prof[i-1], qv_prof[i-1])
            
            if ((T_env_prev - T_env) / (z[i] - z[i-1]) > crit_lapse):
                Tv_env_prof[i] = getTv((T_env_prev - (crit_lapse * (z[i] - z[i-1]))), qv_prof[i-1])
            else:
                Tv_env_prof[i] = Tv_env
                
            # Compute qv by assuming that RH varies linearly from the PBL top to tropopause
            
            RH = pbl_top_rh + (z[i] - pbl_z) * (RH_min - pbl_top_rh) / (z_trop - pbl_z)
            qv_prof[i] = getqv(RH, T_env, p_prof[i])

        else:
            
            # It assumed that at the tropopause, T = Tv b/c qv is small
            
            if T_trop < 0:
                T_trop = Tv_env_prof[i-1]

            Tv_env_prof[i] = T_trop
            qv_prof[i] = getqv(RH, T_trop, p_prof[i])

        # Find pressure of next vertical level using hydrostatic balance

        if i < (z.size - 1):
            p_prof[i+1] = p_prof[i] - ((p_prof[i] * g * (z[i+1] - z[i])) / (Rd * Tv_env_prof[i]))

    # Correct the profile above the mixed layer iteratively following the procedure discussed
    # in the appendix of Warren et al. (2017)

    T_env_prof = getTfromTv(Tv_env_prof, qv_prof)    
    RH_prof = getRH(T_env_prof, p_prof, qv_prof)
    E_t, _, _, _, _, thv_parcel = getcape(p_prof, T_env_prof, qv_prof, returnTHV=True)
    ratio = E / E_t

    while (np.abs(E - E_t) > 0.5):
        
        # Re-compute temperature and qv profiles using E_t factor

        T_trop = -999.0
        Tv_parcel_prof = getTfromTheta(thv_parcel, p_prof)

        for i in range(pbl_top_ind, z.size):

            if (z[i] > pbl_z) and (z[i] <= z_trop):

                # Determine environmental temperature using buoyancy profile (eqn A1 from McCaul and
                # Weisman 2001). Use the critical lapse rate if environmental lapse rate exceeds 
                # crit_lapse

                B = ratio * E * ((m / H) ** 2) * (z[i] - pbl_z) * np.exp(-(m / H) * (z[i] - pbl_z))
                
                Tv_env = Tv_parcel_prof[i] / (1.0 + (B / g))
                T_env = getTfromTv(Tv_env, qv_prof[i-1])
                T_env_prev = getTfromTv(Tv_env_prof[i-1], qv_prof[i-1])
            
                if ((T_env_prev - T_env) / (z[i] - z[i-1]) > crit_lapse):
                    Tv_env_prof[i] = getTv((T_env_prev - (crit_lapse * (z[i] - z[i-1]))),
                                           qv_prof[i-1])
                else:
                    Tv_env_prof[i] = Tv_env
                
                # Compute qv by assuming that RH varies linearly from PBL top to tropopause
            
                RH_prof[i] = pbl_top_rh + (z[i] - pbl_z) * (RH_min - pbl_top_rh) / (z_trop - pbl_z)
                qv_prof[i] = getqv(RH_prof[i], T_env, p_prof[i])

            elif (z[i] > z_trop):

                if T_trop < 0:
                    T_trop = Tv_env_prof[i-1]

                Tv_env_prof[i] = T_trop
                RH_prof[i] = RH_prof[i-1]
                qv_prof[i] = getqv(RH_prof[i], T_trop, p_prof[i])

            # Find pressure of next vertical level using hydrostatic balance

            if i < (z.size - 1):

                p_prof[i+1] = p_prof[i] - ((p_prof[i] * g * (z[i+1] - z[i])) / 
                                           (Rd * Tv_env_prof[i]))

        # Re-compute E_t

        T_env_prof = getTfromTv(Tv_env_prof, qv_prof)
        E_t, _, _, _, _, thv_parcel = getcape(p_prof, T_env_prof, qv_prof, returnTHV=True)
        ratio = ratio * (E / E_t)
    
    # Fill thermo_prof DataFrame
    
    thermo_prof = pd.DataFrame()
    thermo_prof['z']   = pd.Series(z)
    thermo_prof['prs'] = pd.Series(p_prof)
    thermo_prof['T']   = pd.Series(T_env_prof)
    thermo_prof['qv']  = pd.Series(qv_prof)
    thermo_prof['Td']  = pd.Series(getTd(T_env_prof, p_prof, qv_prof))

    return thermo_prof


"""
End idealized_sounding_fcts.py
"""