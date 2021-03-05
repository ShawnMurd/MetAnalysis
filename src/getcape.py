"""
getcape.F, but it's in Python!

This code works, but it's ~100x slower than the fortran version

Timing tests using the WK82 sample sounding from CM1:
    This code    ~ 0.6 s   (on fujita)
    Fortran code ~ 0.006 s (on Roar interactive node)

Shawn Murdzek
sfm5282@psu.edu
Date Created: 3 February 2021
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import MetAnalysis.src.idealized_sounding_fcts as isf


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def getcape(p, T, qv, source='sfc', adiabat=1, ml_depth=500.0, pinc=10.0):
    """
    Compute various sounding parameters.

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
    
    pi = isf.exner(p)
    th = isf.theta(T, p)
    thv = isf.thetav(T, p, qv)
    z = isf.sounding_height(p, th, qv, 0.0)

    # Determine initial parcel location

    if source == 'sfc':
        kmax = 0

    elif source == 'mu':
        idxmax = (p >= 50000.0).sum()
        thetae = isf.getthe(T[:idxmax], p[:idxmax], qv[:idxmax])
        kmax = np.argmax(thetae)

    elif source == 'ml':
        
        if z[1] > ml_depth:
            avgth = th[0]
            avgqv = qv[0]
            kmax = 0
        elif z[-1] < ml_depth:
            avgth = th[-1]
            avgqv = qv[-1]
            kmax = th.size -1
        else:
            
            # Compute the average theta and qv weighted by the distance between two sounding levels
            
            ktop = np.where(z <= ml_depth)[0][-1]
            ml_th = 0.5 * (th[:ktop] + th[1:(ktop+1)])
            ml_qv = 0.5 * (qv[:ktop] + qv[1:(ktop+1)])
            depths = z[1:(ktop+1)] - z[:ktop]
            
            avgth = np.average(ml_th, weights=depths)
            avgqv = np.average(ml_qv, weights=depths)
            kmax = 0

    else:
        print()
        print('Unknown value for source (source = %s), using surface-based parcel instead' % source)
        print()
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
        T2   = isf.getTfromTheta(th2, p2)
        thv2 = isf.thetav(T2, p2, qv2)
        B2   = isf.buoy(T2, p2, qv2, T[kmax], p[kmax], qv[kmax])
        
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

    zlcl = -1.0
    zlfc = -1.0
    zel  = -1.0

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
                    
                qv2 = min(qt, fliq * isf.get_qvs(T2, p2) + fice * isf.get_qvs(T2, p2, sfc='i'))
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

                if i > 90: 
                    print('%d, %.2f, %.2f, %.2f' % (i, th2, thlast, th2 - thlast))
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
            elif (adiabat <= 0 or adiabat >= 5):
                raise RuntimeError('Undefined adiabat (%d)' % adiabat)

        thv2 = th2 * (1. + reps*qv2) / (1. + qv2 + ql2 + qi2)
        B2 = g * (thv2 - thv[k]) / thv[k]
        dz = -cpdg * 0.5 * (thv[k] + thv[k-1]) * (pi[k] - pi[k-1])

        if (zlcl > 0.0 and zlfc < 0.0 and B2 > 0.0):
            if B1 > 0.0:
                zlfc = zlcl
            else:
                zlfc = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2 - B1)

        if (zlfc > 0.0 and zel < 0.0 and B2 < 0.0):
            zel = z[k-1] + (z[k] - z[k-1]) * (0.0 - B1) / (B2-B1)

        # Get contributions to CAPE and CIN:

        if (B2 >= 0.0 and B1 < 0.0):
            # first trip into positive area
            frac = B2 / (B2 - B1)
            parea =  0.5*B2*dz*frac
            narea = narea - 0.5*B1*dz*(1.-frac)
            cin  = cin  + narea
            narea = 0.0
        elif (B2 < 0.0 and B1 > 0.0):
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

    return cape, cin, zlcl, zlfc, zel

# Perform tests

import pandas as pd
import datetime as dt

wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')    

T = wk_df['theta (K)'].values * wk_df['pi'].values
qv = wk_df['qv (kg/kg)'].values
p = wk_df['prs (Pa)'].values

print('Calling getcape...')
print(dt.datetime.now())
start = dt.datetime.now()
gc_out = getcape(p, T, qv, source='sfc', adiabat=1)
print('total time =', dt.datetime.now() - start)
print(gc_out)


"""
End getcape.py
"""