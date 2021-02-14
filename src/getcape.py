"""
getcape.F, but it's in Python!

Shawn Murdzek
sfm5282@psu.edu
Date Created: 3 February 2021
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import idealized_sounding_fcts as isf


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
    eps   = rd/rv
    reps  = rv/rd
    rddcp = rd/cp
    cpdrd = cp/rd
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
        thetae = getthe(T, p, qv)
        kmax = np.argmax(thetae)
        maxthe = np.amax(thetae)

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

    # Initialize arrays for parcel quantities

    pT  = np.zeros(nlvl)
    pTd = np.zeros(nlvl)
    pTv = np.zeros(nlvl)
    pB  = np.zeros(nlvl)
    pc  = np.zeros(nlvl)
    pn  = np.zeros(nlvl)
    pqv = np.zeros(nlvl)
    pql = np.zeros(nlvl)
    
    pT[kmax]  = T2
    pTd[kmax] = isf.getTd(T2, p2, qv2)
    pTv[kmax] = isf.getTv(T2, qv2)
    pqv[kmax] = qv2
        
    # Initialize variables for parcel ascent

    narea = 0.0

    psource  = p2
    Tsource  = T2
    qvsource = qv2

    ql2 = 0.0
    qi2 = 0.0
    qt  = qv2

    cape = 0.0
    cin  = 0.0
    lfc  = 0.0

    cloud = False
    if (adiabat == 1 or adiabat == 2):
        ice = False
    else:
        ice = True

    the = isf.getthe(T2, p2, qv2)

    zlcl = -1.0
    zlfc = -1.0
    zel  = -1.0

    # Parcel ascent: Loop over each vertical level in sounding

    for k in range(kmax, nlvl):

        B1 =  B2
        dp = p[k-1] - p[k]

        # Substep dp in increments equal to pinc

        nloop = 1 + int( dp/pinc )
        dp = dp / float(nloop)

        for n in range(nloop):

            p1   =  p2
            T1   =  t2
            pi1  = pi2
            th1  = th2
            qv1  = qv2
            ql1  = ql2
            qi1  = qi2
            thv1 = thv2

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
            qv2 = min(qt, fliq * isf.get_qvl(T2, p2) + fice * isf.get_qvi(T2, p2))
            qi2 = max(fice * (qt - qv2), 0.0)
            ql2 = max(qt - qv2 - qi2, 0.0)

"""
End here on 2/14/2021
"""

            tbar  = 0.5*(t1+t2)
            qvbar = 0.5*(qv1+qv2)
            qlbar = 0.5*(ql1+ql2)
            qibar = 0.5*(qi1+qi2)

            lhv = lv1-lv2*tbar
            lhs = ls1-ls2*tbar
            lhf = lhs-lhv

            rm=rd+rv*qvbar
            cpm=cp+cpv*qvbar+cpl*qlbar+cpi*qibar
            th2=th1*exp(lhv*(ql2-ql1)/(cpm*tbar)  
     $                  +lhs*(qi2-qi1)/(cpm*tbar) 
     $                  +(rm/cpm-rd/cp)*alog(p2/p1))

            if(i.gt.90) print *,i,th2,thlast,th2-thlast
            if(i.gt.100)then
              print *
              print *,'  Error:  lack of convergence'
              print *
              print *,'  ... stopping iteration '
              print *
!!!            stop 1001
            ! 171020:
              cape = 0.0
              cin = 0.0
              psource = 0.0
              tsource = 0.0
              qvsource = 0.0
              return
            endif
            if( abs(th2-thlast).gt.converge )then
              thlast=thlast+0.3*(th2-thlast)
            else
              not_converged = .false.
            endif
          enddo

        ! Latest pressure increment is complete.  Calculate some
        ! important stuff:

          if( ql2.ge.1.0e-10 ) cloud = .true.
          if( cloud .and. zlcl.lt.0.0 )then
            zlcl = z(k-1)+(z(k)-z(k-1))*float(n)/float(nloop)
            plcl = p(k-1)+(p(k)-p(k-1))*float(n)/float(nloop)
          endif

          IF(adiabat.eq.1.or.adiabat.eq.3)THEN
            ! pseudoadiabat
            qt  = qv2
            ql2 = 0.0
            qi2 = 0.0
          ELSEIF(adiabat.le.0.or.adiabat.ge.5)THEN
            print *
            print *,'  Undefined adiabat'
            print *
            stop 10000
          ENDIF

        enddo

        thv2 = th2*(1.0+reps*qv2)/(1.0+qv2+ql2+qi2)
        b2 = g*( thv2-thv(k) )/thv(k)
        dz = -cpdg*0.5*(thv(k)+thv(k-1))*(pi(k)-pi(k-1))

        if( zlcl.gt.0.0 .and. zlfc.lt.0.0 .and. b2.gt.0.0 )then
          if( b1.gt.0.0 )then
            zlfc = zlcl
            plfc = plcl
          else
            zlfc = z(k-1)+(z(k)-z(k-1))*(0.0-b1)/(b2-b1)
            plfc = p(k-1)+(p(k)-p(k-1))*(0.0-b1)/(b2-b1)
          endif
        endif

        if( zlfc.gt.0.0 .and. zel.lt.0.0 .and. b2.lt.0.0 )then
          zel = z(k-1)+(z(k)-z(k-1))*(0.0-b1)/(b2-b1)
          pel = p(k-1)+(p(k)-p(k-1))*(0.0-b1)/(b2-b1)
        endif

        the = getthx(p2,t2,t2,qv2)

        pt(k) = t2
        if( cloud )then
          ptd(k) = t2
        else
          ptd(k) = gettd(p2,t2,qv2)
        endif
        ptv(k) = t2*(1.0+reps*qv2)/(1.0+qv2)
        pb(k) = b2
        pqv(k) = qv2
        pql(k) = ql2

      ! Get contributions to CAPE and CIN:

        if( (b2.ge.0.0) .and. (b1.lt.0.0) )then
          ! first trip into positive area
          ps = p(k-1)+(p(k)-p(k-1))*(0.0-b1)/(b2-b1)
          frac = b2/(b2-b1)
          parea =  0.5*b2*dz*frac
          narea = narea-0.5*b1*dz*(1.0-frac)
          cin  = cin  + narea
          narea = 0.0
        elseif( (b2.lt.0.0) .and. (b1.gt.0.0) )then
          ! first trip into neg area
          ps = p(k-1)+(p(k)-p(k-1))*(0.0-b1)/(b2-b1)
          frac = b1/(b1-b2)
          parea =  0.5*b1*dz*frac
          narea = -0.5*b2*dz*(1.0-frac)
        elseif( b2.lt.0.0 )then
          ! still collecting negative buoyancy
          parea =  0.0
          narea = narea-0.5*dz*(b1+b2)
        else
          ! still collecting positive buoyancy
          parea =  0.5*dz*(b1+b2)
          narea =  0.0
        endif

        cape = cape + max(0.0,parea)
        pc(k) = cape

        if (p[k] <= 10000. and b2 <= 0.):
            break

    return cape, cin, zlcl, zlfc, zel


"""
End getcape.py
"""