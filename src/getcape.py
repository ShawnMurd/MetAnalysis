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

def getcape(p, T, qv, source='sfc', adiabat=1, ml=500.0, pinc=10.0):
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
    ml : float, optional
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
    Credit to George Bryan (NCAR) for writing the original getcape.F subroutine.
    
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

    # Compute derived quantities
    
    pi = isf.exner(p)
    th = isf.theta(T, p)
    thv = isf.thetav(T, p, qv)
    z = isf.sounding_height(p, th, qv, 0.0)

    # Determine initial parcel properties

!----------------------------------------------------------------

!---- find source parcel ----!

      IF(source.eq.1)THEN
        kmax = 1

      ELSEIF(source.eq.2)THEN

        IF(p(1).lt.50000.0)THEN
          kmax = 1
          maxthe = getthx(p(1),t(1),td(1),q(1))
        ELSE
          maxthe = 0.0
          do k=1,nk
            if(p(k).ge.50000.0)then
              the = getthx(p(k),t(k),td(k),q(k))
              if( the.gt.maxthe )then
                maxthe = the
                kmax = k
              endif
            endif
          enddo
        ENDIF

      ELSEIF(source.eq.3)THEN

        IF( (z(2)-z(1)).gt.ml_depth )THEN

          avgth = th(1)
          avgqv = q(1)
          kmax = 1

        ELSEIF( z(nk).lt.ml_depth )THEN
          ! the top-most level is within the mixed layer:  just use the
          ! upper-most level

          avgth = th(nk)
          avgqv = q(nk)
          kmax = nk

        ELSE
          ! calculate the mixed-layer properties:

          avgth = 0.0
          avgqv = 0.0
          k = 2

          do while( (z(k).le.ml_depth) .and. (k.le.nk) )

            avgth = avgth + 0.5*(z(k)-z(k-1))*(th(k)+th(k-1))
            avgqv = avgqv + 0.5*(z(k)-z(k-1))*(q(k)+q(k-1))

            k = k + 1

          enddo

          th2 = th(k-1)+(th(k)-th(k-1))*(ml_depth-z(k-1))/(z(k)-z(k-1))
          qv2 =  q(k-1)+( q(k)- q(k-1))*(ml_depth-z(k-1))/(z(k)-z(k-1))

          avgth = avgth + 0.5*(ml_depth-z(k-1))*(th2+th(k-1))
          avgqv = avgqv + 0.5*(ml_depth-z(k-1))*(qv2+q(k-1))

          avgth = avgth/ml_depth
          avgqv = avgqv/ml_depth

          kmax = 1

        ENDIF

      ELSE

        print *
        print *,'  Unknown value for source'
        print *
        print *,'  source = ',source
        print *
        stop

      ENDIF

!---- define parcel properties at initial location ----!
      narea = 0.0

      if( (source.eq.1).or.(source.eq.2) )then
        k    = kmax
        th2  = th(kmax)
        pi2  = pi(kmax)
        p2   = p(kmax)
        t2   = t(kmax)
        thv2 = thv(kmax)
        qv2  = q(kmax)
        b2   = 0.0
      elseif( source.eq.3 )then
        k    = kmax
        th2  = avgth
        qv2  = avgqv
        thv2 = th2*(1.0+reps*qv2)/(1.0+qv2)
        pi2  = pi(kmax)
        p2   = p(kmax)
        t2   = th2*pi2
        b2   = g*( thv2-thv(kmax) )/thv(kmax)
      endif

      psource = p2
      tsource = t2
      qvsource = qv2

      ql2 = 0.0
      qi2 = 0.0
      qt  = qv2

      cape = 0.0
      cin  = 0.0
      lfc  = 0.0

      doit = .true.
      cloud = .false.
      if(adiabat.eq.1.or.adiabat.eq.2)then
        ice = .false.
      else
        ice = .true.
      endif

      the = getthx(p2,t2,t2,qv2)
      pt(k) = t2
      if( cloud )then
        ptd(k) = t2
      else
        ptd(k) = gettd(p2,t2,qv2)
      endif
      ptv(k) = t2*(1.0+reps*qv2)/(1.0+qv2)
      pb(k) = 0.0
      pqv(k) = qv2
      pql(k) = 0.0

      zlcl = -1.0
      zlfc = -1.0
      zel  = -1.0

!---- begin ascent of parcel ----!

      do while( doit .and. (k.lt.nk) )

        k = k+1
        b1 =  b2

        dp = p(k-1)-p(k)

        if( dp.lt.pinc )then
          nloop = 1
        else
          nloop = 1 + int( dp/pinc )
          dp = dp/float(nloop)
        endif

        do n=1,nloop

          p1 =  p2
          t1 =  t2
          pi1 = pi2
          th1 = th2
          qv1 = qv2
          ql1 = ql2
          qi1 = qi2
          thv1 = thv2

          p2 = p2 - dp
          pi2 = (p2*rp00)**rddcp

          thlast = th1
          i = 0
          not_converged = .true.

          do while( not_converged )
            i = i + 1
            t2 = thlast*pi2
            if(ice)then
              fliq = max(min((t2-233.15)/(273.15-233.15),1.0),0.0)
              fice = 1.0-fliq
            else
              fliq = 1.0
              fice = 0.0
            endif
            qv2 = min( qt , fliq*getqvl(p2,t2) + fice*getqvi(p2,t2) )
            qi2 = max( fice*(qt-qv2) , 0.0 )
            ql2 = max( qt-qv2-qi2 , 0.0 )

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

        if( (p(k).le.10000.0).and.(b2.lt.0.0) )then
          ! stop if b < 0 and p < 100 mb
          doit = .false.
        endif

      enddo

!!!    print *,'  zlcl,zlfc,zel = ',zlcl,zlfc,zel
!!!    print *,'  plcl,plfc,pel = ',plcl,plfc,pel

!---- All done ----!

      end subroutine getcape

!-----------------------------------------------------------------------
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!-----------------------------------------------------------------------

      function getqvl(p,t)
      implicit none

      real :: p,t
      real :: getqvl

      real :: es
      real, parameter :: eps = 287.04/461.5

      es = 611.2*exp(17.67*(t-273.15)/(t-29.65))
    ! 171023 (fix for very cold temps):
      es = min( es , p*0.5 )
      getqvl = eps*es/(p-es)

      end function getqvl

!-----------------------------------------------------------------------
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!-----------------------------------------------------------------------

      function getqvi(p,t)
      implicit none

      real :: p,t
      real :: getqvi

      real :: es
      real, parameter :: eps = 287.04/461.5

      es = 611.2*exp(21.8745584*(t-273.15)/(t-7.66))
    ! 171023 (fix for very cold temps):
      es = min( es , p*0.5 )
      getqvi = eps*es/(p-es)

      end function getqvi

!-----------------------------------------------------------------------
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!-----------------------------------------------------------------------

      function getthx(p,t,td,q)
      implicit none

      real :: p,t,td,q
      real :: getthx
      real :: tlcl

      if( (td-t).ge.-0.1 )then
        tlcl = t
      else
        tlcl = 56.0 + ( (td-56.0)**(-1) + 0.00125*alog(t/td) )**(-1)
      endif

      getthx=t*( (100000.0/p)**(0.2854*(1.0-0.28*q)) )   
     $        *exp( ((3376.0/tlcl)-2.54)*q*(1.0+0.81*q) )

      end function getthx

!-----------------------------------------------------------------------
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!-----------------------------------------------------------------------

      function gettd(p,t,q)
      implicit none

      real :: p,t,q
      real :: gettd

      real :: el
      real, parameter :: eps = 287.04/461.5

      el = alog((q/eps)*p/100.0/(1.0+(q/eps)))
      gettd = 273.15+(243.5*el-440.8)/(19.48-el)

      end function gettd

!-----------------------------------------------------------------------
!ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!-----------------------------------------------------------------------

      function getq(p,t,the)
      implicit none

      real :: p,t,the
      real :: getq

! p (Pa), t (K), the (K), getq (kg/kg)

      real :: qhigh,qlow,qmid
      real :: tdhigh,tdlow,tdmid
      real :: thehigh,thelow,themid
      real :: getqvl,gettd,getthx
      integer :: i

      qhigh = getqvl(p,t)
      qlow = 0.0001
      qmid = 0.5*(qhigh+qlow)

      tdhigh = gettd(p,t,qhigh)
      tdlow = gettd(p,t,qlow)
      tdmid = gettd(p,t,qmid)

      thehigh = getthx(p,t,tdhigh,qhigh)
      thelow = getthx(p,t,tdlow,qlow)
      themid = getthx(p,t,tdmid,qmid)

      i = 0
      do while (abs(themid-the).gt.0.01)
        
        if ((thehigh-the)*(themid-the).lt.0) then
          qlow = qmid
          tdlow = tdmid
          thelow = themid
        else
          qhigh = qmid
          tdhigh = tdmid
          thehigh = themid
        endif

        qmid = 0.5*(qhigh+qlow)
        tdmid = gettd(p,t,qmid)
        themid = getthx(p,t,tdmid,qmid)

        i = i+1
        if (i.gt.100) then
          print *, 'max number of iterations (100) reached'
          stop
        endif

      enddo

      getq = qmid

      end function getq


