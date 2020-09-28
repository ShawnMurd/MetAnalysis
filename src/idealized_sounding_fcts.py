"""
Idealized Soundings Functions

Functions to create the McCaul and Weisman (2001) analytic soundings.

The functions used herein compute CAPE, CIN, LCL, etc. using a the formulation found in getcape.F
within George Bryan's CM1 (cm1r19.8).

Shawn Murdzek
sfm5282@psu.edu
Date Created: October 10, 2019
Environment: local_py (Python 3.6)
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd

import metpy.calc as mc
from metpy.units import units

import MetAnalysis.src.getcape as gc
import MetAnalysis.src.getB as gB


#---------------------------------------------------------------------------------------------------
# Define Basic Thermodynamic Functions
#---------------------------------------------------------------------------------------------------

def exner(p):
    """
    Compute the Exner function
    Reference:
        Markowski and Richardson (2010) Chpt 2, footnote 8 (pg 20)
    Inputs:
        p = Pressure (Pa)
    Outputs:
        pi = Exner function (unitless)
    """

    rd = 287.04
    cp = 1005.7
    p00 = 100000.0

    pi = (p / p00) ** (rd / cp)

    return pi


def theta(T, p):
    """
    Compute potential temperature
    Reference:
        Markowski and Richardson (2010) eqn 2.7
    Inputs:
        T = Temperature (K)
        p = Pressure (Pa)
    Outputs:
        theta = Potential temperature  (K)
    """
    
    return T / exner(p)


def getTfromTheta(theta, p):
    """
    Compute the temperature using the potential temperature and pressure
    Inputs:
        theta = Potential temperature (K)
        p = Pressure (Pa)
    Outputs:
        T = Temperature (K)
    """
    
    return theta * exner(p)


def get_es(T):
    """
    Compute equilibrium vapor pressure (over liquid water).
    Reference:
        Markowski and Richardson (2010) eqn 2.16
    Inputs:
        T = Temperature (K)
    Outputs:
        e_s = equilibrium vapor pressure over liquid water (Pa)
    """
  
    T = T - 273.15
    e_s = 611.2 * np.exp(17.67 * T / (T + 243.5))

    return e_s


def get_qvl(T, p):
    """
    Compute equilibrium water vapor mass mixing ratio (over liquid water).
    Inputs:
        T = Temperature (K)
        p = Pressure (Pa)
    Outputs:
        qvl = Equilibrium water vapor mass mixing ratio (kg / kg)
    """

    Rv = 461.5
    Rd = 287.04
    eps = Rd / Rv
    es = get_es(T)
    qvl = eps * es / (p - es)

    return qvl


def getTv(T, qv):
    """
    Compute virtual tempertaure
    Reference:
        Markowski and Richardson (2010) eqn 2.19
    Inputs:
        T = Temperature (K)
        qv = Water vapor mass mixing ratio (kg / kg)
    Outputs:
        Tv = Virtual potential temperature (K)
    """
    
    Rv = 461.5
    Rd = 287.04
    
    Tv = T * (1.0 + ((Rv / Rd) * qv)) / (1.0 + qv)
    
    return Tv


def thetav(T, p, qv):
    """
    Compute virtual potential temperature
    Reference:
        Markowski and Richardson (2010) eqn 2.20
    Inputs:
        T = Temperature (K)
        p = Pressure (Pa)
        qv = Water vapor mass mixing ratio (kg / kg)
    Outputs:
        thetav = Virtual potential temperature (K)
    """
    
    return getTv(theta(T, p), qv)


def getTfromTv(Tv, qv):
    """
    Compute the temperature from the virtual temperature
    Inputs:
        Tv = Virtual temperature (K)
        qv = Water vapor mass mixing ratio (kg / kg)
    Outputs:
        T = Temperature (K)
    """
    
    Rv = 461.5
    Rd = 287.04
    
    T = Tv * (1.0 + qv) / (1.0 + ((Rv / Rd) * qv))
    
    return T


def buoy(T_p, p_p, qv_p, T_env, p_env, qv_env):
    """
    Compute buoyancy
    Reference:
        Markowski and Richardson (2010) eqn 2.78
    Inputs:
        T_p = Parcel temperature (K)
        p_p = Parcel pressure (Pa)
        qv_p = Parcel water vapor mass mixing ratio (kg / kg)
        T_env = Environmental temperature (K)
        p_env = Environmental pressure (Pa)
        qv_env = Environmental water vapor mass mixing ratio (kg / kg)
    Outputs:
        B = Buoyancy (m / s^2)
    """
    
    g = 9.81
    
    thetav_p = thetav(T_p, p_p, qv_p)
    thetav_env = thetav(T_env, p_env, qv_env)
    
    B = g * (thetav_p - thetav_env) / thetav_env
    
    return B


#---------------------------------------------------------------------------------------------------
# Define Function to Print Environmental Parameters
#---------------------------------------------------------------------------------------------------

def print_env_param(T, p, qv, print_results=True, adiabat=1):
    """
    Print various thermodynamic environmental parameters (e.g., CAPE, CIN, LFC) using getcape.
    Inputs:
        T = Environmental temperature profile (K)
        p = Environmental pressure profile (Pa)
        qv = Environmental water vapor mixing ratio profile (kg / kg)
    Outputs:
        param_dict = Dictionary containing the same environmental parameters printed to the screen
    Keywords:
        print_results = Option to print parameters
        adiabat = Adiabat option for getcape
            1: Pseudoadiabatic, liquid only
            2: Reversible, liquid only
            3: Pseudoadiabatic, with ice
            4: Reversible, with ice
    """
    
    # Put T and p in correct units (deg C and hPa)
    
    T = T - 273.15
    p = p / 100.0
    
    # Compute sounding parameters using getcape and fill param_dict
    
    param_dict = {}
    params = ['CAPE', 'CIN', 'LCL', 'LFC', 'EL']
    units = ['J / kg', 'J / kg', 'm', 'm', 'm']
    
    for i, p in enumerate(['SB', 'MU', 'ML']):
        out = gc.getcape(i+1, adiabat, p, T, qv)
        for j, param in enumerate(params):
            param_dict[p + param] = out[j]
    
    # Print results to screen
    
    if print_results:
        for p in ['SB', 'MU', 'ML']:
            for k, (param, u) in enumerate(zip(params, units)):
                print('%s%s = %.2f %s' % (p, param, param_dict[p + param], u))
            print()
        
    return param_dict


#---------------------------------------------------------------------------------------------------
# Define Functions Related to Vertical Profiles of Sounding Parameters
#---------------------------------------------------------------------------------------------------

def sounding_pressure(z, th, qv, p0):
    """
    Computes the pressure profile for the given height, temperature, and water vapor mixing ratio
    profile using an upward integration of the hydrostatic balance equation.
    Inputs:
        z = Sounding heights (m)
        th = Sounding potential temperatures (K)
        qv = Sounding water vapor mass mixing ratios (kg / kg)
        p0 = Pressure corresponding to z[0] (Pa)
    Outputs:
        p = Sounding pressure (Pa)
    """

    # Define constants

    reps = 461.5 / 287.04
    rd = 287.04
    cp = 1005.7
    p00 = 100000.0
    g = 9.81

    # Compute Exner function and virtual potential temperature

    pi = np.zeros(z.shape)
    pi[0] = exner(p0)
    thv = th * (1.0 + (reps * qv)) / (1.0 + qv)

    # Integrate hydrostatic equation upward from surface

    for i in range(1, z.size):
        pi[i] = pi[i-1] - g * (z[i] - z[i-1]) / (cp * 0.5 * (thv[i] + thv[i-1]))

    p = p00 * (pi ** (cp / rd))

    return p


def cm1_snd_helper(cm1_sounding):
    """
    Extracts the temperature, water vapor mass mixing ratio, pressure, height, and wind profiles 
    from a CM1 input sounding file.
    Inputs:
        cm1_sounding = CM1 input sounding file
    Outputs:
        out_df = DataFrame with:
            temperature (K)
            water vapor mass mixing ratio (kg / kg)
            pressure (Pa)
            height (m)
            u wind component (m / s)
            v wind component (m / s)
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
    Create input .dat file for Kerry Emanuel's calcsound program.
    Inputs:
        p = Pressure (Pa)
        T = Temperature (K)
        qv = Water vapor mass mixing ratio (kg / kg)
        fname = File to save calcsound input file to (including path)
    Outputs:
        None, creates input file in the directory specified by path
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
    Read in a .out file from Kerry Emanuel's calcsound program as a pandas DataFrame.
    Inputs:
        out = Output file from Kerry Emanuel's calcsound program
    Outputs:
        out_df = DataFrame containing information from out
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
    Determine the effective inflow layer using the empirical definition of Thompson et al. 
    (2007, WAF).
    Inputs:
        p = Pressure (Pa)
        T = Temperature (K)
        qv = Water vapor mass mixing ratio (kg / kg)
    Outputs:
        p_top = Pressure at top of effective inflow layer (Pa)
        p_bot = Pressure at bottom of effective inflow layer (Pa)
    Keywords:
        min_cape = Minimum amount of CAPE a parcel must have to be in the inflow layer (J / kg)
        max_cin = Maximum amount of CIN a parcel can have to be in the inflow layer (J / kg)
        adiabat = Adiabat option for getcape
            1: Pseudoadiabatic, liquid only
            2: Reversible, liquid only
            3: Pseudoadiabatic, with ice
            4: Reversible, with ice
    """
    
    # Put p, T, and q in correct units
    
    p = p / 100.0
    T = T - 273.15
    
    # Determine bottom of effective inflow layer
    
    i = 0
    cape, cin, _, _, _, _, _, _ = gc.getcape(1, adiabat, p[i:], T[i:], qv[i:])
    while (cape < min_cape) or (cin > max_cin):
        cape, cin, _, _, _, _, _, _ = gc.getcape(1, adiabat, p[i:], T[i:], qv[i:])
        i = i + 1
    p_bot = p[i] * 100
    
    # Determine top of effective inflow layer
    
    while (cape > min_cape) and (cin < max_cin):
        cape, cin, _, _, _, _, _, _ = gc.getcape(1, adiabat, p[i:], T[i:], qv[i:])
        i = i + 1
    p_top = p[i] * 100
    
    return p_top, p_bot


def param_vprof(p, T, qv, zbot, ztop, adiabat=1, ric=0, rjc=0, zc=1.5, bhrad=10.0, bvrad=1.5,
                bptpert=0.0, maintain_rh=False, xloc=0.0, yloc=0.0):
    """
    Compute vertical profiles of sounding parameters for a CM1 input sounding.
    Inputs:
        p = Pressure (Pa)
        T = Temperature (K)
        qv = Water vapor mass mixing ratio (kg / kg)
        pbot = Bottom of layer to compute sounding parameters (Pa)
        ptop = Top of layer to compute sounding parameters (Pa)
    Outputs:
        param_df = Sounding parameters dictionary
        B = 2D array of parcel buoyancies (m / s^2)
    Keywords:
        adiabat = Adiabat option for getcape
            1: Pseudoadiabatic, liquid only
            2: Reversible, liquid only
            3: Pseudoadiabatic, with ice
            4: Reversible, with ice
        ric, rjc, zc = Center of initiating warm bubble (km)
        bhrad, bvrad = Horizontal and vertical radius of initiating warm bubble (K)
        bptpert = Warm buble perturbation (set to 0 to plot parameters without warm bubble) (K)
        maintain_rh = Keep constant RH in initiating warm bubble
        xloc, yloc = Horizontal location of vertical profile (km)
    """
   
    th = theta(T, p)
    pi = exner(p)

    # Add initiating warm bubble

    beta = np.sqrt(((xloc - ric) / bhrad) ** 2.0 +
                   ((yloc - rjc) / bhrad) ** 2.0 +
                   (((z / 1000) - zc) / bvrad) ** 2.0)

    thpert = np.zeros(th.shape)
    inds = np.where(beta < 1.0)[0]
    thpert[inds] = bptpert * (np.cos(0.5 * np.pi * beta[inds]) ** 2.0)

    if maintain_rh:
        rh = qv / get_qvl(th * pi, p)
        qv = rh * get_qvl(th + thpert) * pi, p)

    th = th * thpert
    T = th * pi
    
    # Determine indices for pbot and ptop

    ibot = np.argmin(np.abs(p - pbot))
    itop = np.argmin(np.abs(p - ptop))
    
    # Put p, T, and q in correct units
    
    p = p / 100.0
    T = T - 273.15
    
    # Initialize output dictionary
    
    nlvls = len(p[ibot:itop+1])

    param_dict = {}
    params['p'] = p[ibot:itop+1]
    params = ['CAPE', 'CIN', 'zlcl', 'zlfc', 'zel']    
    for p in params:
        params[p] = np.zeros(nlvls)
    
    B = np.empty((nlvls, p.size))
    B[:, :] = np.nan
    
    # Loop through each vertical level
    
    for i in range(ibot, itop+1):
        out = gB.getcape(1, adiabat, p[i:], T[i:], qv[i:])
        for j, p in enumerate(params):
            params[p][i] = out[j]
        B[i, :len(out[5])] = out[5]
    
    param_df = pd.DataFrame.from_dict(param_dict)
    
    return param_df, B


def param_vprof_MW_df(MW_df, zbot, ztop, adiabat=1):
    """
    Compute vertical profiles of sounding parameters for an output DataFrame from the 
    mccaul_weisman() function
    Inputs:
        MW_df = Output DataFrame from the mccaul_weisman() function
        zbot = Bottom of layer to compute sounding parameters (m)
        ztop = Top of layer to compute sounding parameters (m)
    Outputs:
        param_df = Sounding parameters dictionary
        B = 2D array of parcel buoyancies (m / s^2)
        z = Heights (m)
    Keywords:
        adiabat = Adiabat option for getcape
            1: Pseudoadiabatic, liquid only
            2: Reversible, liquid only
            3: Pseudoadiabatic, with ice
            4: Reversible, with ice
    """

    # Extract T, p, and qv (in correct units)

    p = MW_df['prs'].values / 100.0
    T = MW_df['T'].values - 273.15
    qv = MW_df['qv'].values
    z = MW_df['z'].values

    # Initialize output dictionary

    ibot = np.argmin(np.abs(z - zbot))
    itop = np.argmin(np.abs(z - ztop))

    nlvls = len(z[ibot:itop+1])
    params = {}

    params['z'] = z[ibot:itop+1]
    params['CAPE'] = np.zeros(nlvls)
    params['CIN'] = np.zeros(nlvls)
    params['zlcl'] = np.zeros(nlvls)
    params['zlfc'] = np.zeros(nlvls)
    params['zel'] = np.zeros(nlvls)

    B = np.empty((nlvls, z.size))
    B[:, :] = np.nan

    # Loop through each vertical level

    for i in range(ibot, itop+1):

        cape, cin, zlcl, zlfc, zel, b, ps, ts, qvs = gB.getcape(1, adiabat, p[i:], T[i:], qv[i:])
        params['CAPE'][i] = cape
        params['CIN'][i] = cin
        params['zlcl'][i] = zlcl
        params['zlfc'][i] = zlfc
        params['zel'][i] = zel
        B[i, :len(b)] = b

    param_df = pd.DataFrame.from_dict(params)

    return param_df, B, z


#---------------------------------------------------------------------------------------------------
# Add Weisman-Klemp Sounding Here
#---------------------------------------------------------------------------------------------------

def weisman_klemp():
    return None

#---------------------------------------------------------------------------------------------------
# Functions for McCaul-Weisman Analytic Sounding
#---------------------------------------------------------------------------------------------------

def dewpt_from_thetae(thetae, p, T, thetae_tol=(0.005 * units.kelvin), max_iter=50):
    """
    Compute the dewpoint from the equivalent potential tempertaure, pressure, and temperature using
    an iterative midpoint method.
    Inputs:
        thetae = Equivalent potential temperature (K)
        p = Atmospheric pressure (Pa)
        T = Temperature (K)
    Outputs:
        Td = Dewpoint (K)
    Keywords:
        thetae_tol = Maximum allowable difference between the input thetae and thetae computed
            using Td (K)
        max_iter = Maximum number of iterations
    """
    
    # Choose initial dewpoint values
    
    Td_high = T
    Td_low = 0.25 * T
    
    # Initial set up for midpoint method
    
    Td_mid = 0.5 * (Td_high + Td_low)
    
    thetae_high = mc.equivalent_potential_temperature(p, T, Td_high)
    thetae_low = mc.equivalent_potential_temperature(p, T, Td_low)
    thetae_mid = mc.equivalent_potential_temperature(p, T, Td_mid)
    
    # Check to see if given theta-e value can actually be reached
    
    if (thetae_high < thetae) or (thetae_low > thetae):
        
        raise ValueError('The specified theta-e value (%.2f) cannot be reached' % 
                         thetae.magnitude)
    
    # Perform midpoint method
    
    i = 0
    while np.abs(thetae_mid - thetae) > thetae_tol:
        
        if (thetae_high - thetae) * (thetae_mid - thetae) < (0 * units.kelvin * units.kelvin):
        
            Td_low = Td_mid
            Td_mid = 0.5 * (Td_low + Td_high)
        
            thetae_low = thetae_mid
            thetae_mid = mc.equivalent_potential_temperature(p, T, Td_mid)
        
        else:
        
            Td_high = Td_mid
            Td_mid = 0.5 * (Td_low + Td_high)
        
            thetae_high = thetae_mid
            thetae_mid = mc.equivalent_potential_temperature(p, T, Td_mid)
        
        i = i + 1
    
        if i > max_iter:
        
            warnings.warn('Max number of iterations (%s) has been reached.' % max_iter)
            break
            
    return Td_mid


def create_pbl(thetae, T_sfc, p_sfc, dz, lapse_rate=(0.0085 * units.kelvin / units.meter),
               thetae_tol=(0.005 * units.kelvin), thetae_max_iter=50, depth=None):
    """
    Construct the PBL of an atmospheric sounding using a constant theta-e value and hydrostatic
    balance. Above the LCL, a constant theta-e, constant RH profile is used (similar to McCaul and
    Cohen 2002).
    Inputs:
        thetae = Constant equivalent potential temperature values in the PBL (K)
        T_sfc = Surface temperature (K)
        p_sfc = Surface pressure (Pa)
        dz = Vertical grid spacing (m)
    Outputs:
        z_prof = 1D array of heights in the sub-LCL layer (m)
        p_prof = 1D array of pressures in the sub-LCL layer (Pa)
        T_prof = 1D array of temperatures in the sub-LCL layer (K)
        Td_prof = 1D array of dewpoints in the sub-LCL layer (K)
        z_lcl = LCL height AGL (m)
    Keywords:
        lapse_rate = PBL lapse rate (K / m)
        thetae_tol = Thetae tolerance for dewpt_from_thetae (K)
        thetae_max_iter = Maximum number of iterations for dewpt_from_thetae
        depth = Height of PBL (m). If None, PBL is terminated at LCL
    """

    # Extract necessary constants from MetPy

    Rd = const.Rd.to(units.joule / units.kilogram / units.kelvin).magnitude
    g = const.g.magnitude

    # Determine surface dewpoint

    Td_sfc = dewpt_from_thetae(thetae, p_sfc, T_sfc, thetae_tol=thetae_tol,
                               max_iter=thetae_max_iter)

    # Determine surface virtual temperature

    RH_sfc = mc.relative_humidity_from_dewpoint(T_sfc, Td_sfc)
    qv_sfc = mc.mixing_ratio_from_relative_humidity(RH_sfc, T_sfc, p_sfc)
    Tv = getTv(T_sfc, qv_sfc).magnitude

    # Determine LCL temperature and pressure

    p_lcl, T_lcl = mc.lcl(p_sfc, T_sfc, Td_sfc)

    # Determine LCL height via upward integration of hydrostatic balance eqn

    p_prof = [p_sfc.magnitude]
    T_prof = [T_sfc.magnitude]
    Td_prof = [Td_sfc.magnitude]
    qv_prof = [qv_sfc.magnitude]
    z_prof = [0]

    i = 0
    while p_prof[i] > p_lcl.magnitude:

        i = i + 1

        # Compute p and T at next level using hydrostatic balance

        p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz.magnitude) / (Rd * Tv)))
        T_prof.append(T_prof[i-1] - (lapse_rate.magnitude * dz.magnitude))
        z_prof.append(z_prof[i-1] + dz.magnitude)

        # Update qv by forcing thetae to be constant

        Td_prof.append(dewpt_from_thetae(thetae, p_prof[i] * units.pascal, T_prof[i] * units.kelvin,
                                         thetae_tol=thetae_tol,
                                         max_iter=thetae_max_iter).magnitude)

        RH = mc.relative_humidity_from_dewpoint(T_prof[i] * units.kelvin,
                                                Td_prof[i] * units.kelvin)
        qv_prof.append(mc.mixing_ratio_from_relative_humidity(RH, T_prof[i] * units.kelvin,
                                                              p_prof[i] * units.pascal))
        Tv = getTv(T_prof[i] * units.kelvin, qv_prof[i]).magnitude

    z_lcl = z_prof[-1]

    # Create constant theta-e layer above LCL using an upward integration of the hydrostatic
    # balance eqn, if desired

    if depth != None:

        the_adjust = 1.0 * units.kelvin
        T_adjust = 0.00
        print(the_adjust)
        print(T_adjust)

        while z_prof[i] < depth:

            i = i + 1
            p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz.magnitude) / (Rd * Tv)))

            # Have T decrease at the PBL lapse rate until T_lcl is reached

            if T_prof[i-1] > (T_lcl.magnitude - T_adjust):
                T_prof.append(T_prof[i-1] - (lapse_rate.magnitude * dz.magnitude))
                if T_prof[i] < (T_lcl.magnitude - T_adjust):
                    T_prof[i] = T_lcl.magnitude - T_adjust
            else:
                T_prof.append(mc.moist_lapse(np.array(p_prof[i-1:]) * units.pascal,
                                             T_prof[i-1] * units.kelvin).magnitude[-1])

            # Update qv by forcing thetae to be constant

            Td_prof.append(dewpt_from_thetae(thetae - the_adjust, p_prof[i] * units.pascal, 
                                             T_prof[i] * units.kelvin,
                                             thetae_tol=thetae_tol,
                                             max_iter=thetae_max_iter).magnitude)

            RH = mc.relative_humidity_from_dewpoint(T_prof[i] * units.kelvin,
                                                    Td_prof[i] * units.kelvin)
            qv_prof.append(mc.mixing_ratio_from_relative_humidity(RH, T_prof[i] * units.kelvin,
                                                                  p_prof[i] * units.pascal))
            Tv = getTv(T_prof[i] * units.kelvin, qv_prof[i]).magnitude

            z_prof.append(z_prof[i-1] + dz.magnitude)

    # Turn lists into array

    z_prof = np.array(z_prof) * units.meter
    p_prof = np.array(p_prof) * units.pascal
    T_prof = np.array(T_prof) * units.kelvin
    Td_prof = np.array(Td_prof) * units.kelvin

    return z_prof, p_prof, T_prof, Td_prof, z_lcl


def create_pbl_20200304(thetae, T_sfc, p_sfc, dz, lapse_rate=(0.0085 * units.kelvin / units.meter),
                        thetae_tol=(0.005 * units.kelvin), depth=None, lr=0.0001):
    """
    Construct the PBL of an atmospheric sounding using a constant theta-e value and hydrostatic
    balance. Above the LCL, a constant theta-e layer with a lapse rate slightly less than the moist
    adibatic lapse rate is used (similar to McCaul and Cohen 2002).
    Inputs:
        thetae = Constant equivalent potential temperature values in the PBL (K)
        T_sfc = Surface temperature (K)
        p_sfc = Surface pressure (Pa)
        dz = Vertical grid spacing (m)
    Outputs:
        z_prof = 1D array of heights in the sub-LCL layer (m)
        p_prof = 1D array of pressures in the sub-LCL layer (Pa)
        T_prof = 1D array of temperatures in the sub-LCL layer (K)
        Td_prof = 1D array of dewpoints in the sub-LCL layer (K)
        z_lcl = LCL height AGL (m)
    Keywords:
        lapse_rate = PBL lapse rate (K / m)
        thetae_tol = Thetae tolerance for dewpt_from_thetae (K)
        depth = Height of PBL (m). If None, PBL is terminated at LCL
        lr = Lapse rate in LFC-LCL layer is equal to the MALR - lr (K / m)
    """

    # Extract necessary constants from MetPy

    Rd = const.Rd.to(units.joule / units.kilogram / units.kelvin).magnitude
    g = const.g.magnitude

    # Determine surface dewpoint

    qv_sfc = gc.getq(p_sfc.magnitude, T_sfc.magnitude, thetae.magnitude)
    Td_sfc = gc.gettd(p_sfc.magnitude, T_sfc.magnitude, qv_sfc) * units.kelvin

    # Determine surface virtual temperature

    Tv = getTv(T_sfc, qv_sfc).magnitude

    # Determine LCL temperature and pressure

    p_lcl, T_lcl = mc.lcl(p_sfc, T_sfc, Td_sfc)

    # Determine LCL height via upward integration of hydrostatic balance eqn

    p_prof = [p_sfc.magnitude]
    T_prof = [T_sfc.magnitude]
    Td_prof = [Td_sfc.magnitude]
    qv_prof = [qv_sfc]
    z_prof = [0]

    i = 0
    while p_prof[i] > p_lcl.magnitude:

        i = i + 1

        # Compute p and T at next level using hydrostatic balance

        p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz.magnitude) / (Rd * Tv)))
        T_prof.append(T_prof[i-1] - (lapse_rate.magnitude * dz.magnitude))
        z_prof.append(z_prof[i-1] + dz.magnitude)

        # Update qv by forcing thetae to be constant

        qv_prof.append(gc.getq(p_prof[i], T_prof[i], thetae.magnitude))
        Td_prof.append(gc.gettd(p_prof[i], T_prof[i], qv_prof[i]))
        Tv = getTv(T_prof[i], qv_prof[i])

    z_lcl = z_prof[-1]
    
    # Create constant theta-e layer above LCL using an upward integration of the hydrostatic
    # balance eqn, if desired

    if depth != None:
        
        T_adjust = lr * dz.magnitude

        while z_prof[i] < depth:

            i = i + 1
            p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz.magnitude) / (Rd * Tv)))

            # Have T decrease at MALR - lr
            
            T_prof.append(mc.moist_lapse(np.array(p_prof[i-1:]) * units.pascal,
                                         T_prof[i-1] * units.kelvin).magnitude[-1] + T_adjust)
            qv_prof.append(gc.getq(p_prof[i], T_prof[i], thetae.magnitude))
            Td_prof.append(gc.gettd(p_prof[i], T_prof[1], qv_prof[i]))
            
            # Update Tv

            Tv = getTv(T_prof[i], qv_prof[i])

            z_prof.append(z_prof[i-1] + dz.magnitude)

    # Turn lists into array

    z_prof = np.array(z_prof) * units.meter
    p_prof = np.array(p_prof) * units.pascal
    T_prof = np.array(T_prof) * units.kelvin
    Td_prof = np.array(Td_prof) * units.kelvin

    return z_prof, p_prof, T_prof, Td_prof, z_lcl


def create_pbl_new(thetae, T_sfc, p_sfc, dz, lapse_rate=(0.0085 * units.kelvin / units.meter), 
                   thetae_tol=(0.005 * units.kelvin), thetae_max_iter=50, depth=None, 
                   const_the_opt=1, the_b=0.15, the_m=0.25):
    """
    Construct the PBL of an atmospheric sounding using a constant theta-e value and hydrostatic 
    balance. Above the LCL, a constant theta-e, constant RH profile is used (similar to McCaul and
    Cohen 2002). The formulation of the constant theta-e layer between the LCL and LFC used here
    produces layers with CIN = 0 and CAPE > 1000 J / kg (particularly for low LCLs and high PBL
    depths), so this method is NOT recommended.
    Inputs:
        thetae = Constant equivalent potential temperature values in the PBL (K)
        T_sfc = Surface temperature (K)
        p_sfc = Surface pressure (Pa)
        dz = Vertical grid spacing (m)
    Outputs:
        z_prof = 1D array of heights in the sub-LCL layer (m)
        p_prof = 1D array of pressures in the sub-LCL layer (Pa)
        T_prof = 1D array of temperatures in the sub-LCL layer (K)
        Td_prof = 1D array of dewpoints in the sub-LCL layer (K)
        z_lcl = LCL height AGL (m)
    Keywords:
        lapse_rate = PBL lapse rate (K / m)
        thetae_tol = Thetae tolerance for dewpt_from_thetae (K)
        thetae_max_iter = Maximum number of iterations for dewpt_from_thetae
        depth = Height of PBL (m). If None, PBL is terminated at LCL
        const_the_opt = Option for constant-thetae layer
            1: Set T_env = T_parcel + the_b
            2: T_env = T_parcel + the_b + ((z - z_lcl) / depth) * the_m
    """
    
    # Extract necessary constants from MetPy
    
    Rd = const.Rd.to(units.joule / units.kilogram / units.kelvin).magnitude
    g = const.g.magnitude
    
    # Determine surface dewpoint
    
    Td_sfc = dewpt_from_thetae(thetae, p_sfc, T_sfc, thetae_tol=thetae_tol, 
                               max_iter=thetae_max_iter)
    
    # Determine surface virtual temperature
    
    RH_sfc = mc.relative_humidity_from_dewpoint(T_sfc, Td_sfc)
    qv_sfc = mc.mixing_ratio_from_relative_humidity(RH_sfc, T_sfc, p_sfc)
    Tv = getTv(T_sfc, qv_sfc).magnitude
    
    # Determine LCL temperature and pressure
    
    p_lcl, T_lcl = mc.lcl(p_sfc, T_sfc, Td_sfc)
    
    # Determine LCL height via upward integration of hydrostatic balance eqn
    
    p_prof = [p_sfc.magnitude]
    T_prof = [T_sfc.magnitude]
    Td_prof = [Td_sfc.magnitude]
    qv_prof = [qv_sfc.magnitude]
    z_prof = [0]
    
    i = 0
    while p_prof[i] > p_lcl.magnitude:
        i = i + 1
        
        # Compute p and T at next level using hydrostatic balance
        
        p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz.magnitude) / (Rd * Tv)))
        T_prof.append(T_prof[i-1] - (lapse_rate.magnitude * dz.magnitude))
        z_prof.append(z_prof[i-1] + dz.magnitude)
        
        # Update qv by forcing thetae to be constant
        
        Td_prof.append(dewpt_from_thetae(thetae, p_prof[i] * units.pascal, T_prof[i] * units.kelvin, 
                                         thetae_tol=thetae_tol, 
                                         max_iter=thetae_max_iter).magnitude)
        
        RH = mc.relative_humidity_from_dewpoint(T_prof[i] * units.kelvin, 
                                                Td_prof[i] * units.kelvin)
        qv_prof.append(mc.mixing_ratio_from_relative_humidity(RH, T_prof[i] * units.kelvin, 
                                                              p_prof[i] * units.pascal))
        Tv = getTv(T_prof[i] * units.kelvin, qv_prof[i]).magnitude
        
    z_lcl = z_prof[-1]
    RH_lcl = RH

    # Create a nearly-constant theta-e layer above the LCL by computing the parcel T
    # The environmental T is equal to the parcel T + 0.2 K
    
    if depth != None:
        
        while z_prof[i] < depth:
            
            i = i + 1
            p_prof.append(p_prof[i-1] - ((p_prof[i-1] * g * dz.magnitude) / (Rd * Tv)))
            
            # Update z
                
            z_prof.append(z_prof[i-1] + dz.magnitude)
            
            # Determine the parcel temperature at z_prof[i]

            p_array = np.array([p_sfc.magnitude, p_prof[i]]) * units.pascal
            T_parcel = mc.parcel_profile(p_array, T_sfc, Td_sfc)[-1]

            # Determine environmental temperature. 
            # Keep lapse rate from exceeding PBL lapse rate

            if const_the_opt == 1:            
                T_env = T_parcel.magnitude + the_b
            elif const_the_opt == 2:
                T_env = T_parcel.magnitude + the_b + (((z_prof[i] - z_lcl) / depth) * the_m)
            
            if ((T_prof[i-1] - T_env) / dz.magnitude > lapse_rate.magnitude):             
                T_prof.append(T_prof[i-1] - (lapse_rate * dz).magnitude)               
            else:              
                T_prof.append(T_env)
            
            # Update qv by having a constant RH (which is the RH at the LCL)
        
            Td_prof.append(mc.dewpoint_rh(T_prof[i] * units.kelvin, RH_lcl).magnitude + 273.15)
        
            qv_prof.append(mc.mixing_ratio_from_relative_humidity(RH_lcl, T_prof[i] * units.kelvin, 
                                                                  p_prof[i] * units.pascal))

    # Turn lists into array
    
    z_prof = np.array(z_prof) * units.meter
    p_prof = np.array(p_prof) * units.pascal
    T_prof = np.array(T_prof) * units.kelvin
    Td_prof = np.array(Td_prof) * units.kelvin
    z_lcl = z_lcl * units.meter
        
    return z_prof, p_prof, T_prof, Td_prof, z_lcl


def mccaul_weisman(z, E=2000.0, m=2.2, H=12500.0, z_trop=12000.0, RH_min=0.1, p_sfc=1e5,
                   T_sfc=300.0, thetae_pbl=335.0, pbl_lapse=0.009, crit_lapse=0.0095, 
                   pbl_depth=None, lr=0.0001):
    """
    Create an analytic thermodynamic profile following the method in the appendix of McCaul and 
    Weisman (2001) and using the algorithm written by Paul Markowski in saminit.f. Unlike McCaul and 
    Weisman (2001), the thermodynamic profile created by this function features relative humidity 
    values in the free troposphere that decrease linearly with height (similar to Warren et al. 
    2017). The virtual temperature correction is used when computing CAPE in this function.
    Inputs:
        z = Heights AGL to provide thermodynamic profile data (m)
    Outputs:
        thermo_prof = Pandas DataFrame containing the height (m), pressure (Pa), potential 
            temperature (K), water vapor mixing ratio (kg / kg), temperature (K), and 
            dewpoint (K)
    Keywords:
        E = Tv-corrected CAPE value (J / kg)
        m = Buoyancy profile compression parameter
        H = Vertical scale (m)
        z_trop = Height of tropopause (m)
        RH_min = Minimum RH at tropopause (decimal)
        p_sfc = Pressure at the surface (Pa)
        T_sfc = Temperature at the surface (K)
        thetae_pbl = Constant theta-e in PBL (K)
        pbl_lapse = Lapse rate below LCL (K / m)
        crit_lapse = Lapse rates greater than crit_lapse are set to the PBL lapse rate (K / m)
        pbl_depth = PBL depth (m). Set to None to use the LCL as the PBL top
        lr = Lapse rate in LCL-LFC layer is the MALR - lr (K / m)
    """
    
    # Define constants
    
    Rd = 287.04 * units.joule / units.kilogram / units.kelvin
    g = 9.81 * units.meter / units.second / units.second
    
    # Add units to appease MetPy
    
    p_sfc = p_sfc * units.pascal
    T_sfc = T_sfc * units.kelvin
    z = z * units.meter
    E = E * units.joule / units.kilogram
    z_trop = z_trop * units.meter
    pbl_lapse = pbl_lapse * units.kelvin / units.meter
    crit_lapse = crit_lapse * units.kelvin / units.meter
    thetae_pbl = thetae_pbl * units.kelvin
    
    # Initialize arrays
    
    Tv_env_prof = np.zeros(z.shape) * units.kelvin
    Tv_parcel_prof = np.zeros(z.shape) * units.kelvin
    p_prof = np.zeros(z.shape) * units.pascal
    qv_prof = np.zeros(z.shape)
    
    # Determine LCL height and create sub-LCL thermodynamic profile
    
    dz = z[1] - z[0]
    z_pbl, p_pbl, T_pbl, Td_pbl, lcl_z = create_pbl_20200304(thetae_pbl, T_sfc, p_sfc, dz, 
                                                             lapse_rate=pbl_lapse, 
                                                             depth=pbl_depth, lr=lr)
    
    pbl_top_ind = z_pbl.size
    pbl_z = z_pbl[-1]
    
    rh_pbl = mc.relative_humidity_from_dewpoint(T_pbl, Td_pbl)
    qv_prof[:pbl_top_ind] = mc.mixing_ratio_from_relative_humidity(rh_pbl, T_pbl, p_pbl)
    
    Tv_env_prof[:pbl_top_ind] = getTv(T_pbl, qv_prof[:pbl_top_ind])
    Tv_parcel_prof[:pbl_top_ind] = getTv(mc.dry_lapse(p_pbl, T_sfc), qv_prof[0])
    p_prof[:pbl_top_ind] = p_pbl
    
    # Extract surface dewpoint and LCL relative humidity for later
    
    Td_sfc = Td_pbl[0]
    pbl_top_rh = rh_pbl[-1]
    
    # Determine virtual temperature profile

    T_trop = -999.0 * units.kelvin
    
    p_prof[pbl_top_ind] = p_prof[pbl_top_ind-1] - ((p_prof[pbl_top_ind-1] * g * dz) / 
                                                   (Rd * Tv_env_prof[pbl_top_ind-1]))

    for i in range(pbl_top_ind, z.size):

        if (z[i] <= z_trop):

            # Determine the parcel virtual temperature

            p_array = np.array([p_sfc.magnitude, p_prof[i].magnitude]) * units.pascal
            T_parcel = mc.parcel_profile(p_array, T_sfc, Td_sfc)[-1]
            qv_parcel = mc.saturation_mixing_ratio(p_prof[i], T_parcel)
            Tv_parcel_prof[i] = getTv(T_parcel, qv_parcel)

            # Determine environmental temperature using buoyancy profile (eqn A1 from McCaul and
            # Weisman 2001). Use the critical lapse rate if environmental lapse rate exceeds 
            # crit_lapse

            B = (E * ((m / H) ** 2) * (z[i] - pbl_z).magnitude * 
                 np.exp(-(m / H) * (z[i] - pbl_z).magnitude))
            
            Tv_env = Tv_parcel_prof[i] / (1.0 + (B / g).magnitude)
            T_env = getTfromTv(Tv_env, qv_prof[i-1])
            T_env_prev = getTfromTv(Tv_env_prof[i-1], qv_prof[i-1])
            
            if ((T_env_prev - T_env) / (z[i] - z[i-1]) > crit_lapse):
                
                Tv_env_prof[i] = getTv((T_env_prev - (crit_lapse * (z[i] - z[i-1]))),
                                       qv_prof[i-1])
                
            else:
                
                Tv_env_prof[i] = Tv_env
                
            # Compute qv by assuming that RH varies linearly from the PBL top to tropopause
            
            RH = pbl_top_rh + (z[i] - pbl_z) * (RH_min - pbl_top_rh) / (z_trop - pbl_z)
            qv_prof[i] = mc.mixing_ratio_from_relative_humidity(RH, T_env, p_prof[i])

        else:
            
            # It assumed that at the tropopause, T = Tv since qv decreases with height in the 
            # troposphere
            
            if T_trop < 0:

                T_trop = Tv_env_prof[i-1]

            Tv_env_prof[i] = T_trop
            qv_prof[i] = mc.mixing_ratio_from_relative_humidity(RH, T_trop, p_prof[i])

            p_array = np.array([p_sfc.magnitude, p_prof[i].magnitude]) * units.pascal
            T_parcel = mc.parcel_profile(p_array, T_sfc, Td_sfc)[-1]
            qv_parcel = mc.saturation_mixing_ratio(p_prof[i], T_parcel)
            Tv_parcel_prof[i] = getTv(T_parcel, qv_parcel)

        # Find pressure of next vertical level using hydrostatic balance

        if i < (z.size - 1):

            p_prof[i+1] = p_prof[i] - ((p_prof[i] * g * (z[i+1] - z[i])) / (Rd * Tv_env_prof[i]))

    # Correct the profile above the mixed layer iteratively following the procedure discussed
    # in the appendix of Warren et al. (2017)

    T_env_prof = getTfromTv(Tv_env_prof, qv_prof)    
    RH_prof = mc.relative_humidity_from_mixing_ratio(qv_prof, T_env_prof, p_prof)
    getcape_out = gc.getcape(1, 1, p_prof.magnitude / 100.0, T_env_prof.magnitude - 273.15,
                             qv_prof)
    E_t = getcape_out[0] * units.joule / units.kilogram
    ratio = (E / E_t)

    while (np.abs(E - E_t) > (0.5 * units.joule / units.kilogram)):
        
        # Compute parcel profile
        
        T_parcel_prof = mc.parcel_profile(p_prof, T_sfc, Td_sfc)
        parcel_qv = np.ones(T_parcel_prof.shape) * qv_prof[0]
        parcel_qv[pbl_top_ind:] = mc.saturation_mixing_ratio(p_prof[pbl_top_ind:],
                                                             T_parcel_prof[pbl_top_ind:])
        Tv_parcel_prof = getTv(T_parcel_prof, parcel_qv)
        
        # Re-compute temperature and qv profiles using E_t factor

        T_trop = -999.0 * units.kelvin

        for i in range(pbl_top_ind, z.size):

            if (z[i] > pbl_z) and (z[i] <= z_trop):

                # Determine environmental temperature using buoyancy profile (eqn A1 from McCaul and
                # Weisman 2001). Use the critical lapse rate if environmental lapse rate exceeds 
                # crit_lapse

                B = (ratio * E * ((m / H) ** 2) * (z[i] - pbl_z).magnitude * 
                     np.exp(-(m / H) * (z[i] - pbl_z).magnitude))
                
                Tv_env = Tv_parcel_prof[i] / (1.0 + (B / g).magnitude)
                T_env = getTfromTv(Tv_env, qv_prof[i-1])
                T_env_prev = getTfromTv(Tv_env_prof[i-1], qv_prof[i-1])
            
                if ((T_env_prev - T_env) / (z[i] - z[i-1]) > crit_lapse):
                
                    Tv_env_prof[i] = getTv((T_env_prev - (crit_lapse * (z[i] - z[i-1]))),
                                           qv_prof[i-1])
                
                else:
                
                    Tv_env_prof[i] = Tv_env
                
                # Compute qv by assuming that RH varies linearly from PBL top to tropopause
            
                RH_prof[i] = pbl_top_rh + (z[i] - pbl_z) * (RH_min - pbl_top_rh) / (z_trop - pbl_z)
                qv_prof[i] = mc.mixing_ratio_from_relative_humidity(RH_prof[i], T_env, p_prof[i])

            elif (z[i] > z_trop):

                if T_trop < 0:

                    T_trop = Tv_env_prof[i-1]

                Tv_env_prof[i] = T_trop
                RH_prof[i] = RH_prof[i-1]
                qv_prof[i] = mc.mixing_ratio_from_relative_humidity(RH_prof[i], T_trop, p_prof[i])

            # Find pressure of next vertical level using hydrostatic balance

            if i < (z.size - 1):

                p_prof[i+1] = p_prof[i] - ((p_prof[i] * g * (z[i+1] - z[i])) / 
                                           (Rd * Tv_env_prof[i]))

        # Re-compute E_t

        T_env_prof = getTfromTv(Tv_env_prof, qv_prof)
        getcape_out = gc.getcape(1, 1, p_prof.magnitude / 100.0, T_env_prof.magnitude - 273.15,
                                 qv_prof)
        E_t = getcape_out[0] * units.joule / units.kilogram
        ratio = ratio * (E / E_t)
    
    # Fill thermo_prof DataFrame
    
    thermo_prof = pd.DataFrame()
    
    T_env_prof = getTfromTv(Tv_env_prof, qv_prof)
    
    thermo_prof['z'] = pd.Series(z.magnitude)
    thermo_prof['prs'] = pd.Series(p_prof.magnitude)
    thermo_prof['T'] = pd.Series(T_env_prof.to(units.kelvin).magnitude)
    thermo_prof['qv'] = pd.Series(qv_prof)
    
    # Add temperature and dewpoint
    
    RH_prof = mc.relative_humidity_from_mixing_ratio(qv_prof, T_env_prof, p_prof)
    Td_prof = mc.dewpoint_rh(T_env_prof, RH_prof).to(units.degC)
    
    thermo_prof['Td'] = pd.Series(Td_prof.to(units.kelvin).magnitude)

    return thermo_prof


"""
End idealized_sounding_fcts.py
"""
