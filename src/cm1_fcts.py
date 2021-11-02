"""
Functions for Analyzing CM1 Output 

Shawn Murdzek
sfm5282@psu.edu
Date Created: August 12, 2019
Environment: local_py base (Python 3.6)
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import numpy.ma as ma
import xarray as xr
import scipy.stats as ss
import scipy.special as spec
import MetAnalysis.src.idealized_sounding_fcts as isf
import MetAnalysis.src.largest_area as la
import MetAnalysis.src.kine_fcts as kf
import metpy.calc as mc
from metpy.units import units


#---------------------------------------------------------------------------------------------------
# Thermodynamic Functions
#---------------------------------------------------------------------------------------------------

def thetav_prime(cm1_ds):
    """
    Compute virtual potential temperature perturbations from CM1 output using eqn (2.20) in
    Markowski and Richardson (2010).
    Inputs:
        cm1_ds = Xarray dataset of CM1 output that includes the variables 'th', 'th0', 'qv', and 
            'qv0'
    Outputs:
        thvpert = Array of virtual potential temperature perturbations
    """

    # Define constants

    epn = 0.62198

    # Compute base state thv

    thv0 = (cm1_ds['th0'].values * (1.0 + (cm1_ds['qv0'].values / epn)) / 
            (1.0 + cm1_ds['qv0'].values))

    # Compute thv perturbations
 
    thv = cm1_ds['th'].values * (1.0 + (cm1_ds['qv'].values / epn)) / (1.0 + cm1_ds['qv'].values)
    thvpert = thv - thv0

    return thvpert


def thetarho_prime(cm1_ds):
    """
    Compute density potential temperature perturbations from CM1 output using eqn (2.22) in
    Markowski and Richardson (2010).
    Inputs:
        cm1_ds = Xarray dataset of CM1 output that includes the variables 'th', 'th0', 'qv', 'qv0',
            'qc', 'qr', 'qi', 'qs', 'qg', and 'qhl'
    Outputs:
        thrpert = Array of density potential temperature perturbations
    """

    # Define constants

    epn = 0.62198

    # Compute base state thr (it is assumed that the base state hydrometeor mixing ratio is 0)

    thr0 = (cm1_ds['th0'].values * (1.0 + (cm1_ds['qv0'].values / epn)) /
            (1.0 + cm1_ds['qv0'].values))

    # Compute thr perturbations

    qtot = (cm1_ds['qv'].values + cm1_ds['qc'].values + cm1_ds['qr'].values + cm1_ds['qi'].values +
            cm1_ds['qs'].values + cm1_ds['qg'].values + cm1_ds['qhl'].values)
    thr = cm1_ds['th'].values * (1.0 + (cm1_ds['qv'].values / epn)) / (1.0 + qtot)
    thrpert = thr - thr0

    return thrpert


def thepert(cm1_ds):
    """
    Compute equivalent potential temperature base state and perturbations
    Inputs:
        cm1_ds = CM1 XArray dataset that includes prs, qv, th, and base state
    Outputs:
        cm1_ds = CM1 XArray dataset with theta-e perturbations
    """
 
    p = cm1_ds['prs'].values * units.Pa
    qv = cm1_ds['qv'].values * (units.kg / units.kg)
    th = cm1_ds['th'].values * units.K
    p0 = cm1_ds['prs0'].values * units.Pa
    qv0 = cm1_ds['qv0'].values * (units.kg / units.kg)
    th0 = cm1_ds['th0'].values * units.K

    # Compute base state theta-e

    T0 = mc.temperature_from_potential_temperature(p0, th0)
    RH0 = mc.relative_humidity_from_mixing_ratio(qv0, T0, p0)
    Td0 = mc.dewpoint_from_relative_humidity(T0, RH0)
    the0 = mc.equivalent_potential_temperature(p0, T0, Td0).magnitude

    # Compute theta-e perturbations

    T = mc.temperature_from_potential_temperature(p, th)
    RH = mc.relative_humidity_from_mixing_ratio(qv, T, p)
    Td = mc.dewpoint_from_relative_humidity(T, RH)
    thepert = mc.equivalent_potential_temperature(p, T, Td).magnitude - the0

    # Add the0 and thepert to cm1_ds

    cm1_ds['the0'] = xr.DataArray(the0, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['the0'].attrs['long_name'] = 'base state equivalent potential temperature'   
    cm1_ds['the0'].attrs['units'] = 'K'

    cm1_ds['thepert'] = xr.DataArray(the0, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['thepert'].attrs['long_name'] = 'equivalent potential temperature perturbations'   
    cm1_ds['thepert'].attrs['units'] = 'K'

    return cm1_ds


def thetae(cm1_ds):
    """
    Compute equivalent potential temperature
    Inputs:
        cm1_ds = CM1 XArray dataset that includes prs, qv, and th
    Outputs:
        cm1_ds = CM1 XArray dataset with theta-e
    """
 
    p = cm1_ds['prs'].values
    qv = cm1_ds['qv'].values
    th = cm1_ds['th'].values
    
    T = isf.getTfromTheta(th, p)
    thetae = isf.getthe(T, p, qv)

    # Add theta-e to cm1_ds

    cm1_ds['thetae'] = xr.DataArray(thetae, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['thetae'].attrs['long_name'] = 'equivalent potential temperature'   
    cm1_ds['thetae'].attrs['units'] = 'K'

    return cm1_ds


def rh(cm1_ds):
    """
    Compute relative humidity (defined as qv / qvs)
    Inputs:
        cm1_ds = CM1 XArray dataset that includes prs, qv, and T
    Outputs:
        cm1_ds = CM1 XArray dataset with relative humidity
    """

    p = cm1_ds['prs'].values
    qv = cm1_ds['qv'].values

    try:
        T = cm1_ds['T'].values
    except KeyError:
        th = cm1_ds['th'].values
        rovcp = 287.04 / 1005.7
        T = th * ((p / 100000.0) ** rovcp)

    qvs = isf.get_qvs(T, p)

    cm1_ds['RH'] = xr.DataArray(qv / qvs, coords=cm1_ds['prs'].coords, dims=cm1_ds['prs'].dims)
    cm1_ds['RH'].attrs['long_name'] = 'relative humidity (decimal)'
    cm1_ds['RH'].attrs['units'] = 'none'

    return cm1_ds


#---------------------------------------------------------------------------------------------------
# Microphysical Functions
#---------------------------------------------------------------------------------------------------

def Dnr(cm1_ds, mur=0.0):
    """
    Compute number-weighted mean raindrop diameter (for qr > 0.01 g / kg)
    Inputs:
        cm1_ds = CM1 XArray dataset that includes qr and ncr
    Outputs:
        cm1_ds = CM1 XArray dataset with Dr
    """

    qr = cm1_ds['qr'].values
    try:
        nr = cm1_ds['ncr'].values
    except KeyError:
        try:
            nr = cm1_ds['nr'].values
        except KeyError:
            nr = cm1_ds['qnr'].values

    rhow = 1000.0
    qr = ma.masked_array(qr, mask=(qr < 0.01e-3))
    lamr = (np.pi * rhow * nr * spec.gamma(4.+mur) / (6. * spec.gamma(1.+mur) * qr)) ** (1./3.)
    Dr = ((1.+mur) / lamr) * 1e3

    cm1_ds['Dnr'] = xr.DataArray(Dr, coords=cm1_ds['qr'].coords, dims=cm1_ds['qr'].dims)
    cm1_ds['Dnr'].attrs['long_name'] = 'number-weighted mean raindrop diameter'
    cm1_ds['Dnr'].attrs['units'] = 'mm'

    return cm1_ds


def Dmr(cm1_ds, mur=0.0):
    """
    Computes the mass-weighted mean raindrop diameter (for qr > 0.01 g / kg)
    Inputs:
        cm1_ds = CM1 XArray dataset that includes qr and ncr
    Outputs:
        cm1_ds = CM1 XArray dataset with Dmr
    """

    qr = cm1_ds['qr'].values
    try:
        nr = cm1_ds['ncr'].values
    except KeyError:
        try:
            nr = cm1_ds['nr'].values
        except KeyError:
            nr = cm1_ds['qnr'].values

    rhow = 1000.0
    qr = ma.masked_array(qr, mask=(qr < 0.01e-3))
    lamr = (np.pi * rhow * nr * spec.gamma(4.+mur) / (6. * spec.gamma(1.+mur) * qr)) ** (1./3.)
    Dr = ((4.+mur) / lamr) * 1e3

    cm1_ds['Dmr'] = xr.DataArray(Dr, coords=cm1_ds['qr'].coords, dims=cm1_ds['qr'].dims)
    cm1_ds['Dmr'].attrs['long_name'] = 'mass-weighted mean raindrop diameter'
    cm1_ds['Dmr'].attrs['units'] = 'mm'

    return cm1_ds


def Dng(cm1_ds, rhog, CG, DG):
    """
    Compute number-weighted mean RIS diameter (for qg > 0.01 g / kg)
    Inputs:
        cm1_ds = CM1 XArray dataset that includes qg and ncg
        rhog = RIS density (kg / m^3)
        CG = RIS mass-diameter coefficient
        DG = RIS mass-diameter exponent
    Outputs:
        cm1_ds = CM1 XArray dataset with Dg
    """

    qg = cm1_ds['qg'].values
    ng = cm1_ds['ncg'].values

    lamg = (spec.gamma(1.0 + DG) * CG * ng / qg) ** (1./DG)
    Dg = ma.masked_array(1.0 / lamg, mask=(qg < 0.01e-3)) * 1e3

    cm1_ds['Dng'] = xr.DataArray(Dg, coords=cm1_ds['qg'].coords, dims=cm1_ds['qg'].dims)
    cm1_ds['Dng'].attrs['long_name'] = 'number-weighted mean RIS diameter'
    cm1_ds['Dng'].attrs['units'] = 'mm'

    return cm1_ds


def Dm(cm1_ds, qfield='qr', nfield='nr', rho=1000., name='mmDr'):
    """
    Computes the mean mass diameter

    Parameters
    ----------
    cm1_ds : XArray DataSet
        CM1 output file
    qfield : string, optional
        Name of mass mixing ratio field
    nfield : string, optional
        Name of number mixing ratio field
    rho : float, optional
        Density of hydrometeor species (kg / m^3)
    name : string, optional
        Name of field to add to DataSet

    Returns
    -------
    cm1_ds : XArray DataSet
        CM1 output file with the field 'Dm'

    """

    q = ma.masked_array(cm1_ds[qfield].values, mask=(cm1_ds[qfield] < 0.01e-3))
    n = cm1_ds[nfield].values
    Dm = ((6. * q) / (n * np.pi * rho)) ** (1./3.) * 1e3

    cm1_ds[name] = xr.DataArray(Dm, coords=cm1_ds[qfield].coords, dims=cm1_ds[qfield].dims)
    cm1_ds[name].attrs['long_name'] = 'mean mass diameter'
    cm1_ds[name].attrs['units'] = 'mm'

    return cm1_ds


#---------------------------------------------------------------------------------------------------
# Dynamic Functions
#---------------------------------------------------------------------------------------------------

def vorts(cm1_ds):
    """
    Compute streamwise vorticity
    Inputs:
        cm1_ds = CM1 XArray dataset that includes all 3 velocity and vorticity components
    Outputs:
        cm1_ds = CM1 XArray dataset with streamwise vorticity
    """

    u = cm1_ds['uinterp'].values
    v = cm1_ds['vinterp'].values
    w = cm1_ds['winterp'].values

    dot = (cm1_ds['xvort'].values * u + 
           cm1_ds['yvort'].values * v +
           cm1_ds['zvort'].values * w)

    cm1_ds['vorts'] = dot / np.sqrt(u*u + v*v + w*w)
    cm1_ds['vorts'].attrs['long_name'] = 'streamwise vorticity'
    cm1_ds['vorts'].attrs['units'] = '/s'

    return cm1_ds


def hvorts(cm1_ds):
    """
    Compute streamwise horizontal vorticity
    Inputs:
        cm1_ds = CM1 XArray dataset that includes all 3 velocity and vorticity components
    Outputs:
        cm1_ds = CM1 XArray dataset with streamwise horizontal vorticity
    """

    u = cm1_ds['uinterp']
    v = cm1_ds['vinterp']

    dot = cm1_ds['xvort'] * u + cm1_ds['yvort'] * v
    cm1_ds['hvorts'] = dot / np.sqrt(u*u + v*v)
    cm1_ds['hvorts'].attrs['long_name'] = 'streamwise horizontal vorticity'
    cm1_ds['hvorts'].attrs['units'] = '/s'

    return cm1_ds


def dwdz(cm1_ds):
    """
    Compute the vertical gradient of W
    Inputs:
        cm1_ds = CM1 XArray dataset that includes w
    Outputs:
        cm1_ds = CM1 XArray dataset with dw/dz
    """

    w = cm1_ds['w'].values
    try:
        zf = cm1_ds['zf'].values * 1e3
    except KeyError:
        zf = cm1_ds['nkp1'].values * 1e3

    dz = zf[1:] - zf[:-1]
    dz3d = np.zeros(cm1_ds['winterp'].shape)
    for i in range(w.shape[0]):
        for j in range(w.shape[2]):
            for k in range(w.shape[3]):
                dz3d[i, :, j, k] = dz

    dwdz = (w[:, 1:, :, :] - w[:, :-1, :, :]) / dz3d
    cm1_ds['dwdz'] = xr.DataArray(dwdz, coords=cm1_ds['winterp'].coords, 
                                  dims=cm1_ds['winterp'].dims)
    cm1_ds['dwdz'].attrs['long_name'] = 'vertical gradient of vertical velocity'
    cm1_ds['dwdz'].attrs['units'] = '/s'

    return cm1_ds


def vortz_stretch(cm1_ds):
    """
    Compute vertical vorticity stretching
    Inputs:
        cm1_ds = CM1 XArray dataset that includes w and zvort
    Outputs:
        cm1_ds = CM1 XArray dataset with vertical vorticity stretching
    """

    cm1_ds = dwdz(cm1_ds)
    cm1_ds['vortz_stretch'] = cm1_ds['zvort'] * cm1_ds['dwdz']
    cm1_ds['vortz_stretch'].attrs['long_name'] = 'vertical vorticity stretching'
    cm1_ds['vortz_stretch'].attrs['units'] = 's^-2'

    return cm1_ds


def vortz_tilt(cm1_ds):
    """
    Compute vertical vorticity tilting
    Inputs:
        cm1_ds = CM1 XArray dataset that includes w and 3D vorticity vector
    Outputs:
        cm1_ds = CM1 XArray dataset with vertical vorticity tilting
    """

    w = cm1_ds['winterp'].values
    vortx = cm1_ds['xvort'].values
    vorty = cm1_ds['yvort'].values
    try:
        xh = cm1_ds['xh'].values * 1e3
        yh = cm1_ds['yh'].values * 1e3
    except KeyError:
        xh = cm1_ds['ni'].values * 1e3
        yh = cm1_ds['nj'].values * 1e3

    dwdy, dwdx = np.gradient(w, yh, xh, edge_order=2, axis=(2, 3))
    tilt = (vortx * dwdx) + (vorty * dwdy)
    cm1_ds['vortz_tilt'] = xr.DataArray(tilt, coords=cm1_ds['winterp'].coords, 
                                        dims=cm1_ds['winterp'].dims)
    cm1_ds['vortz_tilt'].attrs['long_name'] = 'vertical vorticity tilting'
    cm1_ds['vortz_tilt'].attrs['units'] = 's^-2'
    
    return cm1_ds


def hgrad_w_mag(cm1_ds):
    """
    Compute magnitude of the vertical velocity horizontal gradient
    Inputs:
        cm1_ds = CM1 XArray dataset that includes w
    Outputs:
        cm1_ds = CM1 XArray dataset with vertical velocity horizontal gradient
    """

    w = cm1_ds['winterp'].values
    try:
        xh = cm1_ds['xh'].values * 1e3
        yh = cm1_ds['yh'].values * 1e3
    except KeyError:
        xh = cm1_ds['ni'].values * 1e3
        yh = cm1_ds['nj'].values * 1e3

    dwdy, dwdx = np.gradient(w, yh, xh, edge_order=2, axis=(2, 3))
    mag = np.sqrt(dwdx*dwdx + dwdy*dwdy)
    cm1_ds['hgrad_w_mag'] = xr.DataArray(mag, coords=cm1_ds['winterp'].coords, 
                                         dims=cm1_ds['winterp'].dims)
    cm1_ds['hgrad_w_mag'].attrs['long_name'] = 'vertical velocity horizontal gradient magnitude'
    cm1_ds['hgrad_w_mag'].attrs['units'] = 's^-1'
    
    return cm1_ds


def conv2d(cm1_ds):
    """
    Compute horizontal convergence, assuming constant dx and dy
    Inputs:
        cm1_ds = CM1 XArray dataset that includes u and v
    Outputs:
        cm1_ds = CM1 XArray dataset with horizontal convergence
    """

    u = cm1_ds['u'].values
    v = cm1_ds['v'].values
    try:
        dx = (cm1_ds['xf'].values[1] - cm1_ds['xf'].values[0]) * 1e3
        dy = (cm1_ds['yf'].values[1] - cm1_ds['yf'].values[0]) * 1e3
    except KeyError:
        dx = (cm1_ds['nip1'].values[1] - cm1_ds['nip1'].values[0]) * 1e3
        dy = (cm1_ds['njp1'].values[1] - cm1_ds['njp1'].values[0]) * 1e3
    div = (u[:, :, :, 1:] - u[:, :, :, :-1]) / dx + (v[:, :, 1:, :] - v[:, :, :-1, :]) / dy
    cm1_ds['conv2d'] = xr.DataArray(-div, coords=cm1_ds['uinterp'].coords, 
                                    dims=cm1_ds['uinterp'].dims)
    cm1_ds['conv2d'].attrs['long_name'] = 'horizontal convergence'
    cm1_ds['conv2d'].attrs['units'] = '/s'

    return cm1_ds


def OW(cm1_ds):
    """
    Compute the Okubo-Weiss number (Markowski et al. 2011, EJSSM), assuming constant dx and dy
    Inputs:
        cm1_ds = CM1 XArray dataset that includes u, v, and zvort
    Outputs:
        cm1_ds = CM1 XArray dataset with Okubo-Weiss number
    """

    print('THERE IS A BUG IN THE OW FUNCTION')

    # D2 is not computed correctly. Also, double-check that OW = D1*D1 + D2*D2 - zeta*zeta

    u = cm1_ds['u'].values
    v = cm1_ds['v'].values
    zeta = cm1_ds['zvort'].values
    try:
        dx = (cm1_ds['xf'].values[1] - cm1_ds['xf'].values[0]) * 1e3
        dy = (cm1_ds['yf'].values[1] - cm1_ds['yf'].values[0]) * 1e3
    except KeyError:
        dx = (cm1_ds['nip1'].values[1] - cm1_ds['nip1'].values[0]) * 1e3
        dy = (cm1_ds['njp1'].values[1] - cm1_ds['njp1'].values[0]) * 1e3
    D1 = (u[:, :, :, 1:] - u[:, :, :, :-1]) / dx - (v[:, :, 1:, :] - v[:, :, :-1, :]) / dy
    D2 = (v[:, :, 1:, :] - v[:, :, :-1, :]) / dx + (u[:, :, :, 1:] - u[:, :, :, :-1]) / dy
    cm1_ds['OW'] = xr.DataArray(D1*D1 + D2*D2 - zeta*zeta, coords=cm1_ds['uinterp'].coords, 
                                dims=cm1_ds['uinterp'].dims)
    cm1_ds['OW'].attrs['long_name'] = 'Okubo-Weiss number'
    cm1_ds['OW'].attrs['units'] = '/s^2'

    return cm1_ds


def vort_mag(cm1_ds):
    """
    Compute magnitude of the 3D vorticity vector
    Inputs:
        cm1_ds = CM1 XArray dataset that includes 3D vorticity vector components
    Outputs:
        cm1_ds = CM1 XArray dataset with magnitude of 3D vorticity vector
    """

    cm1_ds['vort_mag'] = np.sqrt(cm1_ds['xvort']*cm1_ds['xvort'] + 
                                 cm1_ds['yvort']*cm1_ds['yvort'] +
                                 cm1_ds['zvort']*cm1_ds['zvort'])
    cm1_ds['vort_mag'].attrs['long_name'] = '3D vorticity vector magnitude'
    cm1_ds['vort_mag'].attrs['units'] = 's^-1'
    
    return cm1_ds


def hvort_mag(cm1_ds):
    """
    Compute magnitude of the horizontal vorticity vector
    Inputs:
        cm1_ds = CM1 XArray dataset that includes 2D vorticity vector components
    Outputs:
        cm1_ds = CM1 XArray dataset with magnitude of 2D vorticity vector
    """

    cm1_ds['hvort_mag'] = np.sqrt(cm1_ds['xvort']*cm1_ds['xvort'] + 
                                  cm1_ds['yvort']*cm1_ds['yvort'])
    cm1_ds['hvort_mag'].attrs['long_name'] = '2D vorticity vector magnitude'
    cm1_ds['hvort_mag'].attrs['units'] = 's^-1'
    
    return cm1_ds


def wXcirc(cm1_ds):
    """
    Compute w X circ2 where circ2 >= 0 at the lowest model level
    
    Parameters
    ----------
    cm1_ds : xr.Dataset
        CM1 output dataset

    Returns
    -------
    cm1_ds : xr.Dataset
        CM1 output dataset with winterp at LML times circulation (m^3 / s^2)

    """

    w = cm1_ds['winterp'][:, 0, :, :].values
    x1d = cm1_ds['xh'].values
    y1d = cm1_ds['yh'].values
    wavg = kf.avg_var(w, x1d, y1d, 2.0)
    C = cm1_ds['circ2'][:, 0, :, :].values
    with np.errstate(invalid='ignore'):
        C[C < 0] = 0 
    dum = np.reshape(wavg * C, cm1_ds['uh'].shape)

    cm1_ds['wXcirc'] = xr.DataArray(dum, coords=cm1_ds['uh'].coords, dims=cm1_ds['uh'].dims)
    cm1_ds['wXcirc'].attrs['long_name'] = 'LML 2-km avg W X 2-km circulation'
    cm1_ds['wXcirc'].attrs['units'] = 'm^3 s^-2'
    
    return cm1_ds

def SRH(cm1_ds, zbot=0, ztop=3, dim=2):
    """
    Compute storm-relative helicity over a given layer
    NOTE: THIS FUNCTION HAS NOT BEEN TESTED YET!
    Inputs:
        cm1_ds = CM1 XArray dataset that includes all 3 velocity and vorticity components
    Outputs:
        cm1_ds = CM1 XArray dataset with SRH
    Keywords:
        zbot = Bottom of layer to compute SRH (km)
        ztop = Top of layer to compute SRH (km)
        dim = Option to compute SRH using the 2D or 3D velocity/vorticity vector
    """

    # Compute streamwise vorticity

    if dim == 2:
        cm1_ds = hvorts(cm1_ds)
        vorts = cm1_ds['hvorts']
    elif dim == 3:
        cm1_ds = vorts(cm1_ds)
        vorts = cm1_ds['vorts']

    # Determine indices of zbot and ztop

    try:
        zh = cm1_ds['zh']
    except KeyError:
        zh = cm1_ds['nk']

    kbot = np.argmin(np.abs(zh - zbot))
    ktop = np.argmin(np.abs(zh - ztop))

    # Save SRH to cm1_ds

    var = 'SRH%d%d' % (zbot, ztop)
    cm1_ds[var] = np.trapz(vorts[0, kbot:(ktop+1), :, :], x=zh, axis=0)
    cm1_ds[var].attrs['long_name'] = '%.3f-%.3f km storm-relative helicity' % (zbot, ztop)
    cm1_ds[var].attrs['units'] = 'm^2 / s^2'

    return cm1_ds


def advect(cm1_ds, field, wind):
    """
    Compute advection of a field in a certain direction.
    Note: Only works for fields defined on the scalar grid in CM1
    Inputs:
        cm1_ds = CM1 XArray dataset that includes 'field' and the 3D wind vector
        field = Name of the field being advected
        wind = Wind component advecting 'field'. Options: 'u', 'v', or 'w'
    Outputs:
        cm1_ds = CM1 XArray dataset with advection (named '<wind>adv<field>')
    """

    if wind == 'u':
        x = cm1_ds['xh'].values * 1e3
        v = cm1_ds['uinterp'].values
        axis = 3
    elif wind == 'v':
        x = cm1_ds['yh'].values * 1e3
        v = cm1_ds['vinterp'].values
        axis = 2
    elif wind == 'w':
        x = cm1_ds['zh'].values * 1e3
        v = cm1_ds['winterp'].values
        axis = 1

    adv = -v * np.gradient(cm1_ds[field].values, x, axis=axis, edge_order=2)

    name = '%sadv%s' % (wind, field)
    cm1_ds[name] = xr.DataArray(adv, coords=cm1_ds[field].coords, dims=cm1_ds[field].dims)
    cm1_ds[name].attrs['long_name'] = '%s advection of %s' % (wind, field)
    cm1_ds[name].attrs['units'] = '%s /s' % cm1_ds[field].attrs['units']

    return cm1_ds


#---------------------------------------------------------------------------------------------------
# Predictability Functions
#---------------------------------------------------------------------------------------------------

def diff_tot_energy(ds1, ds2, xlim=None, ylim=None):
    """
    Compute difference total energy between two simulations

    Parameters:
    -----------
    ds1 : xr.Dataset
        Dataset of first CM1 simulation
    ds2 : xr.Dataset
        Dataset of second CM1 simulation
    xlim : list, optional
        Domain over which to compute DTE (set to None to use entire domain)
    ylim : list, optional
        Domain over which to compute DTE (set to None to use entire domain)

    Returns:
    --------
    dte : array
        Difference total energy (J / kg)

    Notes:
    ------
    See F. Zhang et al. (2003, JAS) and Y. Zhang et al. (2016, MWR)

    """

    cp = 1005.7
    Tr = 270.

    if xlim != None:
        ib1 = np.argmin(np.abs(ds1['xh'].values - xlim[0]))
        ib2 = np.argmin(np.abs(ds2['xh'].values - xlim[0]))
        ie1 = np.argmin(np.abs(ds1['xh'].values - xlim[1])) + 1
        ie2 = np.argmin(np.abs(ds2['xh'].values - xlim[1])) + 1
    else:
        ib1 = 0
        ib2 = 0
        ie1 = len(ds1['xh'])
        ie2 = len(ds2['xh'])

    if ylim != None:
        jb1 = np.argmin(np.abs(ds1['yh'].values - ylim[0]))
        jb2 = np.argmin(np.abs(ds2['yh'].values - ylim[0]))
        je1 = np.argmin(np.abs(ds1['yh'].values - ylim[1])) + 1
        je2 = np.argmin(np.abs(ds2['yh'].values - ylim[1])) + 1
    else:
        jb1 = 0
        jb2 = 0
        je1 = len(ds1['yh'])
        je2 = len(ds2['yh'])

    T1 = isf.getTfromTheta(ds1['th'][:, :, jb1:je1, ib1:ie1].values, 
                           ds1['prs'][:, :, jb1:je1, ib1:ie1].values)
    T2 = isf.getTfromTheta(ds2['th'][:, :, jb2:je2, ib2:ie2].values, 
                           ds2['prs'][:, :, jb2:je2, ib2:ie2].values)

    dte = 0.5 * ((ds1['uinterp'][:, :, jb1:je1, ib1:ie1].values - 
                  ds2['uinterp'][:, :, jb2:je2, ib2:ie2].values) ** 2 + 
                 (ds1['vinterp'][:, :, jb1:je1, ib1:ie1].values - 
                  ds2['vinterp'][:, :, jb2:je2, ib2:ie2].values) ** 2 +
                 (cp / Tr) * (T1 - T2) ** 2)

    return dte


#---------------------------------------------------------------------------------------------------
# Function to Compute Supercell Metrics
#---------------------------------------------------------------------------------------------------

def super_cp_metrics(cm1_ds, coord, r=5.0, cp_thres=-0.5, cp_z_max=2.0):
    """
    Compute supercell cold pool metrics. 
    Inputs:
        cm1_ds = CM1 output DataSet
        coord = Supercell centroid [(x, y) ordered pair in km]
    Outputs:
        min_thp = Minimum surface potential temperature perturbation (K)
        avg_thp = Average surface cold pool potential temperature perturbation (K)
        cp_ext = Surface cold pool areal extent (fraction)
        Bint2d = Integrated surface cold pool buoyancy (m / s^2)
        Bint3d = Integrated 3D cold pool buoyancy (m / s^2)
        totB2d = Integrated surface buoyancy (m / s^2)
    Keywords:
        r = Distance from supercell centroid to compute cold pool metrics (km)
        cp_thres = Maximum potential temperature perturbation to define the cold pool (K)
        cp_z_max = Maximum height AGL to search for cold pool when computing Bint3d (km)
    """

    # Determine index where z = cp_z_max

    cp_kmax = np.where(cm1_ds['zh'] <= cp_z_max)[0][-1]

    # Extract necessary fields

    x2d, y2d = np.meshgrid(cm1_ds['xh'], cm1_ds['yh'])
    thp = cm1_ds['th'][0, :(cp_kmax+1), :, :].values - cm1_ds['th0'][0, :(cp_kmax+1), :, :].values
    B = cm1_ds['buoyancy'][0, :(cp_kmax+1), :, :].values

    # Create mask based on thpert and distance from supercell centroid

    cp_cond = (thp <= cp_thres)
    d_cond = ((x2d - coord[0])**2 + (y2d - coord[1])**2) <= (r*r)
    for i in range(cp_kmax+1):
        cp_cond[i, :, :] = cp_cond[i, :, :] * d_cond

    # Compute total integrated surface buoyancy (irrespective of cold pool)

    totB2d = (B[0, :, :] * d_cond).sum()

    # Compute cold pool properties

    sfc_thp = thp[0, :, :] * cp_cond[0, :, :]
    B = B * cp_cond
    npts = cp_cond[0, :, :].sum()

    min_thp = sfc_thp.min()
    avg_thp = sfc_thp.sum() / max(npts, 1)
    cp_ext = npts / d_cond.sum()
    Bint2d = B[0, :, :].sum()
    Bint3d = B.sum()

    return min_thp, avg_thp, cp_ext, Bint2d, Bint3d, totB2d


def super_str_metrics(cm1_ds, coord, r=5.0, circ_field ='circ2'):
    """
    Compute supercell strength and tornadic potential metrics. 
    Inputs:
        cm1_ds = CM1 output DataSet
        coord = Supercell centroid [(x, y) ordered pair in km]
    Outputs:
        wmax1km = Maximum 1-km updraft (m / s)
        zvmax1km = Maximum 1-km zvort (m / s)
        wmax = Maximum updraft (at any level)
        corr = Correlation between 1-km W and 1-km zvort (unitless)
        circmax1km = Maximum Eulerian circulation at 1 km AGL (m^2 / s)
        circmaxsfc = Maximum Eulerian circulation at the lowest model level (m^2 / s)
        uhmax = Maximum 2--5 km updraft helicity (m^2 / s^2)
        wcircdist = Distance between 1-km W maximum and near-surface circulation maximum (km)
        uhcircdist = Distance between supercell centroid and near-surface circulation maximum (km)
        up_Mf = Average 1-km updraft (W > 5 m/s) mass flux within r of supercell centroid 
            (kg / s / m^4). See updraft mass flux from Klees et al. (2016)
        Mfcircmax = 1-km vertical mass flux computed using a 2-km radius ring centered on circmaxsfc 
            (kg / s / m^4)
        up_Mfcircmax = 1-km up_Mf computed using a 2-km radius ring centered on circmaxsfc 
            (kg / s / m^4)
        zvmaxsfc = Maximum zvort at LML (/ s)
        zvminsfc = Minimum zvort at LML (/ s)
        uparea = 1-km updraft (w > 5 m/s) (km^2)
    Keywords:
        r = Distance from supercell centroid to compute metrics (km)
        circ_field = Name of Eulerian circulation field
    """

    # Determine vertical index corresponding to 1 km AGL

    kh1km = np.argmin(np.abs(cm1_ds['zh'].values - 1.0))
    kf1km = np.argmin(np.abs(cm1_ds['zf'].values - 1.0))

    # Create mask based on distance from supercell centroid

    x2d, y2d = np.meshgrid(cm1_ds['xh'], cm1_ds['yh'])
    mask = ((x2d - coord[0])**2 + (y2d - coord[1])**2) <= (r*r)

    mask3d = np.zeros(cm1_ds['w'][0, :, :, :].shape)
    for i in range(mask3d.shape[0]):
        mask3d[i, :, :] = mask

    # Compute updraft and zvort metrics

    wmax1km = (cm1_ds['w'][0, kf1km, :, :] * mask).max()
    zvmax1km = (cm1_ds['zvort'][0, kh1km, :, :] * mask).max()
    zvmaxsfc = (cm1_ds['zvort'][0, 0, :, :] * mask).max()
    zvminsfc = (cm1_ds['zvort'][0, 0, :, :] * mask).min()
    wmax = (cm1_ds['w'][0, :, :, :] * mask3d).max()

    # Compute correlation between W and zvort at 1 km AGL

    w1km = np.ma.masked_array(cm1_ds['winterp'][0, kh1km, :, :].values, 
                              mask=np.logical_not(mask)).flatten()
    zv1km = np.ma.masked_array(cm1_ds['zvort'][0, kh1km, :, :].values, 
                               mask=np.logical_not(mask)).flatten()
    corr = np.corrcoef(w1km, zv1km)[0, 1] 

    # Compute circulation metrics

    circmax1km = (cm1_ds[circ_field][0, kh1km, :, :] * mask).max()
    circmaxsfc = (cm1_ds[circ_field][0, 0, :, :] * mask).max()

    # Compute UH metrics

    uhmax = (cm1_ds['uh'][0, :, :] * mask).max()

    # Compute distance between 1-km updraft max and near-surface circulation max

    wi, wj = np.unravel_index(np.argmax(cm1_ds['w'][0, kf1km, :, :].values * mask), mask.shape)
    ci, cj = np.unravel_index(np.nanargmax(cm1_ds[circ_field][0, 0, :, :].values * mask), 
                              mask.shape)
    wcircdist = np.sqrt((x2d[wi, wj] - x2d[ci, cj])**2 + (y2d[wi, wj] - y2d[ci, cj])**2)

    # Compute distance between supercell centroid and near-surface circulation max

    uhcircdist = np.sqrt((x2d[ci, cj] - coord[0])**2 + (y2d[ci, cj] - coord[1])**2)

    # Compute convective updraft mass flux metrics (see Klees et al. 2016)

    wmask = cm1_ds['winterp'][0, kh1km, :, :].values >= 5
    up_Mf = (cm1_ds['winterp'][0, kh1km, :, :].values * cm1_ds['rho'][0, kh1km, :, :].values * 
             mask * wmask).sum() / (np.pi * r*r*1.e6)

    mask_lmlcirc = ((x2d - x2d[ci, cj])**2 + (y2d - y2d[ci, cj])**2) <= 4.001
    Mfcircmax = (cm1_ds['winterp'][0, kh1km, :, :].values * cm1_ds['rho'][0, kh1km, :, :].values * 
                 mask_lmlcirc).sum() / (np.pi * 4.e6)
    up_Mfcircmax = (cm1_ds['winterp'][0, kh1km, :, :].values * cm1_ds['rho'][0, kh1km, :, :].values * 
                    mask_lmlcirc * wmask).sum() / (np.pi * 4.e6)

    # Determine largest, continuous area with W > 5 m/s

    wmask = wmask * mask
    uparea = 0
    if wmask.sum() > 0:
        ind = np.where(mask)
        size, _, _ = la.largestArea(wmask)
        dx = cm1_ds['xh'][1].values - cm1_ds['xh'][0].values
        dy = cm1_ds['yh'][1].values - cm1_ds['yh'][0].values
        uparea = size * dx * dy    

    return (wmax1km, zvmax1km, wmax, corr, circmax1km, circmaxsfc, uhmax, wcircdist, uhcircdist, 
            up_Mf, Mfcircmax, up_Mfcircmax, zvmaxsfc, zvminsfc, uparea)


def id_tlv(cm1_ds, coord, r=5.0, zv_thres=0.1, prspert_thres=-200):
    """
    Identify tornado-like vortices (TLVs) 
    Inputs:
        cm1_ds = CM1 output DataSet
        coord = Supercell centroid [(x, y) ordered pair in km]
    Outputs:
        x = X coordinate of TLV (km)
        y = Y coordinate of TLV (km)
        zv = Vertical vorticity of TLV (s^-1)
        ppert = Pressure perturbation of TLV (Pa)
    Keywords:
        r = Distance from supercell centroid to search for TLVs (km)
        zv_thres = Vertical vorticity threshold to identify TLVs (s^-1)
        prspert_thres = Pressure perturbation threshold to identify TLVs (Pa)
    """

    # Initialize output lists

    x, y, zv, ppert = [], [], [], []

    # Create mask based on distance from supercell centroid

    x2d, y2d = np.meshgrid(cm1_ds['xh'], cm1_ds['yh'])
    mask = ((x2d - coord[0])**2 + (y2d - coord[1])**2) <= (r*r)

    # Identify TLVs (both cyclonic and anticyclonic)

    pcond = ((cm1_ds['prs'][0, 0, :, :] - cm1_ds['prs0'][0, 0, :, :]) * mask <= prspert_thres)

    iind, jind, = np.where((np.abs(cm1_ds['zvort'][0, 0, :, :]) >= zv_thres) * pcond)
    for i, j in zip(iind, jind):
        x.append(x2d[i, j])
        y.append(y2d[i, j])
        zv.append(cm1_ds['zvort'][0, 0, i, j].values)
        ppert.append(cm1_ds['prs'][0, 0, i, j].values - cm1_ds['prs0'][0, 0, i, j].values)

    return x, y, zv, ppert


#---------------------------------------------------------------------------------------------------
# Function to Identify the FFD and RFD
#---------------------------------------------------------------------------------------------------

def id_ffd_rfd(cm1_ds, x_pts, y_pts, ffd_max_dist=15.0, rfd_max_dist=5.0, ffd_min_dist=1.0, 
               rfd_min_dist=0.5, meso_height=1.5, ref_thres=15.0, search_x=[-5, 5], 
               search_y=[-5, 5]):
    """
    Identifies the (x, y) coordinates from x_pts and y_pts that lie within the forward-flank 
    downdraft (FFD) and rear-flank downdraft (RFD) using an Xarray dataset that comes from the 
    output of CM1. This algorithm is based on the definition of the forward flank from sect 2a of 
    Shabbott and Markowski (2006) with the added constraints that (1) the points must lie above the 
    line that passes through the mesocyclone center and is parallel to the line bisecting the 
    forward-flank precipitation shield, (2) the points must lie within max_dist of the mesocyclone 
    center, and (3) the points must lie at least min_dist away from the mesocyclone center. The rear 
    flank is defined the same way, except the rear flank includes those points on the rear side of 
    a line drawn orthogonal to the major axis of the echo. Constraint (1) above does not apply to 
    the rear flank (but constraints (2) and (3) do). The mesocyclone is diagnosed using the maximum 
    azimuthal shear at meso_height AGL.
    Inputs:
        cm1_ds = Xarray dataset from CM1
        x_pts = X-coordinates of input data points (km)
        y_pts = Y-coordinates of input data points (km)
    Outputs:
        ffd_ind = Indices of the data points (x_pts, y_pts) that lie within the FFD
        rfd_ind = Indices of the data points (x_pts, y_pts) that lie within the RFD
        x_meso_ctr = X-coordinate of mesocyclone center (km)
        y_meso_ctr = Y-coordinate of mesocyclone center (km)
        m_major_axis = Slope of line that passes through the major axis of the storm
        m_spine = Slope of the line orthogonal to m_major_axis
    Keywords:
        ffd_max_dist = Maximum distance a point within the FFD is allowed to lie from the 
            mesocyclone center (km)
        rfd_max_dist = Maximum distance a point within the RFD is allowed to lie from the 
            mesocyclone center (km)
        meso_height = Height AGL where the mesocyclone center is diagnosed
        ref_thres = Minimum reflectivity of points considered to be in the precipitation shield 
            (dBZ)
        search_x = X-coordinates of the lower-left and upper-right corners of the box to search
            for the mesocyclone center in (km)
        search_y = Y-coordinates of the lower-left and upper-right corners of the box to search
            for the mesocyclone center in (km)
    """
    
    # Determine the circulation center using the maximum vertical vorticity
    
    z_ind = np.argmin(np.abs(cm1_ds['nk'].values - meso_height))
    
    j_search = np.logical_and(cm1_ds['ni'].values >= search_x[0], 
                              cm1_ds['ni'].values <= search_x[1])
    i_search = np.logical_and(cm1_ds['nj'].values >= search_y[0], 
                              cm1_ds['nj'].values <= search_y[1])
    
    vortz = np.squeeze(cm1_ds['zvort'][0, z_ind, i_search, j_search].values)
    vortz = np.ma.masked_array(vortz, mask=np.isnan(vortz))
    
    # Note that i_meso_ctr and j_meso_ctr are the indices for the reduced set of x gridpoints
    # (i.e., only the gridpoints within the search window are considered)
    
    i_meso_ctr, j_meso_ctr = np.unravel_index(np.argmax(vortz), vortz.shape)
    x_meso_ctr = cm1_ds['ni'].values[j_search][j_meso_ctr]
    y_meso_ctr = cm1_ds['nj'].values[i_search][i_meso_ctr]
    
    # Find the line bisecting the precipitation shield along the major axis using linear regression
    
    i_precip, j_precip = np.where(np.squeeze(cm1_ds['dbz'][0, z_ind, :, :]) >= ref_thres)
    x_precip = cm1_ds['ni'].values[j_precip]
    y_precip = cm1_ds['nj'].values[i_precip]
    
    m_major_axis, b_major_axis, r, p, sterr = ss.linregress(x_precip, y_precip)
    
    # Find the slope of the line orthogonal to the line bisecting the precipitation shield. This 
    # line should be roughly parallel to the spine of the hook echo
    
    m_spine = -1.0 / m_major_axis
    
    # Find the indices of the points that lie within the FFD
    
    dist2_meso = (y_pts - y_meso_ctr) ** 2.0 + (x_pts - x_meso_ctr) ** 2.0
    
    ffd_cond1 = x_pts >= x_meso_ctr + (1.0 / m_spine) * (y_pts - y_meso_ctr)    
    ffd_cond2 = y_pts >= y_meso_ctr + m_major_axis * (x_pts - x_meso_ctr)
    ffd_cond3 = dist2_meso <= (ffd_max_dist * ffd_max_dist)
    ffd_cond4 = dist2_meso >= (ffd_min_dist * ffd_min_dist)
    
    ffd_ind = np.where(np.logical_and(np.logical_and(ffd_cond1, ffd_cond2), 
                                      np.logical_and(ffd_cond3, ffd_cond4))) 

    # Find the indices of the points that lie within the RFD
    
    rfd_cond1 = x_pts <= x_meso_ctr + (1.0 / m_spine) * (y_pts - y_meso_ctr)    
    rfd_cond2 = dist2_meso <= (rfd_max_dist * rfd_max_dist)
    rfd_cond3 = dist2_meso >= (rfd_min_dist * rfd_min_dist)
    
    rfd_ind = np.where(np.logical_and(np.logical_and(rfd_cond1, rfd_cond2), rfd_cond3)) 
    
    return ffd_ind, rfd_ind, x_meso_ctr, y_meso_ctr, m_major_axis, m_spine


"""
End cm1_fcts.py
"""
