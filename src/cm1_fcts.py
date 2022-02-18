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
import warnings
import MetAnalysis.src.idealized_sounding_fcts as isf
import MetAnalysis.src.largest_area as la
import MetAnalysis.src.kine_fcts as kf
import MetAnalysis.src.micro_fcts as mf


#---------------------------------------------------------------------------------------------------
# Helper Functions
#---------------------------------------------------------------------------------------------------

def _qfields(cm1_ds):
    """
    Return a list of all hydrometeor mass mixing ratio fields
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain hydrometeor mixing ratios
        
    Returns
    -------
    qlist : list
        List of strings associated with the hydrometeor mixing ratio fields
        
    """
    
    qlist = []
    for k in cm1_ds.keys():
        if k[0] == 'q' and k[1] != 'v':
            qlist.append(k)
            
    return qlist


def _sgrid_spacing(cm1_ds):
    """
    Return arrays of dx, dy, and dz for the scalar grid
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset
        
    Returns
    -------
    dx : array
        1-D array of grid spacing in the x direction (km)
    dy : array
        1-D array of grid spacing in the y direction (km)
    dz : array
        1-D array of grid spacing in the z direction (km)
        
    """
    
    version = float(cm1_ds.attrs['CM1 version'][4:])
    if version < 20:
        xfield = 'nip1'
        yfield = 'njp1'
        zfield = 'nkp1'
    else:
        xfield = 'xf'
        yfield = 'yf'
        zfield = 'zf'
        
    dx = np.round_(cm1_ds[xfield][1:].values - cm1_ds[xfield][:-1].values, decimals=5)
    dy = np.round_(cm1_ds[yfield][1:].values - cm1_ds[yfield][:-1].values, decimals=5)
    dz = np.round_(cm1_ds[zfield][1:].values - cm1_ds[zfield][:-1].values, decimals=5)
            
    return dx, dy, dz


#---------------------------------------------------------------------------------------------------
# Thermodynamic Functions
#---------------------------------------------------------------------------------------------------

def thvpert(cm1_ds):
    """
    Compute virtual potential temperature perturbations
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain th0, qv0, th, and qv
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes thvpert (K)
    
    """

    # Compute base state thv

    p0 = cm1_ds['p0'].values
    T0 = isf.getTfromTheta(cm1_ds['th0'].values, p0)
    thv0 =isf.thetav(T0, p0, cm1_ds['qv0'].values)

    # Compute thv
 
    p = cm1_ds['p'].values
    T = isf.getTfromTheta(cm1_ds['th'].values, p)
    thv =isf.thetav(T, p, cm1_ds['qv'].values)

    # Add thvpert to cm1_ds
    
    cm1_ds['thvpert'] = xr.DataArray(thv - thv0, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['thvpert'].attrs['long_name'] = 'virtual potential temperature perturbations'   
    cm1_ds['thvpert'].attrs['units'] = 'K'

    return cm1_ds


def thetarho_prime(cm1_ds):
    """
    Compute density potential temperature perturbations
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain th0, qv0, th, qv, and hydrometeor mixing ratios
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes thrpert (K)
        
    """

    # Compute base state thr (assume base state hydrometeor mixing ratio is 0)

    p0 = cm1_ds['p0'].values
    T0 = isf.getTfromTheta(cm1_ds['th0'].values, p0)
    thr0 =isf.thetav(T0, p0, cm1_ds['qv0'].values)

    # Compute thr

    qt = cm1_ds['qv'].values
    for f in _qfields(cm1_ds):
        qt = qt + cm1_ds['q'+f].values

    p = cm1_ds['p'].values
    T = isf.getTfromTheta(cm1_ds['th'].values, p)
    thr =isf.thetarho(T, p, cm1_ds['qv'].values, qt)
    
    # Add thvpert to cm1_ds
    
    cm1_ds['thrpert'] = xr.DataArray(thr - thr0, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['thrpert'].attrs['long_name'] = 'density potential temperature perturbations'   
    cm1_ds['thrpert'].attrs['units'] = 'K'

    return cm1_ds


def thepert(cm1_ds):
    """
    Compute equivalent potential temperature and perturbations
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain th0, qv0, th, and qv
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes the and thepert (K)
        
    """
 
    # Compute base state the

    p0 = cm1_ds['p0'].values
    T0 = isf.getTfromTheta(cm1_ds['th0'].values, p0)
    the0 =isf.getthe(T0, p0, cm1_ds['qv0'].values)

    # Compute the
 
    p = cm1_ds['p'].values
    T = isf.getTfromTheta(cm1_ds['th'].values, p)
    the =isf.getthe(T, p, cm1_ds['qv'].values)

    # Add the0 and thepert to cm1_ds

    cm1_ds['the0'] = xr.DataArray(the0, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['the0'].attrs['long_name'] = 'base state equivalent potential temperature'   
    cm1_ds['the0'].attrs['units'] = 'K'

    cm1_ds['thepert'] = xr.DataArray(the - the0, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['thepert'].attrs['long_name'] = 'equivalent potential temperature perturbations'   
    cm1_ds['thepert'].attrs['units'] = 'K'

    return cm1_ds


def thetae(cm1_ds):
    """
    Compute equivalent potential temperature
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain th0, qv0, th, and qv
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes thetae (K)
        
    """
 
    p = cm1_ds['prs'].values    
    T = isf.getTfromTheta(cm1_ds['th'].values, p)
    thetae = isf.getthe(T, p, cm1_ds['qv'].values)

    # Add theta-e to cm1_ds

    cm1_ds['thetae'] = xr.DataArray(thetae, coords=cm1_ds['th'].coords, dims=cm1_ds['th'].dims)
    cm1_ds['thetae'].attrs['long_name'] = 'equivalent potential temperature'   
    cm1_ds['thetae'].attrs['units'] = 'K'

    return cm1_ds


def rh(cm1_ds):
    """
    Compute relative humidity (defined as qv / qvs)
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain th, p, and qv
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes RH (unitless)
        
    """

    p = cm1_ds['prs'].values
    try:
        T = cm1_ds['T'].values
    except KeyError:
        T = isf.getTfromTheta(cm1_ds['th'].values, p)

    qvs = isf.get_qvs(T, p)

    cm1_ds['RH'] = xr.DataArray(cm1_ds['qv'].values / qvs, coords=cm1_ds['prs'].coords, 
                                dims=cm1_ds['prs'].dims)
    cm1_ds['RH'].attrs['long_name'] = 'relative humidity (decimal)'
    cm1_ds['RH'].attrs['units'] = 'none'

    return cm1_ds


def supersat(cm1_ds):
    """
    Compute absolute supersaturation (defined as qv - qvs)
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain th, p, and qv
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes supersat (unitless)
        
    """

    p = cm1_ds['prs'].values
    try:
        T = cm1_ds['T'].values
    except KeyError:
        T = isf.getTfromTheta(cm1_ds['th'].values, p)

    qvs = isf.get_qvs(T, p)

    cm1_ds['supersat'] = xr.DataArray(cm1_ds['qv'].values - qvs, coords=cm1_ds['prs'].coords, 
                                      dims=cm1_ds['prs'].dims)
    cm1_ds['supersat'].attrs['long_name'] = 'absolute supersaturation w/r to liquid water'
    cm1_ds['supersat'].attrs['units'] = 'kg / kg'

    return cm1_ds


#---------------------------------------------------------------------------------------------------
# Microphysical Functions
#---------------------------------------------------------------------------------------------------

def qtot(cm1_ds):
    """
    Compute total hydrometeor mixing ratio
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain hydrometeor mixing ratios
    
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes qtot (kg / kg)
    
    """
    
    fields = _qfields(cm1_ds)
    qt = cm1_ds[fields[0]]
    for f in fields[1:]:
        qt = qt + cm1_ds[f]
        
    cm1_ds['qtot'] = qt
    cm1_ds['qtot'].attrs['long_name'] = 'total hydrometeor mass mixing ratio'
    cm1_ds['qtot'].attrs['units'] = 'kg / kg'
    
    return cm1_ds


def Dnr(cm1_ds, mur=0.0):
    """
    Compute number-weighted mean raindrop diameter (for qr > 0.01 g / kg)
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain rain mass and number mixing ratios
    mur : float, optional
        Rain drop size distribution shape parameter
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes Dnr (mm)
        
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
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain rain mass and number mixing ratios
    mur : float, optional
        Rain drop size distribution shape parameter
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes Dmr (mm)
        
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
    Compute number-weighted mean RIS diameter assuming mu = 0 (for qg > 0.01 g / kg)
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain RIS mass and number mixing ratios
    rhog : float, optional
        RIS density (kg / m^3)
    CG : float, optional
        RIS mass-diameter coefficient
    DG : float, optional
        RIS mass-diameter exponent
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes Dg (mm)

    """

    qg = cm1_ds['qg'].values
    ng = cm1_ds['ncg'].values

    lamg = (spec.gamma(1.0 + DG) * CG * ng / qg) ** (1./DG)
    Dg = ma.masked_array(1.0 / lamg, mask=(qg < 0.01e-3)) * 1e3

    cm1_ds['Dng'] = xr.DataArray(Dg, coords=cm1_ds['qg'].coords, dims=cm1_ds['qg'].dims)
    cm1_ds['Dng'].attrs['long_name'] = 'number-weighted mean RIS diameter'
    cm1_ds['Dng'].attrs['units'] = 'mm'

    return cm1_ds


def Dmg(cm1_ds, rhog, CG, DG):
    """
    Compute mass-weighted mean RIS diameter assuming mu = 0 (for qg > 0.01 g / kg)
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain RIS mass and number mixing ratios
    rhog : float, optional
        RIS density (kg / m^3)
    CG : float, optional
        RIS mass-diameter coefficient
    DG : float, optional
        RIS mass-diameter exponent
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes Dg (mm)

    """

    qg = cm1_ds['qg'].values
    ng = cm1_ds['ncg'].values

    lamg = (spec.gamma(1.0 + DG) * CG * ng / qg) ** (1./DG)
    Dg = ma.masked_array(spec.gamma(2.+DG) / (lamg * spec.gamma(1.+DG)), mask=(qg < 0.01e-3)) * 1e3

    cm1_ds['Dmg'] = xr.DataArray(Dg, coords=cm1_ds['qg'].coords, dims=cm1_ds['qg'].dims)
    cm1_ds['Dmg'].attrs['long_name'] = 'number-weighted mean RIS diameter'
    cm1_ds['Dmg'].attrs['units'] = 'mm'

    return cm1_ds


def Dm(cm1_ds, qfield='qr', nfield='nr', rho=1000., name='mmDr'):
    """
    Computes the mean mass diameter

    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset
    qfield : string, optional
        Name of mass mixing ratio field
    nfield : string, optional
        Name of number mixing ratio field
    rho : float, optional
        Density of hydrometeor species (kg / m^3)
    name : string, optional
        Name of field to add to dataset

    Returns
    -------
    cm1_ds : xarray dataset
        CM1 output file with the field 'Dm'

    """

    q = ma.masked_array(cm1_ds[qfield].values, mask=(cm1_ds[qfield] < 0.01e-3))
    n = cm1_ds[nfield].values
    Dm = ((6. * q) / (n * np.pi * rho)) ** (1./3.) * 1e3

    cm1_ds[name] = xr.DataArray(Dm, coords=cm1_ds[qfield].coords, dims=cm1_ds[qfield].dims)
    cm1_ds[name].attrs['long_name'] = 'mean mass diameter'
    cm1_ds[name].attrs['units'] = 'mm'

    return cm1_ds


def Vm(cm1_ds, qfield='qr', nfield='nr', mu=0, a=mf.Ar, b=mf.Br, c=mf.Cr, d=mf.Dr, name='Vmr'):
    """
    Computes the mean mass-weighted fallspeed

    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset
    qfield : string, optional
        Name of mass mixing ratio field
    nfield : string, optional
        Name of number mixing ratio field
    mu : float, optional
        PSD shape parameter
    a : float, optional
        Velocity-diameter relationship coefficient (m^{1-b} s^{-1})
    b : float, optional
        Velocity-diameter relationship exponent (unitless)
    c : float, optional
        Mass-diameter relationship coefficient
    d : float, optional
        Mass-diameter relationship exponent (unitless)
    name : string, optional
        Name of field to add to dataset

    Returns
    -------
    cm1_ds : xarray dataset
        CM1 output file with the field given by name

    """

    q = ma.masked_array(cm1_ds[qfield].values, mask=(cm1_ds[qfield] < 0.01e-3))
    n = cm1_ds[nfield].values
    lmda = mf.lamda(n, q, mu=mu, c=c, d=d)
    v = mf.Vm(lmda, mu=mu, a=a, b=b, d=d)

    cm1_ds[name] = xr.DataArray(v, coords=cm1_ds[qfield].coords, dims=cm1_ds[qfield].dims)
    cm1_ds[name].attrs['long_name'] = 'mean mass-weighted fallspeed'
    cm1_ds[name].attrs['units'] = 'm/s'

    return cm1_ds


def convert_mix(cm1_ds, field):
    """
    Convert mixing ratio to total amount in each grid cell
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain 'field' and air density (rho)
    field : string
        Name of mixing ratio field
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset with the field 'field_tot'
    
    """
    
    dx, dy, dz = _sgrid_spacing(cm1_ds)
    tdim = np.ones(cm1_ds[field].shape[0])
    vol4d = (dx[np.newaxis, np.newaxis, np.newaxis, :] * 
             dy[np.newaxis, np.newaxis, :, np.newaxis] * 
             dz[np.newaxis, :, np.newaxis, np.newaxis] * 
             tdim[:, np.newaxis, np.newaxis, np.newaxis] * 1e9)
    cm1_ds[field+'_tot'] = cm1_ds[field] * cm1_ds['rho'] * vol4d
    
    return cm1_ds


#---------------------------------------------------------------------------------------------------
# Dynamic Functions
#---------------------------------------------------------------------------------------------------

def vorts(cm1_ds):
    """
    Compute streamwise vorticity
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain all 3 velocity and vorticity components
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes vorts (/s)
        
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
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain horizontal velocity and vorticity components
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes hvorts (/s)
        
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
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain vertical velocity
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes dwdz (/s)
        
    """

    w = cm1_ds['w'].values
    try:
        zf = cm1_ds['zf'].values * 1e3
    except KeyError:
        zf = cm1_ds['nkp1'].values * 1e3

    dz = zf[1:] - zf[:-1]
    s = w.shape
    dz3d = np.tile(dz[np.newaxis, :, np.newaxis, np.newaxis], (s[0], dz.size, s[2], s[3]))

    dwdz = (w[:, 1:, :, :] - w[:, :-1, :, :]) / dz3d
    cm1_ds['dwdz'] = xr.DataArray(dwdz, coords=cm1_ds['winterp'].coords, 
                                  dims=cm1_ds['winterp'].dims)
    cm1_ds['dwdz'].attrs['long_name'] = 'vertical gradient of vertical velocity'
    cm1_ds['dwdz'].attrs['units'] = '/s'

    return cm1_ds


def vortz_stretch(cm1_ds):
    """
    Compute vertical vorticity stretching
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain zvort and dwdz
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes vortz_stretch (/s^2)
        
    """

    cm1_ds = dwdz(cm1_ds)
    cm1_ds['vortz_stretch'] = cm1_ds['zvort'] * cm1_ds['dwdz']
    cm1_ds['vortz_stretch'].attrs['long_name'] = 'vertical vorticity stretching'
    cm1_ds['vortz_stretch'].attrs['units'] = 's^-2'

    return cm1_ds


def vortz_tilt(cm1_ds):
    """
    Compute vertical vorticity tilting
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain horizontal velocity and vorticity components
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes vortz_tilt (/s^2)
        
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
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain vertical velocity
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes hgrad_w_mag (/s)
        
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
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain horizontal velocity components
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes conv2d (/s)
        
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
    Compute the Okubo-Weiss number assuming constant dx and dy
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain all horizontal velocity components and zvort
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes OW (/s^2)
        
    Notes
    -----
    See Markowski et al. 2011, EJSSM
        
    """

    warnings.warn('THERE IS A BUG IN THE OW FUNCTION')

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
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain all 3 vorticity components
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes vort_mag (/s)
        
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
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain horizontal vorticity components
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes hvort_mag (/s)
        
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
    cm1_ds : xarray dataset
        CM1 dataset. Must contain winterp and circ2

    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes wXcirc (m^3 / s^2)

    """

    w = cm1_ds['winterp'][:, 0, :, :].values
    w = w[:, np.newaxis, :, :]
    x1d = cm1_ds['xh'].values
    y1d = cm1_ds['yh'].values
    wavg = kf.avg_var(w, x1d, y1d, 2.0)
    C = cm1_ds['circ2'][:, 0, :, :].values
    C = C[:, np.newaxis, :, :]
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

    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain all 3 velocity and vorticity components
    zbot : float, optional
        Bottom of layer to compute SRH (km)
    ztop : float, optional
        Top of layer to compute SRH (km)
    dim : integer, optional
        Option to compute SRH using the 2D or 3D vorticity/velocity vector
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes SRH (m^2 / s^2)

    """

    warnings.warn('SRH function has not been tested yet')

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
    Compute advection of a field in a certain direction
    
    Parameters
    ----------
    cm1_ds : xarray dataset
        CM1 dataset. Must contain 3D wind vector and field
    field : string
        Name of field being advected
    wind : string
        Wind component advecting 'field' (options: 'u', 'v', or 'w')
        
    Returns
    -------
    cm1_ds : xarray dataset
        CM1 dataset that includes advection, named '<wind>adv<field>'
        
    Notes
    -----
    Only works for fields defined on the CM1 scalar grid
    
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
    ds1 : xarray dataset
        Dataset of first CM1 simulation
    ds2 : xarray dataset
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
    Compute supercell cold pool metrics
    
    Parameters
    ----------
    cm1_ds : xarray dataset 
        CM1 dataset
    coord : tuple or list 
        Supercell centroid [(x, y) ordered pair in km]
    r : float, optional
        Distance from supercell centroid to compute cold pool metrics (km)
    cp_thres : float, optional 
        Maximum potential temperature perturbation to define the cold pool (K)
    cp_z_max : float, optional 
        Maximum height AGL to search for cold pool when computing Bint3d (km)
        
    Returns
    -------
    min_thp : float 
        Minimum surface potential temperature perturbation (K)
    avg_thp : float 
        Average surface cold pool potential temperature perturbation (K)
    cp_ext : float 
        Surface cold pool areal extent (fraction)
    Bint2d : float 
        Integrated surface cold pool buoyancy (m / s^2)
    Bint3d : float 
        Integrated 3D cold pool buoyancy (m / s^2)
    totB2d : float 
        Integrated surface buoyancy (m / s^2)

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
    Compute supercell strength and tornadic potential metrics
    
    Parameters
    ----------
    cm1_ds : xarray dataset 
        CM1 dataset
    coord : list or tuple 
        Supercell centroid [(x, y) ordered pair in km]
    r : float, optional 
        Distance from supercell centroid to compute metrics (km)
    circ_field : string, optional 
        Name of Eulerian circulation field
        
    Returns
    -------
    wmax1km : float 
        Maximum 1-km updraft (m / s)
    zvmax1km : float 
        Maximum 1-km zvort (m / s)
    wmax : float 
        Maximum updraft (at any level)
    corr : float
        Correlation between 1-km W and 1-km zvort (unitless)
    circmax1km : float 
        Maximum Eulerian circulation at 1 km AGL (m^2 / s)
    circmaxsfc : float 
        Maximum Eulerian circulation at the lowest model level (m^2 / s)
    uhmax : float
        Maximum 2--5 km updraft helicity (m^2 / s^2)
    wcircdist : float 
        Distance between 1-km W maximum and near-surface circulation maximum (km)
    uhcircdist : float 
        Distance between supercell centroid and near-surface circulation maximum (km)
    up_Mf : float 
        Average 1-km updraft (W > 5 m/s) mass flux within r of supercell centroid (kg / s / m^4)
    Mfcircmax : float 
        1-km vertical mass flux computed using a 2-km radius ring centered on circmaxsfc 
        (kg / s / m^4)
    up_Mfcircmax : float 
        1-km up_Mf computed using a 2-km radius ring centered on circmaxsfc (kg / s / m^4)
    zvmaxsfc : float 
        Maximum zvort at LML (/ s)
    zvminsfc : float 
        Minimum zvort at LML (/ s)
    uparea : float 
        1-km updraft (w > 5 m/s) (km^2)
        
    Notes
    -----
    Updraft mass flux formula comes from Klees et al. (2016)

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
    
    Parameters
    ----------
    cm1_ds : xarray dataset 
        CM1 dataset
    coord : list or tuple 
        Supercell centroid [(x, y) ordered pair in km]
    r : float, optional 
        Distance from supercell centroid to search for TLVs (km)
    zv_thres : float, optional
        Vertical vorticity threshold to identify TLVs (s^-1)
    prspert_thres : float, optional 
        Pressure perturbation threshold to identify TLVs (Pa)
    
    Returns
    -------
    x : float 
        X coordinate of TLV (km)
    y : float 
        Y coordinate of TLV (km)
    zv : float 
        Vertical vorticity of TLV (s^-1)
    ppert : float 
        Pressure perturbation of TLV (Pa)
        
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
    Identify the FFD and RFD
    
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
    
    Parameters
    ----------
    cm1_ds : xarray dataset 
        CM1 dataset
    x_pts : float 
        X-coordinates of input data points (km)
    y_pts : float 
        Y-coordinates of input data points (km)
    ffd_max_dist : float, optional 
        Maximum distance a point within the FFD is allowed to lie from the mesocyclone center (km)
    rfd_max_dist : float, optional 
        Maximum distance a point within the RFD is allowed to lie from the mesocyclone center (km)
    meso_height : float, optional 
        Height AGL where the mesocyclone center is diagnosed
    ref_thres : float, optional 
        Minimum reflectivity of points considered to be in the precipitation shield (dBZ)
    search_x : float, optional 
        X-coordinates of the lower-left and upper-right corners of the box to search for the 
        mesocyclone center in (km)
    search_y : float, optional 
        Y-coordinates of the lower-left and upper-right corners of the box to search for the 
        mesocyclone center in (km)
    
    Returns
    -------
    ffd_ind : integer 
        Indices of the data points (x_pts, y_pts) that lie within the FFD
    rfd_ind : integer 
        Indices of the data points (x_pts, y_pts) that lie within the RFD
    x_meso_ctr : float 
        X-coordinate of mesocyclone center (km)
    y_meso_ctr : float 
        Y-coordinate of mesocyclone center (km)
    m_major_axis : float 
        Slope of line that passes through the major axis of the storm
    m_spine : float 
        Slope of the line orthogonal to m_major_axis
    
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