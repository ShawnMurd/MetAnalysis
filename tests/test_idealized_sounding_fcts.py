"""
Tests for Idealized Sounding Functions

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import MetAnalysis.src.idealized_sounding_fcts as isf
import pytest


#---------------------------------------------------------------------------------------------------
# Test Thermodynamic Functions
#---------------------------------------------------------------------------------------------------

def test_exner():
    pi = isf.exner(85000.0)
    assert pi == pytest.approx(0.95464068, 0.001)


def test_theta():
    th = isf.theta(273.0, 80000.0)
    assert th == pytest.approx(290.966533, 0.001)


def test_getTfromTheta():
    T = isf.getTfromTheta(290.966533, 80000.0)
    assert T == pytest.approx(273.0, 0.001)


def get_es():
    es = isf.get_es(285.0)
    assert es == pytest.approx(1387.743087, 0.01)


def get_qvl():
    qvl = isf.get_qvl(285.0, 90000.0)
    assert qvl == pytest.approx(0.009749736, 0.00001)


def test_getTv():
    Tv = isf.getTv(273.0, 0.005)
    assert Tv == pytest.approx(273.8254766, 0.001)


def test_thetav():
    thv = isf.thetav(273.0, 80000.0, 0.005)
    assert thv == pytest.approx(291.8463354, 0.001)


def test_getTfromTv():
    T = isf.getTfromTv(273.8254766, 0.005)
    assert T == pytest.approx(273.0, 0.001)


def test_getqv():
    qv = isf.getqv(0.75, 285.0, 80000.0)
    assert qv == pytest.approx(0.00823486627799, 0.00005)


def test_buoy():
    B = isf.buoy(300.0, 90000.0, 0.0075, 295.0, 89500.0, 0.005)
    assert B == pytest.approx(0.1653106, 0.0001)
    

#---------------------------------------------------------------------------------------------------
# Test Functions Related to Vertical Profiles of Sounding Parameters
#---------------------------------------------------------------------------------------------------

def test_sounding_pressure():

    # Read in CM1 Weisman-Klemp sounding

    wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')    

    z = wk_df['z (m)'].values
    th = wk_df['theta (K)'].values
    qv = wk_df['qv (kg/kg)'].values
    p = wk_df['prs (Pa)'].values

    # Compute pressures
    
    p_isf = isf.sounding_pressure(z, th, qv, p[0])

    np.testing.assert_allclose(p_isf, p, atol=0.02)

def test_sounding_height():

    # Read in CM1 Weisman-Klemp sounding

    wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')    

    z = wk_df['z (m)'].values
    th = wk_df['theta (K)'].values
    qv = wk_df['qv (kg/kg)'].values
    p = wk_df['prs (Pa)'].values

    # Compute pressures
    
    z_isf = isf.sounding_height(p, th, qv, z[0])

    np.testing.assert_allclose(z_isf, z, atol=0.02)


def test_calcsound_out_to_df():
    df = isf.calcsound_out_to_df('../sample_data/oun1999050318.out')
    DCAPE = np.array([0.0, 35.6, 49.3, 60.1, 66.1])
    assert len(df) == 76
    np.testing.assert_allclose(df['DCAPE'].values[:5], DCAPE, atol=0.05)

'''
def test_effect_inflow():

    # Read in CM1 Weisman-Klemp sounding

    wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')

    T = wk_df['theta (K)'].values * wk_df['pi'].values
    qv = wk_df['qv (kg/kg)'].values
    p = wk_df['prs (Pa)'].values

    # Compute effective inflow layer

    p_top1, p_bot1 = isf.effect_inflow(p, T, qv)
    p_top2, p_bot2 = isf.effect_inflow(p, T, qv, min_cape=100, max_cin=30)

    assert p_top1 == pytest.approx(68147.65, 0.1)
    assert p_top2 == pytest.approx(68147.65, 0.1)
    assert p_bot1 == pytest.approx(99437.76, 0.1)
    assert p_bot2 == pytest.approx(95014.45, 0.1)


def test_param_vprof():
    
    # Read in CM1 Weisman-Klemp sounding

    wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')

    T = wk_df['theta (K)'].values * wk_df['pi'].values
    qv = wk_df['qv (kg/kg)'].values
    p = wk_df['prs (Pa)'].values

    # Compute vertical profile of sounding parameters

    param, B = isf.param_vprof(p, T, qv, 100000.0, 70000.0)

    # Compute sounding parameters using getcape

    param_truth = gc.getcape(1, 1, p[4:] * 0.01, T[4:] - 273.15, qv[4:])

    param_name = ['CAPE', 'CIN', 'zlcl', 'zlfc', 'zel']
    for i, key in enumerate(param_name):
        assert param[key][4] == pytest.approx(param_truth[i], 0.01)
'''

#---------------------------------------------------------------------------------------------------
# Weisman-Klemp Analytic Sounding Test
#---------------------------------------------------------------------------------------------------

def test_weisman_klemp():

    # Read in CM1 Weisman-Klemp sounding

    cm1_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')

    T = cm1_df['theta (K)'].values * cm1_df['pi'].values
    qv = cm1_df['qv (kg/kg)'].values
    p = cm1_df['prs (Pa)'].values

    # Compute Weisman-Klemp sounding using MetAnalysis

    isf_df = isf.weisman_klemp(cm1_df['z (m)'].values, 
                               cm1_out='../sample_data/weisman_klemp_cm1_in')

    # Compare T, qv, and p profiles

    np.testing.assert_allclose(isf_df['T'][1:].values, T, atol=0.05)
    np.testing.assert_allclose(isf_df['qv'][1:].values, qv, atol=0.00005)
    np.testing.assert_allclose(isf_df['p'][1:].values, p, atol=2.5)
    

"""
End test_idealized_sounding_fcts.py
""" 
