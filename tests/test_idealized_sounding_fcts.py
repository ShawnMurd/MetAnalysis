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


def test_get_es():
    esl = isf.get_es(265.0, sfc='l')
    esi = isf.get_es(265.0, sfc='i')
    assert esl == pytest.approx(331.4659001, abs=0.01)
    assert esi == pytest.approx(305.7150278, abs=0.01)


def test_get_qvs():
    qvl = isf.get_qvs(285.0, 90000.0, sfc='l')
    qvi = isf.get_qvs(265.0, 90000.0, sfc='i')
    assert qvl == pytest.approx(0.009749736, abs=0.00002)
    assert qvi == pytest.approx(0.00211993744596, abs=0.00002)
    

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


def test_getTd():
    Td = isf.getTd(285.0, 90000.0, 0.005)
    assert Td == pytest.approx(275.384, abs=0.05)


def test_buoy():
    B = isf.buoy(300.0, 90000.0, 0.0075, 295.0, 89500.0, 0.005)
    assert B == pytest.approx(0.1653106, 0.0001)
    

def test_getthe():
    thetae = isf.getthe(285.0, 90000.0, 0.005)
    assert thetae == pytest.approx(308.515, abs=0.005)
   

#---------------------------------------------------------------------------------------------------
# Test Sounding Parameter Functions
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

def test_getcape():
    
    # Read in CM1 Weisman-Klemp sounding

    wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')    

    T = wk_df['theta (K)'].values * wk_df['pi'].values
    qv = wk_df['qv (kg/kg)'].values
    p = wk_df['prs (Pa)'].values
    
    # "True" values from fortran version (run on PSU-ICDS Roar)
    
    truth = {}
    tlab = ['sb_pseudo', 'sb_rev', 'sb_pseudo_ice', 'mu_pseudo', 'ml_pseudo']
    
    truth['sb_pseudo']     = (1884.456, 45.003, 958.656, 1549.249, 11432.290)
    truth['sb_rev']        = (1256.796, 49.647, 958.656, 1697.166, 11083.252)
    truth['sb_pseudo_ice'] = (2022.359, 45.003, 958.656, 1549.249, 11674.379)
    truth['mu_pseudo']     = (2384.172, 0.7342, 1232.674, 1306.146, 11967.149)
    truth['ml_pseudo']     = (1974.274, 31.535, 1008.741, 1502.865, 11548.418)
    
    # "True" buoyancy values from getB_sfm5282.F
    
    trueB = np.loadtxt('../sample_data/wk_sb_pseudo_buoy.txt')
    
    for (src, adbt, t) in zip(['sfc', 'sfc', 'sfc', 'mu', 'ml'], [1, 2, 3, 1, 1], tlab):
        gc_out = isf.getcape(p, T, qv, source=src, adiabat=adbt)
        assert gc_out[0] == pytest.approx(truth[t][0], rel=0.005)
        assert gc_out[1] == pytest.approx(truth[t][1], rel=0.01)
        assert gc_out[2] == pytest.approx(truth[t][2], rel=0.005)
        assert gc_out[3] == pytest.approx(truth[t][3], rel=0.005)
        assert gc_out[4] == pytest.approx(truth[t][4], rel=0.005)
        
    B = isf.getcape(p, T, qv, returnB=True)[5]
    np.testing.assert_allclose(B, trueB, atol=0.0005)

#---------------------------------------------------------------------------------------------------
# Test Functions Related to Vertical Profiles of Sounding Parameters
#---------------------------------------------------------------------------------------------------

def test_calcsound_out_to_df():
    df = isf.calcsound_out_to_df('../sample_data/oun1999050318.out')
    DCAPE = np.array([0.0, 35.6, 49.3, 60.1, 66.1])
    assert len(df) == 76
    np.testing.assert_allclose(df['DCAPE'].values[:5], DCAPE, atol=0.05)


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

    param_truth = isf.getcape(p[4:], T[4:], qv[4:])

    param_name = ['CAPE', 'CIN', 'zlcl', 'zlfc', 'zel']
    for i, key in enumerate(param_name):
        assert param[key][4] == pytest.approx(param_truth[i], 0.01)


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
    

#---------------------------------------------------------------------------------------------------
# McCaul-Weisman Analytic Sounding Tests
#---------------------------------------------------------------------------------------------------

def test_getqv_from_thetae():
    
    T = 290.0
    p = 95000.0
    qv_truth = 0.01
    the = isf.getthe(T, p, qv_truth)
    qv = isf.getqv_from_thetae(T, p, the)
    
    assert qv == pytest.approx(qv_truth, abs=0.001)
    

"""
End test_idealized_sounding_fcts.py
""" 
