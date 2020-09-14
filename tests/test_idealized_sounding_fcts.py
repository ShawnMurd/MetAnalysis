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
import MetAnalysis.src.getcape as gc
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


"""
End test_idealized_sounding_fcts.py
""" 
