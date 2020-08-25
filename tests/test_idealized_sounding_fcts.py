"""
Tests for Idealized Sounding Functions

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import MetAnalysis.src.idealized_sounding_fcts as isf
import MetAnalysis.src.getcape as gc
import pytest


#---------------------------------------------------------------------------------------------------
# Test Thermodynamic Functions
#---------------------------------------------------------------------------------------------------

def test_theta():
    th = isf.theta(273.0, 80000.0)
    assert th == pytest.approx(290.966533, 0.001)


def test_thetav():
    thv = isf.thetav(273.0, 80000.0, 0.005)
    assert thv == pytest.approx(291.8463354, 0.001)


def test_getTv():
    Tv = isf.getTv(273.0, 0.005)
    assert Tv == pytest.approx(273.8254766, 0.001)


def test_getTfromTv():
    T = isf.getTfromTv(273.8254766, 0.005)
    assert T == pytest.approx(273.0, 0.001)

def test_buoy():
    B = isf.buoy(300.0, 90000.0, 0.0075, 295.0, 89500.0, 0.005)
    assert B == pytest.approx(0.1653106, 0.0001)

def test_exner():
    pi = isf.exner(85000.0)
    assert pi == pytest.approx(0.95464068, 0.001)


#---------------------------------------------------------------------------------------------------
# Test Functions Related to Vertical Profiles of Sounding Parameters
#---------------------------------------------------------------------------------------------------

def test_sounding_pressure():
    return None
