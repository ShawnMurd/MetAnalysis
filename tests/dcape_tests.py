"""
Test MetAnalysis DCAPE Function

Current Problems:
    1. Why is DCAPE_max < 0? Shouldn't it be positive definite?
    2. Why is DCAPE so small from calcsound? Examining CAPE and CIN from wk82.out shows that both
        CAPE and CIN are way smaller than there should be. I used the version of calcsound on the
        METEO Linux System, but I thought had an updated version somewhere that could handle more
        vertical levels (the default version can only handle a double-digit number of vertical 
        levels). Maybe this is on Roar somewhere?
    3. DCAPE values from MetAnalysis are likely too large. Revisiting my (MSP)^2 presentation from
        1/14/2021 shows that at 600 hPa, the maximum DCAPE from my old McCaul-Weisman base states
        is ~950 J/kg.

Shawn Murdzek
sfm5282@psu.edu
Date Creted: 30 September 2021
"""

#---------------------------------------------------------------------------------------------------
# Import  Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MetAnalysis.src.idealized_sounding_fcts as isf


#---------------------------------------------------------------------------------------------------
# Compare DCAPE Function Output to Kerry Emanuel's calcsound Program
#---------------------------------------------------------------------------------------------------

# Extract Weisman-Klemp analytic sounding

wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')    

T = wk_df['theta (K)'].values * wk_df['pi'].values
qv = wk_df['qv (kg/kg)'].values
p = wk_df['prs (Pa)'].values
th = isf.theta(T, p)
z = isf.sounding_height(p, th, qv, 0.0)

# Create input file for Kerry Emanuel's calcsound program (program is found on METEO Linux servers)

#isf.create_calcsound_input(p, T, qv, 'wk82.dat')

# Read in output from calcsound

cs_df = isf.calcsound_out_to_df('wk82.out')
cs_dcape = cs_df['DCAPE'].values
cs_p = cs_df['p'].values

# Compute DCAPE using MetAnalysis

kmax = 45
kmin = 5
ma_dcape_t = np.zeros(kmax-kmin+1)
ma_dcape_m = np.zeros(kmax-kmin+1)
for k in range(kmin, kmax+1):
    ma_dcape_t[k-kmin], ma_dcape_m[k-kmin] = isf.getdcape(p, T, qv, k)
    
# Plot results

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
ax.plot(ma_dcape_t, 0.01*p[kmin:(kmax+1)], color='b', lw=2, label='MetAnalysis (tot)')
ax.plot(ma_dcape_m, 0.01*p[kmin:(kmax+1)], color='g', lw=2, label='MetAnalysis (max)')
ax.plot(cs_dcape, cs_p, color='r', lw=2, label='calcsound')
ax.set_xlabel('DCAPE (J kg$^{-1}$)', size=14)
ax.set_ylabel('pressure (hPa)', size=14)
ax.grid()
ax.legend()
ax.set_ylim(1000., 600.)
plt.show()


"""
End dcape_tests.py
"""