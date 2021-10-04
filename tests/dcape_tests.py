"""
Test MetAnalysis DCAPE Function

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
from metpy.plots import SkewT
import MetAnalysis.src.idealized_sounding_fcts as isf


#---------------------------------------------------------------------------------------------------
# Compare DCAPE Function Output to Kerry Emanuel's calcsound Program
#---------------------------------------------------------------------------------------------------

# Extract Weisman-Klemp analytic sounding

wk_df = pd.read_csv('../sample_data/cm1_weisman_klemp_snd.csv')    

T = wk_df['theta (K)'].values * wk_df['pi'].values
qv = wk_df['qv (kg/kg)'].values
p = wk_df['prs (Pa)'].values
Td = isf.getTd(T, p, qv)
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
ax.plot(cs_dcape, cs_p, color='r', lw=2, ls='--', label='calcsound')
ax.set_xlabel('DCAPE (J kg$^{-1}$)', size=14)
ax.set_ylabel('pressure (hPa)', size=14)
ax.grid()
ax.legend()
ax.set_ylim(1000., 600.)
plt.show()

# Plot downdraft parcel from k = kmax on a skew-t, log-p

out = isf.getdcape(p, T, qv, kmax, returnTHV=True, returnQV=True)
Tv_p = isf.getTfromTheta(out[2], p[:(kmax+1)])
T_p = isf.getTfromTv(Tv_p, out[3])
Td_p = isf.getTd(T_p, p[:(kmax+1)], out[3])

fig = plt.figure(figsize=(10, 10))
skew = SkewT(fig)

plt.plot(T-273.15, 0.01*p, 'r', lw=3)
plt.plot(Td-273.15, 0.01*p, 'b', lw=3)
plt.plot(T_p-273.15, 0.01*p[:(kmax+1)], 'k', lw=3)
plt.plot(Td_p-273.15, 0.01*p[:(kmax+1)], 'k', lw=3)

skew.plot_dry_adiabats()
skew.plot_mixing_lines()
skew.plot_moist_adiabats()

plt.xlabel('temperature (deg C)', size=14)
plt.ylabel('pressure (hPa)', size=14)
plt.show()


"""
End dcape_tests.py
"""