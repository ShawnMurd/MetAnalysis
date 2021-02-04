"""
Timing Tests for Idealized Sounding Functions

Shawn Murdzek
sfm5282@psu.edu
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import timeit


#---------------------------------------------------------------------------------------------------
# Timing Tests for Various Function Calls
#---------------------------------------------------------------------------------------------------

# Weisman-Klemp analytic sounding

stmt = '''
import numpy as np
import MetAnalysis.src.idealized_sounding_fcts as isf
z = np.arange(10, 12000, 50)
isf_df = isf.weisman_klemp(z, cm1_out='./weisman_klemp_cm1_in')
'''

print('with JIT = %.6f' % timeit.timeit(stmt, number=100))

stmt = '''
import numpy as np
import MetAnalysis.src.idealized_sounding_fcts as isf
z = np.arange(10, 12000, 50)
isf_df = isf.weisman_klemp_noJIT(z, cm1_out='./weisman_klemp_cm1_in')
'''

print('without JIT = %.6f' % timeit.timeit(stmt, number=100))


"""
End idealized_sounding_fcts_timing.py
"""