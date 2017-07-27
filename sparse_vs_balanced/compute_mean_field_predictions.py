import sys, pathlib, os
sep = os.path.sep # Ms-Win vs UNIX
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from neural_network_dynamics.theory.mean_field import find_fp
from model import Model; Model2 = Model.copy()

PRECISION = 60 # 1 to debug, 100 for results

def my_logspace(x1, x2, n):
    return np.logspace(np.log(x1)/np.log(10), np.log(x2)/np.log(10), n)

Model2['COEFFS_RecExc'] = np.load('sparse_vs_balanced/data/COEFFS_RecExc.npy')
Model2['COEFFS_RecInh'] = np.load('sparse_vs_balanced/data/COEFFS_RecInh.npy')

# importing the levels of the numerical simulations
from sparse_vs_balanced.varying_AffExc import get_scan
# _, FAFF, _, DATA = get_scan({},
#                filename='sparse_vs_balanced/data/varying_AffExc.zip')

FAFF2 = my_logspace(3, 25, PRECISION+1)
FE, FI, IRATIO = [], [], []
for faff in FAFF2:
    output = find_fp(Model2,
                     KEY1='RecExc', KEY2='RecInh',
                     F1 = np.logspace(-3., 2.2, 10*PRECISION),
                     F2 = np.logspace(-3., 2.5, 50*PRECISION),
                     KEY_RATES1 = ['AffExc'], VAL_RATES1=[faff],
                     KEY_RATES2 = ['AffExc'], VAL_RATES2=[faff],
                     plot=False)
    FE.append(output['F_RecExc'])
    FI.append(output['F_RecInh'])
    # on excitatory currents
    Isyn = output['Isyn_RecExc']
    IRATIO.append(np.abs(Isyn['RecInh']/(Isyn['RecExc']+Isyn['AffExc'])))
np.save('sparse_vs_balanced/data/varying_AffExc_theory.npy',
        [FAFF2, FE, FI, IRATIO])    
