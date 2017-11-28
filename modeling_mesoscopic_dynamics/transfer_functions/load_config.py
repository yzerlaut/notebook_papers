import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from single_cell_models.cell_library import get_neuron_params
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from transfer_functions.theoretical_tools import pseq_params, TF_my_template
from transfer_functions.tf_simulation import reformat_syn_parameters

def load_transfer_functions(NRN1, NRN2, NTWK):
    """
    returns the two transfer functions of the mean field model
    """

    # NTWK
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    
    # NRN1
    params1 = get_neuron_params(NRN1, SI_units=True)
    reformat_syn_parameters(params1, M)
    try:
        P1 = np.load('data/'+NRN1+'_'+NTWK+'_fit.npy')
        
        params1['P'] = P1
        def TF1(fe, fi):
            return TF_my_template(fe, fi, *pseq_params(params1))
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN1 not available  ====================')
        print('=======================================================')

    # NRN1
    params2 = get_neuron_params(NRN2, SI_units=True)
    reformat_syn_parameters(params2, M)
    try:
        P2 = np.load('data/'+NRN2+'_'+NTWK+'_fit.npy')
        params2['P'] = P2
        def TF2(fe, fi):
            return TF_my_template(fe, fi, *pseq_params(params2))
    except IOError:
        print('=======================================================')
        print('=====  fit for NRN2 not available  ====================')
        print('=======================================================')
        
    return TF1, TF2

if __name__=='__main__':
    NRN1, NRN2, NTWK = 'RS-cell', 'FS-cell', 'CONFIG1'
    P1 = np.load('../transfer_functions/data/'+NRN1+'_'+NTWK+'_fit.npy')
    P2 = np.load('../transfer_functions/data/'+NRN2+'_'+NTWK+'_fit.npy')
    # print('%0.2e' % 1e3*P1[0])
    for P in [P1, P2]:
        S = str(round(1e3*P[0], 1))
        for p in P[1:]:
            s = '%0.1e' % p
            S+=' & '+s.replace('e-0', 'e-')+''
        print(S)
