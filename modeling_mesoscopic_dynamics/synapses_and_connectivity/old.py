"""
Some configuration of neuronal properties so that we pick up
within this file
"""
from __future__ import print_function
import numpy as np


def get_connectivity_and_synapses_matrix(NAME, number=2, SI_units=False):


    # creating empty arry of objects (future dictionnaries)
    M = np.empty((number, number), dtype=object)

    if NAME=='Vogels-Abbott':
        exc_pop = {'p_conn':0.02, 'Q':7., 'Tsyn':5., 'Erev':0.}
        inh_pop = {'p_conn':0.02, 'Q':67., 'Tsyn':10., 'Erev':-80.}
        M[:,0] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : exc
        M[:,1] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : inh
        M[0,0]['name'], M[1,0]['name'] = 'ee', 'ie'
        M[0,1]['name'], M[1,1]['name'] = 'ei', 'ii'

        # in the first element we put the network number and connectivity information
        M[0,0]['Ntot'], M[0,0]['gei'] = 5000, 0.2
        
    elif NAME=='CONFIG1':
        exc_pop = {'p_conn':0.1, 'Q':2., 'Tsyn':5., 'Erev':0.}
        inh_pop = {'p_conn':0.1, 'Q':5., 'Tsyn':5., 'Erev':-80.}
        M[:,0] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : exc
        M[:,1] = [exc_pop.copy(), inh_pop.copy()] # post-synaptic : inh
        M[0,0]['name'], M[1,0]['name'] = 'ee', 'ie'
        M[0,1]['name'], M[1,1]['name'] = 'ei', 'ii'
        
        # in the first element we put the network number and connectivity information
        M[0,0]['Ntot'], M[0,0]['gei'] = 10000, 0.2
        M[0,0]['ext_drive'] = 0.5 # we also store here the choosen excitatory drive 
        M[0,0]['afferent_exc_fraction'] = 0.7 # we also store here the choosen excitatory drive 
        
        
    else:
        print('====================================================')
        print('------------ NETWORK NOT RECOGNIZED !! ---------------')
        print('====================================================')

    if SI_units:
        print('synaptic network parameters in SI units')
        for m in M.flatten():
            m['Q'] *= 1e-9
            m['Erev'] *= 1e-3
            m['Tsyn'] *= 1e-3
    else:
        print('synaptic network parameters --NOT-- in SI units')

    return M

if __name__=='__main__':

    print(__doc__)

    M = get_connectivity_and_synapses_matrix('Vogels-Abbott')

    print('synapses of the exc. pop. (pop. 0) : M[:,0]')
    print(M[:,0])
    print('synapses of the inh. pop. (pop. 1) : M[:,1]')
    print(M[:,1])
    
