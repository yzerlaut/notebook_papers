import numpy as np
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from transfer_functions.load_config import load_transfer_functions
from mean_field.master_equation import find_fixed_point_first_order
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from single_cell_models.cell_library import get_neuron_params
from transfer_functions.theoretical_tools import get_fluct_regime_vars, pseq_params
from transfer_functions.tf_simulation import reformat_syn_parameters

# RING NETWORK PARAMETERS
import ring_model.ring_models as ring
# STIM PARAMETERS
import ring_model.stimulations as stim

########################################################################
##----------------------------------------------------------------------
##  Then rate model for the network (1st order of El Boustani et al. 2009)
##----------------------------------------------------------------------
########################################################################

def Euler_method_for_ring_model(NRN1, NRN2, NTWK, RING, STIM, BIN=5e-3,\
                                custom_ring_params={}, custom_stim_params={}):
    """
    Given two afferent rate input excitatory and inhibitory respectively
    this function computes the prediction of a first order rate model
    (e.g. Wilson and Cowan in the 70s, or 1st order of El Boustani and
    Destexhe 2009) by implementing a simple Euler method
    IN A 2D GRID WITH LATERAL CONNECTIVITY
    the number of laterally connected units is 'connected_neighbors'
    there is an exponential decay of the strength 'decay_connect'
    ----------------------------------------------------------------
    the core of the formalism is the transfer function, see Zerlaut et al. 2015
    it can also be Kuhn et al. 2004 or Amit & Brunel 1997
    -----------------------------------------------------------------
    nu_0 is the starting value value of the recurrent network activity
    it should be the fixed point of the network dynamics
    -----------------------------------------------------------------
    t is the discretization used to solve the euler method
    BIN is the initial sampling bin that should correspond to the
    markovian time scale where the formalism holds (~5ms)
    
    conduction_velocity=0e-3, in ms per pixel !!!
    """
    
    print('----- loading parameters [...]')
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    ext_drive = M[0,0]['ext_drive']
    params = get_neuron_params(NRN2, SI_units=True)
    reformat_syn_parameters(params, M)
    afferent_exc_fraction = M[0,0]['afferent_exc_fraction']

    print('----- ## we look for the fixed point [...]')
    fe0, fi0 = find_fixed_point_first_order(NRN1, NRN2, NTWK, exc_aff=ext_drive)
    muV0, _, _, _ = get_fluct_regime_vars(fe0+ext_drive, fi0, *pseq_params(params))
    
    print('----- ## we load the transfer functions [...]')
    TF1, TF2 = load_transfer_functions(NRN1, NRN2, NTWK)

    print('----- ## ring initialisation [...]')
    X, Xn_exc, Xn_inh, exc_connected_neighbors, exc_decay_connect, inh_connected_neighbors,\
        inh_decay_connect, conduction_velocity = ring.pseq_ring_params(RING, custom=custom_ring_params)
    
    print('----- ## stimulation initialisation [...]')
    t, Fe_aff = stim.get_stimulation(X, STIM, custom=custom_stim_params)
    Fi_aff = 0*Fe_aff # no afferent inhibition yet
    
    print('----- ## model initialisation [...]')
    Fe, Fi, muVn = 0*Fe_aff+fe0, 0*Fe_aff+fi0, 0*Fe_aff+muV0

    print('----- starting the temporal loop [...]')
    dt = t[1]-t[0]
    
    # constructing the Euler method for the activity rate
    for i_t in range(len(t)-1): # loop over time

        for i_x in range(len(X)): # loop over pixels
            
            # afferent excitation + exc DRIVE
            fe = (1-afferent_exc_fraction)*Fe_aff[i_t, i_x]+ext_drive # common both to exc and inh
            fe_pure_exc = (2*afferent_exc_fraction-1)*Fe_aff[i_t, i_x] # only for excitatory pop
            fi = 0 #  0 for now.. !

            # EXC --- we add the recurrent activity and the lateral interactions
            for i_xn in Xn_exc: # loop over neighboring excitatory pixels
                # calculus of the weight
                exc_weight = ring.gaussian_connectivity(i_xn, 0., exc_decay_connect)
                # then we have folded boundary conditions (else they donot
                # have all the same number of active neighbors !!)
                i_xC = (i_x+i_xn)%(len(X))

                if i_t>int(abs(i_xn)/conduction_velocity/dt):
                    it_delayed = i_t-int(abs(i_xn)/conduction_velocity/dt)
                else:
                    it_delayed = 0

                fe += exc_weight*Fe[it_delayed, i_xC]
                
            # INH --- we add the recurrent activity and the lateral interactions
            for i_xn in Xn_inh: # loop over neighboring inhibitory pixels
                # calculus of the weight
                inh_weight = ring.gaussian_connectivity(i_xn, 0., inh_decay_connect)
                # then we have folded boundary conditions (else they donot
                # have all the same number of active neighbors !!)
                i_xC = (i_x+i_xn)%(len(X))
                
                if i_t>int(abs(i_xn)/conduction_velocity/dt):
                    it_delayed = i_t-int(abs(i_xn)/conduction_velocity/dt)
                else:
                    it_delayed = 0
                    
                fi += inh_weight*Fi[it_delayed, i_xC]

            ## NOTE THAT NO NEED TO SCALE : fi*= gei*pconnec*Ntot and fe *= (1-gei)*pconnec*Ntot
            ## THIS IS DONE IN THE TRANSFER FUNCTIONS !!!!
                
            # now we can guess the rate model output
            muVn[i_t+1, i_x], _, _, _ = get_fluct_regime_vars(fe, fi, *pseq_params(params))
            Fe[i_t+1, i_x] = Fe[i_t, i_x] + dt/BIN*( TF1(fe+fe_pure_exc,fi) - Fe[i_t, i_x])
            Fi[i_t+1, i_x] = Fi[i_t, i_x] + dt/BIN*( TF2(fe,fi) - Fi[i_t, i_x])

    print('----- temporal loop over !')

    return t, X, Fe_aff, Fe, Fi, np.abs((muVn-muV0)/muV0)



