import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from mean_field.master_equation import *
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from single_cell_models.cell_library import get_neuron_params
from transfer_functions.theoretical_tools import get_fluct_regime_vars, pseq_params
from transfer_functions.tf_simulation import reformat_syn_parameters
import numpy as np

def func(t):
    return 2.*np.exp(-(t-1)**2/.1)

def run_mean_field(NRN1, NRN2, NTWK, array_func,\
                   T=5e-3, dt=1e-4, tstop=2, extended_output=False,\
                   afferent_exc_fraction=0.,
                   ext_drive_change=0.):

    # find external drive
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    ext_drive = M[0,0]['ext_drive']
    if afferent_exc_fraction<0.5:
        afferent_exc_fraction = M[0,0]['afferent_exc_fraction']

    X0 = find_fixed_point_first_order(NRN1, NRN2, NTWK, exc_aff=ext_drive+ext_drive_change,\
                                      verbose=False)

    TF1, TF2 = load_transfer_functions(NRN1, NRN2, NTWK)

    t = np.arange(int(tstop/dt))*dt
    
    def dX_dt_scalar(X, t=0):
        exc_aff = ext_drive+ext_drive_change+(1-afferent_exc_fraction)*array_func(t)
        pure_exc_aff = (2*afferent_exc_fraction-1)*array_func(t) # what needs to be added
        return build_up_differential_operator_first_order(TF1, TF2, T=T)(X,\
                                                exc_aff=exc_aff, pure_exc_aff=pure_exc_aff)
        
    fe, fi = odeint(dX_dt_scalar, X0, t).T

    if extended_output:
        params = get_neuron_params(NRN2, SI_units=True)
        reformat_syn_parameters(params, M)

        exc_aff = ext_drive+ext_drive_change+(1-afferent_exc_fraction)*array_func(t)
        pure_exc_aff = (2*afferent_exc_fraction-1)*array_func(t) # what needs to be added


        # excitatory neurons have more excitation
        muV_e, sV_e, muGn_e, TvN_e = get_fluct_regime_vars(\
                                                           fe+exc_aff+pure_exc_aff,\
                                                           fi, *pseq_params(params))
        muV_i, sV_i, muGn_i, TvN_i = get_fluct_regime_vars(\
                                                           fe+exc_aff,\
                                                           fi, *pseq_params(params))
        muV, sV, muGn, TvN = .8*muV_e+.2*muV_i, .8*sV_e+.2*sV_i,\
                                        .8*muGn_e+.2*muGn_i, .8*TvN_e+.2*TvN_i,
            
        return t, fe, fi, muV, sV, muGn, TvN
    else:
        return t, fe, fi

def run_mean_field_extended(NRN1, NRN2, NTWK, array_func,\
                            Ne=8000, Ni=2000, T=5e-3,\
                            afferent_exc_fraction=0.,
                            dt=1e-4, tstop=2, extended_output=False,\
                            ext_drive_change=0.):

    # find external drive
    M = get_connectivity_and_synapses_matrix(NTWK, SI_units=True)
    ext_drive = M[0,0]['ext_drive']
    if afferent_exc_fraction<0.5:
        afferent_exc_fraction = M[0,0]['afferent_exc_fraction']
    
    X0 = find_fixed_point(NRN1, NRN2, NTWK, exc_aff=ext_drive+ext_drive_change,\
                          Ne=Ne, Ni=Ni,
                          verbose=True)

    TF1, TF2 = load_transfer_functions(NRN1, NRN2, NTWK)

    t = np.arange(int(tstop/dt))*dt
    
    def dX_dt_scalar(X, t=0):
        exc_aff = ext_drive+ext_drive_change+(1-afferent_exc_fraction)*array_func(t)
        pure_exc_aff = (2*afferent_exc_fraction-1)*array_func(t) # what needs to be added
        return build_up_differential_operator(TF1, TF2, Ne=Ne, Ni=Ni, T=T)(X,\
                                      exc_aff=exc_aff, pure_exc_aff=pure_exc_aff)


    X = np.zeros((len(t), len(X0)))
    X[0,:] = X0
    for i in range(len(t)-1):
        exc_aff = ext_drive+ext_drive_change+(1-afferent_exc_fraction)*array_func(t[i])
        pure_exc_aff = (2*afferent_exc_fraction-1)*array_func(t[i]) # what needs to be added
        # X[i+1,:] = X[i,:] + dt*dX_dt_scalar(X[i,:], t=t[i])
        X[i+1,:] = X[i,:] + (t[1]-t[0])*build_up_differential_operator(TF1, TF2,\
                Ne=Ne, Ni=Ni)(X[i,:], exc_aff=exc_aff, pure_exc_aff=pure_exc_aff)
    
    fe, fi = X[:,0], X[:,1]
    sfe, sfei, sfi = [np.sqrt(X[:,i]) for i in range(2,5)]

    if extended_output:
        params = get_neuron_params(NRN2, SI_units=True)
        reformat_syn_parameters(params, M)

        exc_aff = ext_drive+ext_drive_change+(1-afferent_exc_fraction)*array_func(t)
        pure_exc_aff = (2*afferent_exc_fraction-1)*array_func(t) # what needs to be added


        # excitatory neurons have more excitation
        muV_e, sV_e, muGn_e, TvN_e = get_fluct_regime_vars(\
                                                           fe+exc_aff+pure_exc_aff,\
                                                           fi, *pseq_params(params))
        muV_i, sV_i, muGn_i, TvN_i = get_fluct_regime_vars(\
                                                           fe+exc_aff,\
                                                           fi, *pseq_params(params))
        muV, sV, muGn, TvN = .8*muV_e+.2*muV_i, .8*sV_e+.2*sV_i,\
                                        .8*muGn_e+.2*muGn_i, .8*TvN_e+.2*TvN_i,
            
        return t, fe, fi, sfe, sfei, sfi, muV, sV, muGn, TvN
    else:
        return t, fe, fi, sfe, sfei, sfi
    
if __name__=='__main__':
    
    import matplotlib.pylab as plt
    t, fe, fi, muV, sV, _, _ = run_mean_field('RS-cell', 'FS-cell', 'CONFIG1', func, T=5e-3,\
                               extended_output=True)
    # t, fe, fi, sfe, sfei, sfi, muV, sV, _, _ = run_mean_field_extended('RS-cell', 'FS-cell', 'CONFIG1', func, T=5e-3,\
    #                                                                    extended_output=True)
    plt.figure()
    plt.plot(t, fe, 'g')
    plt.plot(t, fi, 'r')
    plt.plot(t, .8*fe+.2*fi, 'k')
    plt.figure()
    plt.plot(t, 1e3*muV)
    plt.show()

