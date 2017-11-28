from __future__ import print_function

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from brian2 import *
import numpy as np

from network_simulations.time_varying_input import *
from single_cell_models.cell_library import get_neuron_params
from single_cell_models.cell_construct import get_membrane_equation
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from synapses_and_connectivity.syn_and_connec_construct import build_up_recurrent_connections_for_2_pop,\
    build_up_recurrent_connections, build_up_poisson_group_to_pop

def run_simulation(NRN_exc='LIF', NRN_inh='LIF', NTWK='Vogels-Abbott', DT=0.1, tstop=300,\
                   kick_value=50., kick_duration=30., SEED=1, ext_drive=0., input_rate=None,\
                   afferent_exc_fraction=0.,
                   n_rec=3, full_recording=False, filename='data/example_data.npy'):

    seed(SEED%100)
    
    M = get_connectivity_and_synapses_matrix(NTWK, number=2)
    if afferent_exc_fraction<.5:
        afferent_exc_fraction = M[0,0]['afferent_exc_fraction']
        
    # number of neurons
    Ne, Ni= int(M[0,0]['Ntot']*(1-M[0,0]['gei'])), int(M[0,0]['Ntot']*M[0,0]['gei'])
    
    exc_neurons, eqs = get_membrane_equation(get_neuron_params(NRN_exc, number=Ne), M[:,0], return_equations=True)
    inh_neurons, eqs = get_membrane_equation(get_neuron_params(NRN_inh, number=Ni), M[:,1], return_equations=True)

    ## INITIAL CONDITIONS
    exc_neurons.Gee, exc_neurons.Gie, exc_neurons.V = '0.*nS', '0.*nS', '-65*mV'
    inh_neurons.Gei, inh_neurons.Gii, inh_neurons.V = '0.*nS', '0.*nS', '-65*mV'
    
    ## FEEDFORWARD EXCITSTORY CONNECTIONS
    time_array = np.arange(int(tstop/DT))*DT
    rate_array = np.array([kick_value*tt/kick_duration+(tt/kick_duration-1)*ext_drive\
                           if tt<kick_duration else 0. for tt in time_array])+ext_drive

    # input_on_inh, input_on_exc = rate_array, rate_array
    # ### PURE EXC CASE, DELETED !!
    if input_rate is not None:
        input_on_exc, input_on_inh = rate_array+afferent_exc_fraction*input_rate,\
                                     rate_array+(1-afferent_exc_fraction)*input_rate
    else:
        input_on_exc, input_on_inh = rate_array, rate_array

    ## FEEDFORWARD EXCITATION
    input_exc, fdfrwd_to_exc, input_inh, fdfrwd_to_inh = \
        build_up_excitatory_feedforward_connections_for_2_pop(\
                            [exc_neurons, inh_neurons], M,
                            time_array, input_on_exc, input_on_inh,\
                            SEED=(SEED+1)**2)

    ## RECURRENT CONNECTIONS
    exc_exc, exc_inh, inh_exc, inh_inh = \
      build_up_recurrent_connections_for_2_pop([exc_neurons, inh_neurons], M,\
                                               SEED=(SEED+2)**2) # only for 2 pop !

    # setting up the recording
    PRe = PopulationRateMonitor(exc_neurons)
    PRi = PopulationRateMonitor(inh_neurons)
    if full_recording:
        trace_Vm_exc = StateMonitor(exc_neurons, 'V', record=range(n_rec))
        trace_Vm_inh = StateMonitor(inh_neurons, 'V', record=range(n_rec))
        trace_Ge_exc = StateMonitor(exc_neurons, 'Gee', record=range(n_rec))
        trace_Gi_exc = StateMonitor(exc_neurons, 'Gie', record=range(n_rec))
        trace_Ge_inh = StateMonitor(inh_neurons, 'Gei', record=range(n_rec))
        trace_Gi_inh = StateMonitor(inh_neurons, 'Gii', record=range(n_rec))
        raster_exc = SpikeMonitor(exc_neurons)
        raster_inh = SpikeMonitor(inh_neurons)

    # running the simulation
    defaultclock.dt = DT*ms
    run(tstop*ms)
    
    if full_recording:
        Raster_exc, Raster_inh, Vm_exc, Vm_inh, Ge_exc, Ge_inh, Gi_exc, Gi_inh =\
           transform_to_simple_arrays(trace_Vm_exc, trace_Vm_inh, trace_Ge_exc, trace_Gi_exc,\
                                      trace_Ge_inh, trace_Gi_inh, raster_exc, raster_inh,\
                                      M, n_rec=n_rec)
        np.save(filename,
                [time_array, rate_array, PRe.rate/Hz, PRi.rate/Hz, Raster_exc,\
                 Raster_inh, Vm_exc, Vm_inh, Ge_exc, Ge_inh, Gi_exc, Gi_inh])
        return time_array, rate_array, PRe.rate/Hz, PRi.rate/Hz
    else:
        np.save(filename, [time_array, rate_array, PRe.rate/Hz, PRi.rate/Hz])
        return time_array, rate_array, PRe.rate/Hz, PRi.rate/Hz

def transform_to_simple_arrays(trace_Vm_exc, trace_Vm_inh, trace_Ge_exc, trace_Gi_exc,\
                     trace_Ge_inh, trace_Gi_inh, raster_exc, raster_inh, M, n_rec=3):

    Ne= int(M[0,0]['Ntot']*(1-M[0,0]['gei']))
    
    Raster_exc = [raster_exc.t/ms, raster_exc.i]
    Raster_inh = [raster_inh.t/ms, raster_inh.i+Ne]
    
    # now traces
    Vm_exc, Vm_inh = [], []
    Ge_exc, Ge_inh = [], []
    Gi_exc, Gi_inh = [], []

    for i in range(n_rec):
        Vm_exc.append(array(trace_Vm_exc[i].V/mV))
        Vm_inh.append(array(trace_Vm_inh[i].V/mV))
        Ge_exc.append(array(trace_Ge_exc[i].Gee/nS))
        Gi_exc.append(array(trace_Gi_exc[i].Gie/nS))
        Ge_inh.append(array(trace_Ge_inh[i].Gei/nS))
        Gi_inh.append(array(trace_Gi_inh[i].Gii/nS))

    return np.array(Raster_exc, dtype=float),\
        np.array(Raster_inh, dtype=float),\
        np.array(Vm_exc, dtype=float),\
        np.array(Vm_inh, dtype=float),\
        np.array(Ge_exc, dtype=float),\
        np.array(Ge_inh, dtype=float),\
        np.array(Gi_exc, dtype=float),\
        np.array(Gi_inh, dtype=float)

if __name__=='__main__':


    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     ----------------------------------------------------------------------
     Run the a network simulation using brian2

     Choose CELLULAR and NTWK PARAMETERS from the available libraries
     see  ../synapses_and_connectivity.syn_and_connec_library.py for the CELLS
     see ../synapses_and_connectivity.syn_and_connec_library.py for the NTWK

     Then construct the input as "NRN_exc--NRN_inh--NTWK"
     example: "LIF--LIF--Vogels-Abbott"
     ----------------------------------------------------------------------
     """
    ,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--CONFIG",help="Cell and Network configuration !", default='RS-cell--FS-cell--CONFIG1')
    parser.add_argument("--DT",help="time steps in ms", type=float, default=0.1)
    parser.add_argument("--tstop",help="time of simulation in ms", type=float, default=1500)
    parser.add_argument("--kick_value",help=" stimulation (Hz) for the initial kick", type=float, default=0.)
    parser.add_argument("--kick_duration",help=" stimulation duration (ms) for the initial kick", type=float, default=100.)
    parser.add_argument("--ext_drive",help=" stimulation duration (ms) for the initial kick", type=float, default=4.)
    parser.add_argument("--SEED",help="SEED for the simulation", type=int, default=5)
    parser.add_argument("-f", "--file",help="filename for saving", default='data/example.npy')
    parser.add_argument("--n_rec",help="number of recorded neurons", type=int, default=3)

    args = parser.parse_args()
    
    run_simulation(\
                   NRN_exc=args.CONFIG.split('--')[0],\
                   NRN_inh=args.CONFIG.split('--')[1],\
                   NTWK=args.CONFIG.split('--')[2],
                   kick_value=args.kick_value, kick_duration=args.kick_duration,
                   DT=args.DT, tstop=args.tstop, SEED=args.SEED, ext_drive=args.ext_drive,\
                   full_recording=True, n_rec=args.n_rec, filename=args.file)

    
