from brian2 import *


def set_spikes_from_time_varying_rate(time_array, rate_array, N, Nsyn, SEED=1):
    
    seed(SEED) # setting the seed !
    
    ## time_array in ms !!
    # so multplying rate array
    
    indices, times = [], []
    DT = (time_array[1]-time_array[0])
    
    # trivial way to generate inhomogeneous poisson events
    for it in range(len(time_array)):
        rdm_num = np.random.random(N)
        for ii in np.arange(N)[rdm_num<DT*Nsyn*rate_array[it]*1e-3]:
            indices.append(ii) # all the indicces
            times.append(time_array[it]) # all the same time !

    return array(indices), array(times)*ms


def build_up_excitatory_feedforward_connections_for_2_pop(Pops, syn_conn_matrix,\
                                                          time_array,\
                                                          input_on_exc, input_on_inh,\
                                                          SEED=1):
    exc_neurons, inh_neurons = Pops
    P = syn_conn_matrix
    
    # number of synapses per neuron
    Nsyn = P[0,0]['p_conn']*(1-P[0,0]['gei'])*P[0,0]['Ntot']

    # feedforward input on INH pop
    indices, times = set_spikes_from_time_varying_rate(time_array, input_on_inh,\
                                                       inh_neurons.N, Nsyn,\
                                                       SEED=(SEED+2)**3%100)
    input_inh = SpikeGeneratorGroup(inh_neurons.N, indices, times)
    fdfrwd_to_inh = Synapses(input_inh, inh_neurons, on_pre='Gei_post += w',\
                             model='w:siemens')
    fdfrwd_to_inh.connect('i==j')
    fdfrwd_to_inh.w=P[0,1]['Q']*nS
    
    # feedforward input on EXC pop
    indices2, times2 = set_spikes_from_time_varying_rate(time_array, input_on_exc,\
                                                       exc_neurons.N, Nsyn,\
                                                       SEED=(SEED+1)**2%100)
    input_exc = SpikeGeneratorGroup(exc_neurons.N, indices2, times2)
    fdfrwd_to_exc = Synapses(input_exc, exc_neurons, on_pre='Gee_post += w',\
                             model='w:siemens')
    fdfrwd_to_exc.connect('i==j')
    fdfrwd_to_exc.w=P[0,0]['Q']*nS

    return input_exc, fdfrwd_to_exc, input_inh, fdfrwd_to_inh

def build_up_excitatory_feedforward_connections_for_exc_only(exc_neurons, syn_conn_matrix,\
                                                             time_array, rate_array, SEED=1):

    P = syn_conn_matrix
    # number of synapses per neuron
    Nsyn = P[0,0]['p_conn']*(1-P[0,0]['gei'])*P[0,0]['Ntot']
    
    # feedforward input on EXC pop
    indices, times = set_spikes_from_time_varying_rate(time_array, rate_array,\
                                                       exc_neurons.N, Nsyn,\
                                                       SEED=(SEED+4)**2%100)
    input_exc_aff = SpikeGeneratorGroup(exc_neurons.N, indices, times)
    fdfrwd_to_exc_aff = Synapses(input_exc_aff, exc_neurons, on_pre='Gee_post += w',\
                             model='w:siemens')
    fdfrwd_to_exc_aff.connect('i==j')
    fdfrwd_to_exc_aff.w=P[0,0]['Q']*nS

    return input_exc_aff, fdfrwd_to_exc_aff


# def build_up_inhibitory_feedforward_connections_for_2_pop(Pops, syn_conn_matrix,\
#                                                           time_array, rate_array):

#     exc_neurons, inh_neurons = Pops
#     P = syn_conn_matrix

#     inh_input_exc, inh_fdfrwd_to_exc = set_stim_from_time_varying_rate(time_array, rate_array, exc_neurons,\
#                                                     ON_PRE='Gie_post += w')
#     inh_fdfrwd_to_exc.w = P[1,0]['Q']*nS

#     inh_input_inh, inh_fdfrwd_to_inh = set_stim_from_time_varying_rate(time_array, rate_array, inh_neurons,\
#                                                     ON_PRE='Gii_post += w')
#     inh_fdfrwd_to_inh.w = P[1,1]['Q']*nS
    
#     return inh_input_exc, inh_fdfrwd_to_exc, inh_input_inh, inh_fdfrwd_to_inh
