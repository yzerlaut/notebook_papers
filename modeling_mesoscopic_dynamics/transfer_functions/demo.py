import numpy as np
import matplotlib.pyplot as plt
import tf_simulation as sims # where we store the single cells simulation
import sys
sys.path.append('../code/')
from my_graph import set_plot
sys.path.append('../')
from single_cell_models.cell_library import get_neuron_params
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix

## model parameters

def demo_plot(P, dt=0.1e-3, tstop=1000e-3, I0=150e-12, fe=10., fi=10., vpeak=-30):

    t = np.arange(int(tstop/dt))*dt

    fig = plt.figure(figsize=(10,10))

    ax11 = plt.subplot2grid((6,2), (0, 0), rowspan=3)
    ax21 = plt.subplot2grid((6,2), (3,0))
    ax31 = plt.subplot2grid((6,2), (4,0))
    ax41 = plt.subplot2grid((6,2), (5,0))

    ax12 = plt.subplot2grid((6,2), (0, 1), rowspan=3)
    ax22 = plt.subplot2grid((6,2), (3,1))
    ax32 = plt.subplot2grid((6,2), (4,1))
    ax42 = plt.subplot2grid((6,2), (5,1))

    I = np.array([I0 if ( (tt>200e-3) and (tt<900e-3)) else 0 for tt in t])
    v, spikes = sims.adexp_sim(t, I, 0.*I, 0.*I, *sims.pseq_adexp(P))

    ax11.plot(1e3*t, 1e3*v, 'k-')
    for s in spikes: ax11.plot([1e3*s,1e3*s], [1e3*P['Vthre'],vpeak], 'k:')
    ax21.plot(1e3*t, 1e12*I, 'g-')
    ax31.plot(1e3*t, 0*I, 'r-')
    ax41.plot(1e3*t, 0*I, 'b-')


    fe, fi = 10, 10
    Ge = sims.generate_conductance_shotnoise(fe, t, P['Ntot']*(1-P['gei'])*P['pconnec'], P['Qe'], P['Te'], g0=0, seed=0)
    Gi = sims.generate_conductance_shotnoise(fi, t, P['Ntot']*P['gei']*P['pconnec'], P['Qi'], P['Ti'], g0=0, seed=1)

    v, spikes = sims.adexp_sim(t, 0.*I, Ge, Gi, *sims.pseq_adexp(P))

    ax12.plot(1e3*t, 1e3*v, 'k-')
    for s in spikes: ax12.plot([1e3*s,1e3*s], [1e3*P['Vthre'],vpeak], 'k:')
    ax22.plot(1e3*t, 0*I, 'g-')
    ax32.plot(1e3*t, 1e9*Ge, 'r-')
    ax42.plot(1e3*t, 1e9*Gi, 'b-')

    set_plot(ax12, ylabel='V (mV)', xticks_labels=[])
    set_plot(ax22, ylabel='I (pA)', xticks_labels=[])
    set_plot(ax32, ylabel='Ge (nS)', xticks_labels=[])
    set_plot(ax42, ylabel='Gi (nS)', xlabel='time (ms)')
    set_plot(ax11, ylabel='V (mV)', xticks_labels=[])
    set_plot(ax21, ylabel='I (pA)', xticks_labels=[])
    set_plot(ax31, ylabel='Ge (nS)', xticks_labels=[])
    set_plot(ax41, ylabel='Gi (nS)', xlabel='time (ms)')
    
    return fig


if __name__=='__main__':

    import argparse
    
    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ demo plot """,
              formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("Neuron_Model",help="Choose a neuronal model from 'neuronal_models.py'")
    parser.add_argument("Network_Model",help="Choose a network model (synaptic and connectivity properties)"+\
                        "\n      from 'network_models'.py")

    args = parser.parse_args()
    
    params = get_neuron_params(args.Neuron_Model, SI_units=True)
    M = get_connectivity_and_synapses_matrix(args.Network_Model, SI_units=True)

    sims.reformat_syn_parameters(params, M) # merging those parameters

    demo_plot(params)
    
    plt.show()

