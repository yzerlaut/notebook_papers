import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
import numpy as np
import matplotlib.pylab as plt

def bin_array(array, BIN, time_array):
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[-1]-time_array[0])/BIN)
    return array[:N0*N1].reshape((N1,N0)).mean(axis=1)

def plot_ntwk_sim_output(time_array, rate_array, rate_exc, rate_inh,\
                         Raster_exc, Raster_inh,\
                         Vm_exc, Vm_inh, Ge_exc, Ge_inh, Gi_exc, Gi_inh,\
                         zoom_conditions=None, bar_ms=40,\
                         raster_number=10000, vpeak=-10, BIN=5):

    if zoom_conditions is not None:
        z = zoom_conditions
    else:
        z = [time_array[0],time_array[-1]]
    cond_t = (time_array>z[0]) & (time_array<z[1])

    Ne = Raster_inh[1].min()

    FIGS = []
    # plotting 
    FIGS.append(plt.figure(figsize=(4,2)))
    cond = (Raster_exc[0]>z[0]) & (Raster_exc[0]<z[1])& (Raster_exc[1]>Ne-raster_number)
    plt.plot(Raster_exc[0][cond], Raster_exc[1][cond], '.g', ms=2)
    cond = (Raster_inh[0]>z[0]) & (Raster_inh[0]<z[1])& (Raster_inh[1]<Ne+.2*raster_number)
    plt.plot(Raster_inh[0][cond], Raster_inh[1][cond], '.r', ms=2)

    plt.plot([z[0],z[0]+bar_ms], [Ne, Ne], 'k-', lw=5)
    plt.annotate(str(bar_ms)+'ms', (z[0]+bar_ms, Ne))
    
    set_plot(plt.gca(), ['left'], ylabel='Neuron index', xticks=[], yticks=[7700, 8000])

    FIGS.append(plt.figure(figsize=(4,3)))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))
    for i in range(len(Vm_exc)):
        ax1.plot(time_array[cond_t], Vm_exc[i][cond_t]-5.*i, 'g-', lw=1)
        spk = np.where( (Vm_exc[i][cond_t][:-1]>-50) & (Vm_exc[i][cond_t][1:]<-55))[0]
        for ispk in spk: ax1.plot(time_array[cond_t][ispk]*np.ones(2),
                                  [Vm_exc[i][cond_t][ispk]-5.*i,vpeak-5.*i], 'g--', lw=1)
        ax2.plot(time_array[cond_t], Ge_exc[i][cond_t], 'g-', lw=.5)
        ax2.plot(time_array[cond_t], Gi_exc[i][cond_t], 'r-', lw=.5)
        
    ax1.plot(time_array[cond_t][0]*np.ones(2), [-65, -55], lw=4, color='k')
    ax1.annotate('10mV', (time_array[cond_t][0], -65))
    set_plot(ax1, [], xticks=[], yticks=[])
    set_plot(ax2, ['left'], ylabel='$G$ (nS)', xticks=[], num_yticks=3)

    FIGS.append(plt.figure(figsize=(4,3)))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))
    for i in range(len(Vm_inh)):
        ax1.plot(time_array[cond_t], Vm_inh[i][cond_t]-5.*i, 'r-', lw=1)
        spk = np.where( (Vm_inh[i][cond_t][:-1]>-51) & (Vm_inh[i][cond_t][1:]<-55))[0]
        for ispk in spk: ax1.plot(time_array[cond_t][ispk]*np.ones(2), [Vm_inh[i][cond_t][ispk]-5.*i,vpeak-5.*i], 'r--', lw=1)
        ax2.plot(time_array[cond_t], Ge_inh[i][cond_t], 'g-', lw=.5)
        ax2.plot(time_array[cond_t], Gi_inh[i][cond_t], 'r-', lw=.5)
        
    ax1.plot(time_array[cond_t][0]*np.ones(2), [-65, -55], lw=4, color='k')
    ax1.annotate('10mV', (time_array[cond_t][0], -65))
    set_plot(ax1, [], xticks=[], yticks=[])
    set_plot(ax2, ['left'], ylabel='$G$ (nS)', xticks=[], num_yticks=3)

    fig, AX = plt.subplots(figsize=(4,2))
    # we bin the population rate
    rate_exc = bin_array(rate_exc[cond_t], BIN, time_array[cond_t])
    rate_inh = bin_array(rate_inh[cond_t], BIN, time_array[cond_t])
    rate_array = bin_array(rate_array[cond_t], BIN, time_array[cond_t])
    time_array = bin_array(time_array[cond_t], BIN, time_array[cond_t])
    
    AX.plot(time_array, rate_exc, 'g-', lw=1, label='$\\nu_e(t)$')
    AX.plot(time_array, rate_inh, 'r-', lw=1, label='$\\nu_i(t)$')
    AX.plot(time_array, .8*rate_exc+.2*rate_inh, 'k-', lw=1, label='$\\nu(t)$')
    AX.plot(time_array, rate_array, 'k--', lw=1, label='$\\nu_e^{aff}(t)$ \n $\\nu_e^{drive}$')
    AX.legend(prop={'size':'xx-small'})
    set_plot(AX, ['left'], ylabel='$\\nu$ (Hz)', xticks=[], num_yticks=3)
    FIGS.append(fig)

    return AX, FIGS

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

    parser.add_argument("-f", "--file",help="filename", default='data/example.npy')
    parser.add_argument("-z", "--zoom",help="zoom for activity", type=float, nargs=2)
    parser.add_argument("-b", "--bar_ms",help="bar for legend", type=int, default=100)
    parser.add_argument("-r", "--raster_number",help="max neuron number", type=int, default=10000)
    parser.add_argument("-s", "--save",action='store_true')

    args = parser.parse_args()
    
    time_array, rate_array, rate_exc, rate_inh,\
        Raster_exc, Raster_inh, Vm_exc, Vm_inh,\
        Ge_exc, Ge_inh, Gi_exc, Gi_inh = np.load(args.file)

    AX, FIG = plot_ntwk_sim_output(time_array, rate_array, rate_exc, rate_inh,\
                                   Raster_exc, Raster_inh,\
                                   Vm_exc, Vm_inh, Ge_exc, Ge_inh, Gi_exc, Gi_inh,\
                                   zoom_conditions=args.zoom, bar_ms=args.bar_ms,\
                                   raster_number=args.raster_number)









        
