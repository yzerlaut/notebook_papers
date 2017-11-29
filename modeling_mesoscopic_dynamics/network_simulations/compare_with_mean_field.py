import numpy as np
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from mean_field.master_equation import find_fixed_point
from transfer_functions.theoretical_tools import mean_and_var_conductance, get_fluct_regime_vars, pseq_params
from scipy.signal import gaussian
from single_cell_models.cell_library import get_neuron_params
from synapses_and_connectivity.syn_and_connec_library import get_connectivity_and_synapses_matrix
from transfer_functions.tf_simulation import reformat_syn_parameters
from network_simulations.waveform_input import gaussian_func, double_gaussian
from network_simulations.plot_single_sim import bin_array
from mean_field.euler_method import run_mean_field_extended
from transfer_functions.theoretical_tools import get_fluct_regime_vars, pseq_params

def skew(x):
    return np.abs(np.mean((x.mean()-x)**3))**(1./3.)#/x.std()**3)

def plot_ntwk_sim_output(time_array, rate_array, rate_exc, rate_inh,\
                         Raster_exc, Raster_inh,\
                         Vm_exc, Vm_inh, Ge_exc, Ge_inh, Gi_exc, Gi_inh,\
                         BIN=5, min_time=200):
    
    
    cond_t = (time_array>min_time) # transient behavior after 400 ms

    params = get_neuron_params('RS-cell', SI_units=True)
    M = get_connectivity_and_synapses_matrix('CONFIG1', SI_units=True)
    EXC_AFF = M[0,0]['ext_drive']
    
    print('starting fixed point')
    fe0, fi0, sfe, sfie, sfi = find_fixed_point('RS-cell', 'FS-cell', 'CONFIG1',\
                                                exc_aff=EXC_AFF, Ne=8000, Ni=2000, verbose=True)
    print('end fixed point')
    
    reformat_syn_parameters(params, M) # merging those parameters
    
    xfe = fe0+np.linspace(-4,4)*sfe
    fe_pred = gaussian_func(xfe, fe0, sfe)
    xfi = fi0+np.linspace(-4,4)*sfi
    fi_pred = gaussian_func(xfi, fi0, sfi)

    mGe, mGi, sGe, sGi = mean_and_var_conductance(fe0+EXC_AFF, fi0, *pseq_params(params))
    muV, sV, muGn, TvN = get_fluct_regime_vars(fe0+EXC_AFF, fi0, *pseq_params(params))

    FE, FI = np.meshgrid(xfe, xfi)
    pFE, pFI = np.meshgrid(fe_pred, fi_pred)
    MUV, SV, _, _ = get_fluct_regime_vars(FE+EXC_AFF, FI, *pseq_params(params))*pFE*pFI/np.sum(pFE*pFI)
    
    ### MEMBRANE POTENTIAL
    MEAN_VM, STD_VM, KYRT_VM = [], [], []
    for i in range(len(Vm_exc)):
        MEAN_VM.append(Vm_exc[i][(time_array>min_time) & (Vm_exc[i]!=-65) & (Vm_exc[i]<-50)].mean())
        MEAN_VM.append(Vm_inh[i][(time_array>min_time) & (Vm_inh[i]!=-65) & (Vm_inh[i]<-50)].mean())
        for vv in [Vm_exc[i][(time_array>min_time)], Vm_inh[i][(time_array>min_time)]]:
            i0 = np.where((vv[:-1]>-52) & (vv[1:]<-60))[0]
            sv = []
            if len(i0)==0:
                STD_VM.append(vv.std())
            elif len(i0)==1:
                STD_VM.append(vv[0].std())
            else:
                for i1, i2 in zip(i0[:-1], i0[1:]):
                    if i2-i1>60:
                        sv.append(vv[i1+30:i2-30].std())
                STD_VM.append(np.array(sv).mean())
        STD_VM.append(Vm_inh[i][(time_array>min_time) & (Vm_inh[i]<-50)].std())

    fig1, AX1 = plt.subplots(1, 3, figsize=(5,2)) # for means
    plt.subplots_adjust(wspace=1.)
    fig2, AX2 = plt.subplots(1, 3, figsize=(5,2)) # for std
    plt.subplots_adjust(wspace=1.)
    
    AX1[0].bar([0], np.array(MEAN_VM).mean()+65, yerr=np.array(MEAN_VM).std(), color='w', edgecolor='k', lw=3, error_kw=dict(elinewidth=3,ecolor='k'))
    AX2[0].bar([0], np.array(STD_VM).mean(), yerr=np.array(STD_VM).std(), color='w', edgecolor='k', lw=3, error_kw=dict(elinewidth=3,ecolor='k'), label='$V_m$')
    AX1[0].bar([1], [1e3*muV+65], color='gray', alpha=.5, label='$V_m$')
    AX2[0].bar([1], [1e3*sV], color='gray', alpha=.5)

    set_plot(AX1[0], ['left'], xticks=[], ylim=[0,11], yticks=[0, 5, 10], yticks_labels=['-65', '-60', '-55'], ylabel='mean (mV)')
    set_plot(AX2[0], ['left'], xticks=[], ylim=[0,5], yticks=[0, 2, 4], ylabel='std. dev. (mV)')


    
    ### EXCITATORY CONDUCTANCE
    MEAN_GE, STD_GE, KYRT_GE = [], [], []
    for i in range(len(Ge_exc)):
        MEAN_GE.append(Ge_exc[i][(time_array>min_time)].mean())
        MEAN_GE.append(Ge_inh[i][(time_array>min_time)].mean())
        STD_GE.append(Ge_exc[i][(time_array>min_time)].std())
        STD_GE.append(Ge_inh[i][(time_array>min_time)].std())

    AX1[1].bar([0], np.array(MEAN_GE).mean(), yerr=np.array(MEAN_GE).std(), color='w', edgecolor='g', lw=3, error_kw=dict(elinewidth=3,ecolor='g'), label='num. sim.')
    AX2[1].bar([0], np.array(STD_GE).mean(), yerr=np.array(STD_GE).std(), color='w', edgecolor='g', lw=3, error_kw=dict(elinewidth=3,ecolor='g'), label='exc.')
    AX1[1].bar([1], [1e9*mGe], color='g', label='mean field \n pred.')
    AX2[1].bar([1], [1e9*sGe], color='g', label='exc.')
    
    set_plot(AX1[1], ['left'], xticks=[], yticks=[0,15,30], ylabel='mean (nS)')
    set_plot(AX2[1], ['left'], xticks=[], yticks=[0,5,10], ylabel='std. dev. (nS)')
    
    ### INHIBITORY CONDUCTANCE
    MEAN_GI, STD_GI, KYRT_GI = [], [], []
    for i in range(len(Gi_exc)):
        MEAN_GI.append(Gi_exc[i][(time_array>min_time)].mean())
        MEAN_GI.append(Gi_inh[i][(time_array>min_time)].mean())
        STD_GI.append(Gi_exc[i][(time_array>min_time)].std())
        STD_GI.append(Gi_inh[i][(time_array>min_time)].std())

    AX1[2].bar([0], np.array(MEAN_GI).mean(), yerr=np.array(MEAN_GI).std(), color='w', edgecolor='r', lw=3, error_kw=dict(elinewidth=3,ecolor='r'), label='num. sim.')
    AX2[2].bar([0], np.array(STD_GI).mean(), yerr=np.array(STD_GI).std(), color='w', edgecolor='r', lw=3, error_kw=dict(elinewidth=3,ecolor='r'), label='inh.')
    AX1[2].bar([1], [1e9*mGi], color='r', label='mean field \n pred.')
    AX2[2].bar([1], [1e9*sGi], color='r', label='inh.')
    
    set_plot(AX1[2], ['left'], xticks=[], yticks=[0,15,30], ylabel='mean (nS)')
    set_plot(AX2[2], ['left'], xticks=[], yticks=[0,5,10], ylabel='std. dev. (nS)')

    ### POPULATION RATE ###
    
    fig, ax = plt.subplots(figsize=(5,3))
    plt.subplots_adjust(bottom=.3)
    # we bin the population rate
    N0 = int(BIN/(time_array[1]-time_array[0]))
    N1 = int((time_array[cond_t][-1]-time_array[cond_t][0])/BIN)
    time_array = time_array[cond_t][:N0*N1].reshape((N1,N0)).mean(axis=1)
    rate_exc = rate_exc[cond_t][:N0*N1].reshape((N1,N0)).mean(axis=1)
    rate_inh = rate_inh[cond_t][:N0*N1].reshape((N1,N0)).mean(axis=1)
    rate_array = rate_array[cond_t][:N0*N1].reshape((N1,N0)).mean(axis=1)

    hh, bb = np.histogram(rate_exc, bins=8, normed=True)
    ax.bar(.5*(bb[:-1]+bb[1:]), hh, color='w', width=bb[1]-bb[0], edgecolor='g', lw=3, label='exc.', alpha=.7)
    hh, bb = np.histogram(rate_inh, bins=8, normed=True)
    ax.bar(.5*(bb[:-1]+bb[1:]), hh, color='w', width=bb[1]-bb[0], edgecolor='r', lw=3, label='inh', alpha=.7)

    ax.fill_between(xfe, 0*fe_pred, fe_pred, color='g')
    ax.fill_between(xfi, 0*fi_pred, fi_pred, color='r')

    set_plot(plt.gca(), ['bottom', 'left'], xlabel='pop. activity (Hz)', yticks=[], ylabel='density')

    # for ax in AX1: ax.legend()
    # for ax in AX2: ax.legend()
    
    return [fig, fig1, fig2]

def plot_ntwk_sim_output_for_waveform(args,\
                                      BIN=5, min_time=200, bar_ms=50,
                                      zoom_conditions=None, vpeak=-35,\
                                      raster_number=400):
    
    time_array, rate_array, rate_exc, rate_inh,\
        Raster_exc, Raster_inh,\
        Vm_exc, Vm_inh, Ge_exc, Ge_inh, Gi_exc, Gi_inh = np.load(args.file)

    if zoom_conditions is not None:
        z = zoom_conditions
    else:
        z = [time_array[0],time_array[-1]]
    cond_t = (time_array>z[0]) & (time_array<z[1])

    ###### =========================================== ######
    ############# adding the theoretical eval ###############
    ###### =========================================== ######
    
    t0 = args.t0-4*args.T1
    def rate_func(t):
        return double_gaussian(t, 1e-3*args.t0, 1e-3*args.T1, 1e-3*args.T2, args.amp)

    t, fe, fi, sfe, sfei, sfi = run_mean_field_extended(args.CONFIG.split('--')[0],\
                               args.CONFIG.split('--')[1],args.CONFIG.split('--')[2],\
                               rate_func,dt=5e-4,
                               tstop=args.tstop*1e-3)

    params = get_neuron_params('RS-cell', SI_units=True)
    M = get_connectivity_and_synapses_matrix('CONFIG1', SI_units=True)
    reformat_syn_parameters(params, M) # merging those parameters
    ext_drive = M[0,0]['ext_drive']
    afferent_exc_fraction = M[0,0]['afferent_exc_fraction']

    fe_e, fe_i = fe+ext_drive+rate_func(t), fe+ext_drive
    muV_e, sV_e, muGn_e, TvN_e = get_fluct_regime_vars(fe_e, fi, *pseq_params(params))
    muV_i, sV_i, muGn_i, TvN_i = get_fluct_regime_vars(fe_i, fi, *pseq_params(params))
    
    Ne = Raster_inh[1].min()

    FIGS = []
    # plotting 
    FIGS.append(plt.figure(figsize=(5,4)))
    cond = (Raster_exc[0]>z[0]) & (Raster_exc[0]<z[1])& (Raster_exc[1]>Ne-raster_number)
    plt.plot(Raster_exc[0][cond], Raster_exc[1][cond], '.g')
    cond = (Raster_inh[0]>z[0]) & (Raster_inh[0]<z[1])& (Raster_inh[1]<Ne+.2*raster_number)
    plt.plot(Raster_inh[0][cond], Raster_inh[1][cond], '.r')

    plt.plot([z[0],z[0]+bar_ms], [Ne, Ne], 'k-', lw=5)
    plt.annotate(str(bar_ms)+'ms', (z[0]+bar_ms, Ne))
    
    set_plot(plt.gca(), ['left'], ylabel='Neuron index', xticks=[])

    FIGS.append(plt.figure(figsize=(5,5)))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))
    for i in range(len(Vm_exc)):
        ax1.plot(time_array[cond_t], Vm_exc[i][cond_t], 'g-', lw=.5)
        spk = np.where( (Vm_exc[i][cond_t][:-1]>-50) & (Vm_exc[i][cond_t][1:]<-55))[0]
        for ispk in spk: ax1.plot(time_array[cond_t][ispk]*np.ones(2), [Vm_exc[i][cond_t][ispk],vpeak], 'g--')
        ax2.plot(time_array[cond_t], Ge_exc[i][cond_t], 'g-', lw=.5)
        ax2.plot(time_array[cond_t], Gi_exc[i][cond_t], 'r-', lw=.5)
    # vm
    ax1.plot(1e3*t[1e3*t>t0], 1e3*muV_e[1e3*t>t0], 'g-', lw=2)
    ax1.fill_between(1e3*t[1e3*t>t0], 1e3*(muV_e[1e3*t>t0]-sV_e[1e3*t>t0]),\
                     1e3*(muV_e[1e3*t>t0]+sV_e[1e3*t>t0]), color='g', alpha=.4)
    # ge
    ge_th = 1e9*fe_e[1e3*t>t0]*params['Qe']*params['Te']*(1-params['gei'])*params['pconnec']*params['Ntot']
    sge_th = 1e9*params['Qe']*np.sqrt(fe_e[1e3*t>t0]*params['Te']*(1-params['gei'])*params['pconnec']*params['Ntot'])
    ax2.plot(1e3*t[1e3*t>t0], ge_th, 'g-', lw=2)
    ax2.fill_between(1e3*t[1e3*t>t0], ge_th-sge_th, ge_th+sge_th, color='g', alpha=.4)
    gi_th = 1e9*fi[1e3*t>t0]*params['Qi']*params['Ti']*params['gei']*params['pconnec']*params['Ntot']
    sgi_th = 1e9*params['Qi']*np.sqrt(fi[1e3*t>t0]*params['Ti']*params['gei']*params['pconnec']*params['Ntot'])
    ax2.plot(1e3*t[1e3*t>t0], gi_th, 'r-', lw=2)
    ax2.fill_between(1e3*t[1e3*t>t0], gi_th-sgi_th, gi_th+sgi_th, color='r', alpha=.4)

    
    ax1.plot(time_array[cond_t][0]*np.ones(2), [-65, -55], lw=4, color='k')
    ax1.annotate('10mV', (time_array[cond_t][0], -65))
    set_plot(ax1, [], xticks=[], yticks=[])
    set_plot(ax2, ['left'], ylabel='$G$ (nS)', xticks=[])

    FIGS.append(plt.figure(figsize=(5,5)))
    ax1 = plt.subplot2grid((3,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((3,1), (2,0))
    for i in range(len(Vm_inh)):
        ax1.plot(time_array[cond_t], Vm_inh[i][cond_t], 'r-', lw=.5)
        spk = np.where( (Vm_inh[i][cond_t][:-1]>-51) & (Vm_inh[i][cond_t][1:]<-55))[0]
        for ispk in spk: ax1.plot(time_array[cond_t][ispk]*np.ones(2), [Vm_inh[i][cond_t][ispk],vpeak], 'r--')
        ax2.plot(time_array[cond_t], Ge_inh[i][cond_t], 'g-', lw=.5)
        ax2.plot(time_array[cond_t], Gi_inh[i][cond_t], 'r-', lw=.5)
        
    # vm
    ax1.plot(1e3*t[1e3*t>t0], 1e3*muV_i[1e3*t>t0], 'r-', lw=2)
    ax1.fill_between(1e3*t[1e3*t>t0], 1e3*(muV_i[1e3*t>t0]-sV_i[1e3*t>t0]),\
                     1e3*(muV_i[1e3*t>t0]+sV_i[1e3*t>t0]), color='r', alpha=.4)
    ge_th = 1e9*fe_i[1e3*t>t0]*params['Qe']*params['Te']*(1-params['gei'])*params['pconnec']*params['Ntot']
    sge_th = 1e9*params['Qe']*np.sqrt(fe_i[1e3*t>t0]*params['Te']*(1-params['gei'])*params['pconnec']*params['Ntot'])
    ax2.plot(1e3*t[1e3*t>t0], ge_th, 'g-', lw=2)
    ax2.fill_between(1e3*t[1e3*t>t0], ge_th-sge_th, ge_th+sge_th, color='g', alpha=.4)
    ax2.plot(1e3*t[1e3*t>t0], gi_th, 'r-', lw=2)
    ax2.fill_between(1e3*t[1e3*t>t0], gi_th-sgi_th, gi_th+sgi_th, color='r', alpha=.4)
    
    ax1.plot(time_array[cond_t][0]*np.ones(2), [-65, -55], lw=4, color='k')
    ax1.annotate('10mV', (time_array[cond_t][0], -65))
    set_plot(ax1, [], xticks=[], yticks=[])
    set_plot(ax2, ['left'], ylabel='$G$ (nS)', xticks=[])

    fig, AX = plt.subplots(figsize=(5,3))
    # we bin the population rate
    rate_exc = bin_array(rate_exc[cond_t], BIN, time_array[cond_t])
    rate_inh = bin_array(rate_inh[cond_t], BIN, time_array[cond_t])
    rate_array = bin_array(rate_array[cond_t], BIN, time_array[cond_t])
    time_array = bin_array(time_array[cond_t], BIN, time_array[cond_t])
    
    AX.plot(time_array, rate_exc, 'g-', lw=2, label='$\\nu_e(t)$')
    AX.plot(time_array, rate_inh, 'r-', lw=2, label='$\\nu_i(t)$')
    AX.plot(time_array, .8*rate_exc+.2*rate_inh, 'k-', lw=2, label='$\\nu(t)$')

    
    AX.plot(1e3*t[1e3*t>t0], rate_func(t[1e3*t>t0]), 'k:',\
            lw=2, label='$\\nu_e^{aff}(t)$ \n $\\nu_e^{drive}$')
    AX.plot(1e3*t[1e3*t>t0], fe[1e3*t>t0], 'g-', label='mean field \n pred.')
    AX.fill_between(1e3*t[1e3*t>t0], fe[1e3*t>t0]-sfe[1e3*t>t0], fe[1e3*t>t0]+sfe[1e3*t>t0],\
                        color='g', alpha=.3, label='mean field \n pred.')
    AX.plot(1e3*t[1e3*t>t0], fi[1e3*t>t0], 'r-', label='num. sim.')
    AX.fill_between(1e3*t[1e3*t>t0], fi[1e3*t>t0]-sfi[1e3*t>t0], fi[1e3*t>t0]+sfi[1e3*t>t0],\
                        color='r', alpha=.3)
    # AX.plot(1e3*t[1e3*t>t0], .8*fe[1e3*t>t0]+.2*fi[1e3*t>t0], 'k-', label='..')

    # AX.legend(prop={'size':'xx-small'})
    
    set_plot(AX, ['left'], ylabel='$\\nu$ (Hz)', xticks=[], num_yticks=3)
    FIGS.append(fig)

    return FIGS
    

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

    parser.add_argument("-f", "--file",help="filename for saving", default='data/example.npy')
    parser.add_argument("-z", "--zoom",help="zoom for activity", type=float, nargs=2)
    parser.add_argument("-b", "--bar_ms",help="bar for legend", type=int, default=100)
    parser.add_argument("-r", "--raster_number",help="max neuron number", type=int, default=10000)
    parser.add_argument("-t_after_transient", type=float, default=500)
    parser.add_argument("-s", "--save",action='store_true')

    args = parser.parse_args()
    
    time_array, rate_array, rate_exc, rate_inh,\
        Raster_exc, Raster_inh, Vm_exc, Vm_inh,\
        Ge_exc, Ge_inh, Gi_exc, Gi_inh = np.load(args.file)

    FIG = plot_ntwk_sim_output(time_array, rate_array, rate_exc, rate_inh,\
                               Raster_exc, Raster_inh,\
                               Vm_exc, Vm_inh, Ge_exc, Ge_inh, Gi_exc, Gi_inh,\
                               min_time=args.t_after_transient)

    plt.show()







        
