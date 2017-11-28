import numpy as np
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
from PIL import Image # BITMAP (png, jpg, ...)
from matplotlib.cm import copper
import matplotlib as mpl

from data_analysis.processing.signanalysis import autocorrel
from scipy.optimize import minimize
from scipy.stats import skew

def get_acf_time(Vm, dt):
    acf, shift = autocorrel(Vm, 50, dt)
    def func(X):
        return np.sum(np.abs(X[1]*np.exp(-shift/X[0])-acf))
    res = minimize(func, [10, 1.])
    return res.x[0]

Blue, Orange, Green, Red, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

def raster_plot(data, ax=None,
                exc_pop_key='RecExc',
                inh_pop_key='RecInh',
                ratio_between_exc_and_inh_neurons=0.2,
                tzoom=[1000, 2000],
                Nmax=5000, Tbar=50, Nbar=500):

    if ax is None:
        fig, ax = plt.subplots(figsize=(2,2))
    else:
        fig = None
        
    # number of considered exc and inh neurons
    NEmax, NImax = Nmax*(1-ratio_between_exc_and_inh_neurons), Nmax*ratio_between_exc_and_inh_neurons
    ax.plot([tzoom[0], tzoom[0]], [-NImax, NEmax], 'w.') # to have the nice limit
    # raster activity
    nn = 0
    try:
        cond = (data['tRASTER_'+exc_pop_key]>tzoom[0]) & (data['tRASTER_'+exc_pop_key]<tzoom[1]) &\
               (data['iRASTER_'+exc_pop_key]<=NEmax)
        ax.plot(data['tRASTER_'+exc_pop_key][cond], data['iRASTER_'+exc_pop_key][cond], '.',
                 color=Green, ms=1)
        cond = (data['tRASTER_'+inh_pop_key]>tzoom[0]) & (data['tRASTER_'+inh_pop_key]<tzoom[1])&\
               (data['iRASTER_'+inh_pop_key]<=NImax)
        ax.plot(data['tRASTER_'+inh_pop_key][cond], -data['iRASTER_'+inh_pop_key][cond], '.',
                 color=Red, ms=1)
    except ValueError:
        pass
    ax.plot(tzoom[0]*np.ones(2), [-NImax, Nbar-NImax], lw=5, color='gray')
    ax.annotate(str(Nbar)+' neurons',\
                 (-0.1, .7), rotation=90, fontsize=14, xycoords='axes fraction')
    # plt.ylabel(str(Nnrn)+' neurons', fontsize=14)
    ax.plot([tzoom[0],tzoom[0]+Tbar], [-NImax, -NImax], lw=5, color='gray')
    ax.annotate(str(Tbar)+' ms',
                 (0.1, -0.1), fontsize=14, xycoords='axes fraction')
    # plt.xlabel(str(Tbar)+' ms            ', fontsize=14)
    set_plot(ax, [], yticks=[], xticks=[])
    return fig
    

def pop_act(SET_OF_DATA, tdiscard=200,
            exc_pop_key='RecExc', inh_pop_key='RecInh',
            bg_colors=[Blue, Orange], LABELS=['SAS', 'BS']):
    """
    analyze population activity and 
    """
    
    fig, ax = plt.subplots(1, figsize=(2.6,2))
    plt.subplots_adjust(left=.33, bottom=.2)
    plt.yscale('log')
    XTICKS, XTICKS_LABELS = [], []
    
            
    for k, data in enumerate(SET_OF_DATA):
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        cond = t>tdiscard
        for i, f, color, label in zip(range(2), [data['POP_ACT_'+exc_pop_key], data['POP_ACT_'+inh_pop_key]],
                                      [Green, Red], ['$\\nu_e$', '$\\nu_i$']):
            mean = f[cond].mean()
            std = f[cond].std()
            # ax.bar([i], [mean], yerr=[std], color=color, width=.7, bottom=0.01)
            ax.bar([i+3*k], [mean], edgecolor=color,
                      width=.6, bottom=0.007, facecolor=bg_colors[k], lw=3)
            XTICKS.append(i+3*k)
            XTICKS_LABELS.append(label)

    set_plot(ax, ['left'],
             yticks=np.array([1e-2,1e-1,1,10,100]), ylabel='rate (Hz)',
             yticks_labels=['0.01', '0.1', '1', '10', '100'],
             xticks=XTICKS, xticks_labels=XTICKS_LABELS)
    
    return fig

def Vm_signature(SET_OF_DATA, tdiscard=200, tspkdiscard=10.,
                 pop_key='RecExc', TH_DATA=None,
                 colors=[Blue, Orange], LABELS=['SAS', 'BS']):
    """
    analyze population activity and 
    """
    
    fig, AX = plt.subplots(1, 4, figsize=(10,2))
    plt.subplots_adjust(left=.33, bottom=.2, wspace=3.)
    
    for k, data in enumerate(SET_OF_DATA):

        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        dt = data['dt']
        MUV, SV, SKV, TV = [], [], [], []
        for i in range(len(data['VMS_'+pop_key])):
            cond = (t>tdiscard) & (data['VMS_'+pop_key][i]!=-70)
            # then removing spikes
            tspikes = data['tRASTER_'+str(pop_key)][\
                                    np.argwhere(data['iRASTER_'+str(pop_key)]==i).flatten()]
            for ts in tspikes:
                cond = cond & np.invert((t>=ts) & (t<=(ts+tspkdiscard)))
            MUV.append(data['VMS_'+pop_key][i][cond].mean()+70)
            SV.append(data['VMS_'+pop_key][i][cond].std())
            SKV.append(skew(data['VMS_'+pop_key][i][cond]))
            TV.append(get_acf_time(data['VMS_'+pop_key][i][cond], dt))

        AX[0].bar([k], np.mean(np.array(MUV)), yerr=np.std(np.array(MUV)), color=colors[k],
                  label='num. sim.')
        AX[1].bar([k], np.mean(np.array(SV)), yerr=np.std(np.array(SV)), color=colors[k])
        AX[2].bar([k], np.mean(np.array(SKV)), yerr=np.std(np.array(SKV)), color=colors[k])
        AX[3].bar([k], np.mean(np.array(TV)), yerr=np.std(np.array(TV)), color=colors[k])
        
        if TH_DATA is not None:
            AX[0].bar([-1.+k*3.0], 1e3*TH_DATA[k]['muV_RecExc']+70, color=colors[k], edgecolor='k', hatch="///", label='mean field')
            AX[1].bar([-1.+k*3.0], 1e3*TH_DATA[k]['sV_RecExc'], color=colors[k], edgecolor='k', hatch="///")
            AX[2].bar([-1.+k*3.0], TH_DATA[k]['gV_RecExc'], color=colors[k], edgecolor='k', hatch="///")
            AX[3].bar([-1.+k*3.0], 1e3*TH_DATA[k]['Tv_RecExc'], color=colors[k], edgecolor='k', hatch="///")
            # AX[0].legend(loc=(-6.4,0.2))
    # set_plot(AX[0], ['left'], xticks=[], ylabel='mean depol. \n  $\mu_V$ (mV)',
    set_plot(AX[0], ['left'], xticks=[], ylabel='$\mu_V$ (mV)',
             yticks=[0,5,10], yticks_labels=['-70','-65','-60'])
    set_plot(AX[1], ['left'], xticks=[], ylabel='$\sigma_V$ (mV)',
             yticks=np.arange(3)*2)
    set_plot(AX[2], ['left'], xticks=[], ylabel='$\gamma_V$')
    set_plot(AX[3], ['left'], xticks=[], ylabel='$\\tau_V$ (ms)',
             yticks=np.arange(3)*10)
    return fig

def Vm_autocorrel(SET_OF_DATA, tdiscard=200, tspkdiscard=10.,
                  pop_key='RecExc',
                  colors=[Blue, Orange], LABELS=['SAS', 'BS']):
    """
    analyze population activity and 
    """
    
    fig, ax = plt.subplots(figsize=(2,1.5))
    plt.subplots_adjust(left=.33, bottom=.2, wspace=3.)
    for k, data in enumerate(SET_OF_DATA):

        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        dt = data['dt']
        ACF = []
        for i in range(len(data['VMS_'+pop_key])):
            cond = (t>tdiscard) & (data['VMS_'+pop_key][i]!=-70)
            # then removing spikes
            tspikes = data['tRASTER_'+str(pop_key)][\
                                    np.argwhere(data['iRASTER_'+str(pop_key)]==i).flatten()]
            for ts in tspikes:
                cond = cond & np.invert((t>=ts) & (t<=(ts+tspkdiscard)))
            acf, shift = autocorrel(data['VMS_'+pop_key][i][cond], 50, dt)
            ACF.append(acf)
            
        ax.plot(shift, np.mean(np.array(ACF), axis=0), color=colors[k], lw=2)
        ax.fill_between(shift,
                        np.mean(np.array(ACF), axis=0)-np.std(np.array(ACF), axis=0),
                        np.mean(np.array(ACF), axis=0)+np.std(np.array(ACF), axis=0),
                        color=colors[k], alpha=.5)
    set_plot(ax, xticks=[0, 25, 50], yticks=[0, 1.], yticks_labels=['0', '1'],
             ylabel='n.ACF', xlabel='$\delta t$ (ms)')
    return fig

def conductances(SET_OF_DATA, tdiscard=200,
                 NVm=4, pop_key='RecExc',
                 bg_colors=[Blue, Orange], LABELS=['SAS', 'BS']):
    """
    analyze conductances
    """
    
    fig, AX = plt.subplots(1, len(SET_OF_DATA), figsize=(1.5*len(SET_OF_DATA),2))
    plt.subplots_adjust(left=.33, bottom=.2, wspace=1)
            
    for k, data in enumerate(SET_OF_DATA):
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        cond = t>tdiscard
        # excitation
        # for i, f, color, label in zip(range(2), [data['POP_ACT_'+exc_pop_key], data['POP_ACT_'+inh_pop_key]],
        #                               [Green, Red], ['$\\nu_e$', '$\\nu_i$']):
        for i, key, color, label in zip(range(NVm), ['GSYNe', 'GSYNi'],
                                        [Green, Red], ['$\\nu_e$', '$\\nu_i$']):
        
            mean = np.mean([data[key+'_'+str(pop_key)][j].mean() for j in range(NVm)])
            std = np.std([data[key+'_'+str(pop_key)][j].mean() for j in range(NVm)])
            AX[k].bar([i], [mean], yerr=[std], edgecolor=color,
                      width=.5, facecolor=bg_colors[k], lw=3)

        set_plot(AX[k], ['left'],
                 ylabel=['$G_{syn}$ (nS)','', ''][k],
                 xticks=[0,1], xticks_labels=['$G_e$','$G_i$'])
    
    return fig

def conductances_ratio(SET_OF_DATA, tdiscard=200,
                       NVm=4, pop_key='RecExc',
                       bg_colors=[Blue, Orange], LABELS=['SAS', 'BS']):
    """
    analyze conductances
    """
    
    fig, ax = plt.subplots(1, figsize=(0.7,1.2))
    plt.subplots_adjust(left=.33, bottom=.2, wspace=1)

    for k, data in enumerate(SET_OF_DATA):
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        cond = t>tdiscard
        G = []
        for i, key, color, label in zip(range(NVm), ['GSYNe', 'GSYNi'],
                                        [Green, Red], ['$\\nu_e$', '$\\nu_i$']):
        
            mean = np.mean([data[key+'_'+str(pop_key)][j].mean() for j in range(NVm)])
            G.append(mean)
        ax.bar([k], [G[1]/G[0]], color=bg_colors[k], width=.8)

    set_plot(ax, ['left'], yticks=[0,1,2],
             ylabel='$G_i$/$G_e$', xticks=[])
    
    return fig

def currents(SET_OF_DATA, tdiscard=200,
             NVm=4, pop_key='RecExc',
             bg_colors=[Blue, Orange], LABELS=['SAS', 'BS']):
    """
    analyze currents
    """
    
    fig, ax = plt.subplots(1, figsize=(3.*len(SET_OF_DATA),2))
    plt.subplots_adjust(left=.33, bottom=.2, wspace=2.)
            
    for k, data in enumerate(SET_OF_DATA):
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        cond = t>tdiscard
        for i, key in zip(range(NVm), ['ISYNe', 'ISYNi']):
            # current means
            mean = np.mean([np.abs(data[key+'_'+str(pop_key)][j]).mean() for j in range(NVm)])
            std = np.std([np.abs(data[key+'_'+str(pop_key)][j]).mean() for j in range(NVm)])
            ax.bar([i+5*k], [mean], yerr=[std],
                      width=.8, color=bg_colors[k], bottom=10)
            # current fluctuations
            mean = np.mean([np.abs(data[key+'_'+str(pop_key)][j]).std() for j in range(NVm)])
            std = np.std([np.abs(data[key+'_'+str(pop_key)][j]).std() for j in range(NVm)])
            ax.bar([i+2+5*k], [mean], yerr=[std],
                   width=.8, color=bg_colors[k], bottom=10)

    ax.set_yscale('log')
            
    set_plot(ax, ['left'],
             ylabel='$\| I_{syn} \|$ (pA)', yticks=[10, 100, 1000],
             xticks=np.arange(9), xticks_labels=['$\mu_{e}$', '$\mu_{i}$',
                                                 '$\sigma_{e}$', '$\sigma_{i}$',
                                                 '',
                                                 '$\mu_{e}$', '$\mu_{i}$',
                                                 '$\sigma_{e}$', '$\sigma_{i}$'])
    
    return fig

def IE_ratio(SET_OF_DATA, tdiscard=200,
             NVm=4, pop_key='RecExc',
             bg_colors=[Blue, Orange], LABELS=['SAS', 'BS']):
    """
    analyze current ratio
    """
    
    fig, AX = plt.subplots(1, figsize=(1.5,2))
    plt.subplots_adjust(left=.33, bottom=.2, wspace=1)
            
    for k, data in enumerate(SET_OF_DATA):
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        cond = t>tdiscard
        RATIOS = []
        for i in zip(range(NVm)):
            IEmean = np.mean([data['ISYNe_'+str(pop_key)][j].mean() for j in range(NVm)])
            IImean = np.mean([data['ISYNi_'+str(pop_key)][j].mean() for j in range(NVm)])
            RATIOS.append(np.abs(IImean/IEmean))

        AX.bar([k], np.array(RATIOS).mean(), yerr=[np.array(RATIOS).std()],
                      width=.7, facecolor=bg_colors[k], lw=3)

    set_plot(AX, ['left'],
             ylabel='|| $I_{i}$ / $I_{e}$ ||', yticks=[0,0.5,1],
             xticks=[0,1], xticks_labels=['SA','BA'])
    
    return fig


def few_Vm_fig(SET_OF_DATA, pop_key='Exc',
               tspace=200.,
               tzoom=[200, np.inf],
               VMS1=np.arange(3), VMS2=None,
               Tbar=100, Vbar=10,
               Vshift=30, vpeak=-42, vbottom=-80):

    fig, ax = plt.subplots(figsize=(5*len(SET_OF_DATA),5))
    # plt.subplots_adjust(left=.15, bottom=.1, right=.99)

    if VMS2 is None:
        VMS2 = VMS1 # ID of cells

    t0 = 0
    for k, data in enumerate(SET_OF_DATA):
        VMS = [VMS1, VMS2][k]
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        cond = (t>tzoom[0]) & (t<tzoom[1])
        for i, j in enumerate(VMS):
            ax.plot(t[cond]+t0, data['VMS_'+str(pop_key)][j][cond]+Vshift*i,\
                    color=copper(i/len(VMS)/1.5), lw=1)
            ax.plot(t[cond]+t0, 0*t[cond]+-70+Vshift*i, ':', color=copper(i/len(VMS)/1.5), lw=1)
            ax.plot([t[cond][0]+t0], [-70+Vshift*i], '>', color=copper(i/len(VMS)/1.5))
        # adding spikes
        for i, j in enumerate(VMS):
            tspikes = data['tRASTER_'+str(pop_key)][np.argwhere(data['iRASTER_'+str(pop_key)]==j).flatten()]
            cond = (tspikes>tzoom[0]) & (tspikes<tzoom[1])
            for ts in tspikes[cond]:
                ax.plot([ts+t0, ts+t0],
                        Vshift*i+np.array([-50, vpeak]), '--', color=copper(i/len(VMS)/1.5), lw=1)
        t0=tspace+tzoom[1]-tzoom[0]

    ax.plot([tzoom[0],tzoom[0]+Tbar], ax.get_ylim()[0]*np.ones(2),
                 lw=5, color='gray')
    ax.annotate(str(Tbar)+'ms', (tzoom[0], .9*ax.get_ylim()[0]), fontsize=14)
    ax.plot(tzoom[1]*np.ones(2), [-60,-60+Vbar], lw=5, color='gray')
    ax.annotate(str(Vbar)+'mV', (tzoom[1], -60), fontsize=14)
    ax.annotate('-70mV', (tzoom[1], -70), fontsize=14)
    set_plot(ax, [], xticks=[], yticks=[])

    c = plt.axes([.5, .5, .01, .2])
    cmap = mpl.colors.ListedColormap(copper(np.linspace(0,1/1.5,len(VMS1))))
    cb = mpl.colorbar.ColorbarBase(c, cmap=cmap,
                                   orientation='vertical')
    cb.set_label('Cell ID', fontsize=12)
    cb.set_ticks([])
    return fig

def four_Vm_traces(data, ax,
                   tzoom=[1000, 2000],
                   Tbar = 50,
                   Vshift=30, vpeak=-42, vbottom=-80):

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])
    ## excitatory traces:
    for i in range(3):
        ax.plot(t[cond], data['VMS_RecExc'][i][cond]+Vshift*i, color=copper(i/3/1.5), lw=1)
    # adding spikes
    for i in range(3):
        tspikes = data['tRASTER_RecExc'][np.argwhere(data['iRASTER_RecExc']==i).flatten()]
        for ts in tspikes[(tspikes>tzoom[0]) & (tspikes<tzoom[1])]:
            ax.plot([ts, ts], Vshift*i+np.array([-50, vpeak]), '--',
                    color=copper(i/3/1.5), lw=1)
    # ## inhibitory trace:
    cond = (t>tzoom[0]) & (t<tzoom[1])
    ax.plot(t[cond], data['VMS_RecInh'][0][cond]+Vshift*3, color=Red, lw=1)
    tspikes = data['tRASTER_RecInh'][np.argwhere(data['iRASTER_RecInh']==0).flatten()]
    for ts in tspikes[(tspikes>tzoom[0]) & (tspikes<tzoom[1])]:
        ax.plot([ts, ts], Vshift*3+np.array([-53, vpeak]), '--', color=Red, lw=1)

    ax.plot(tzoom[0]*np.ones(2), [-50, -40], lw=5, color='gray')
    ax.annotate('10mV',(-0.1, .4), rotation=90, fontsize=14, xycoords='axes fraction')
    ax.plot([tzoom[0],tzoom[0]+Tbar], [-75, -75], lw=5, color='gray')
    ax.annotate(str(Tbar)+' ms',
                 (0.1, -0.1), fontsize=14, xycoords='axes fraction')
    set_plot(ax, [], xticks=[], yticks=[])
    

def Isyn_fig(SET_OF_DATA, pop_key='RecExc',
             tspace=200.,
             tzoom=[200, 1000],
             VMS1=1, VMS2=None,
             Tbar=100, Ibar=1000, trace_factor=5.,
             Vshift=30, vpeak=-42, vbottom=-80):

    fig, ax = plt.subplots(figsize=(5*len(SET_OF_DATA),2.5))
    # plt.subplots_adjust(left=.15, bottom=.1, right=.99)

    if VMS2 is None:
        VMS2 = VMS1 # ID of cells

    t0 = 0
    for k, data in enumerate(SET_OF_DATA):
        VMS = [VMS1, VMS2][k]
        FACTOR = [trace_factor, 1][k]
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        cond = (t>tzoom[0]) & (t<tzoom[1])
        ax.plot(t[cond]+t0, FACTOR*data['ISYNe_'+str(pop_key)][VMS][cond], color=Green, lw=1)
        ax.plot(t[cond]+t0, FACTOR*data['ISYNi_'+str(pop_key)][VMS][cond], color=Red, lw=1)
        ax.plot(t[cond]+t0, 0*t[cond], 'k:', lw=1)
        t0=tspace+tzoom[1]-tzoom[0]

    ax.plot([tzoom[0],tzoom[0]+Tbar], [0, 0], lw=5, color='gray')
    ax.annotate(str(Tbar)+'ms', (tzoom[0], .9*ax.get_ylim()[0]), fontsize=14)
    # first bar
    ax.plot(tzoom[0]*np.ones(2), [0,Ibar], lw=5, color='gray')
    ax.annotate(str(round(Ibar/trace_factor))+'pA', (tzoom[0], 0), fontsize=14)
    # second bar
    ax.plot(tzoom[1]*np.ones(2)+tspace, [0,Ibar], lw=5, color='gray')
    ax.annotate(str(round(Ibar/1000))+'nA', (tzoom[1]+tspace, 0), fontsize=14)
    set_plot(ax, [], xticks=[], yticks=[])

    return fig


def one_Isyn_sample(data, ax,
                    pop_key='RecExc',
                    tzoom=[1000, 2000],
                    Tbar = 50):

    t = np.arange(int(data['tstop']/data['dt']))*data['dt']
    cond = (t>tzoom[0]) & (t<tzoom[1])
    ax.plot(t[cond], data['ISYNe_'+str(pop_key)][0][cond], color=Green, lw=1)
    ax.plot(t[cond], data['ISYNi_'+str(pop_key)][0][cond], color=Red, lw=1)
    ax.plot(t[cond], 0.*t[cond], 'k--')

    y1, y2 = ax.get_ylim()
    Ibar = int((y2-y1)/5/100)*100
    ax.plot(tzoom[0]*np.ones(2), [0, Ibar], lw=5, color='gray')
    ax.annotate(str(Ibar)+'pA',(-0.1, .7), rotation=90, fontsize=14, xycoords='axes fraction')
    ax.plot([tzoom[0],tzoom[0]+Tbar], [y1, y1], lw=5, color='gray')
    ax.annotate(str(Tbar)+' ms', (0.1, -0.1), fontsize=14, xycoords='axes fraction')
    set_plot(ax, [], xticks=[], yticks=[])

def Vm_histograms(SET_OF_DATA, pop_key='RecExc',
                  colors=[Orange, Blue],
                  tdiscard=200., tspkdiscard=10.):
    
    fig, ax = plt.subplots(figsize=(2,1.5))
    # plt.subplots_adjust(left=.15, bottom=.1, right=.99)

    for k, data in enumerate(SET_OF_DATA):
        Vm = np.empty(0)
        for i in range(len(data['VMS_'+pop_key])):
            t = np.arange(int(data['tstop']/data['dt']))*data['dt']
            cond = (t>tdiscard)
            # then removing spikes
            tspikes = data['tRASTER_'+str(pop_key)][\
                                    np.argwhere(data['iRASTER_'+str(pop_key)]==i).flatten()]
            for ts in tspikes:
                cond = cond & np.invert((t>=ts) & (t<=(ts+tspkdiscard)))
            Vm = np.concatenate([Vm, data['VMS_'+pop_key][i][cond]])
        hist, be = np.histogram(Vm, bins=20, normed=True)
        ax.bar(.5*(be[1:]+be[:-1]), hist, color=colors[k], alpha=.8, width=be[1]-be[0])
    set_plot(ax, yticks=[], xlabel='$V_m$ (mV)', ylabel='density',
             xticks=[-70, -60, -50])
    return fig
    
def plot_act_vs_aff_level(npy_file,
                          with_axes=False,
                          min_level=1.5, XTICKS=[2,5,10,20],\
                          Iratio_th=None, Fe_th=None, Fi_th=None, Fa_th=None):
    
    FA, SYNCH, _, BALANCE, EXC_ACT, INH_ACT, _, _, _ = np.load(npy_file)

    Fa, Fe, Fi, sFe, sFi, Balance, sBalance  = [], [], [], [], [], [], []
    Synch, sSynch = [], []
    for fa in np.unique(np.unique(FA)):
        i0 = np.argwhere(np.array(FA)==fa)
        Fe.append(EXC_ACT[i0].mean())
        sFe.append(EXC_ACT[i0].std())
        Fi.append(INH_ACT[i0].mean())
        sFi.append(INH_ACT[i0].std())
        Balance.append(BALANCE[i0].mean())
        sBalance.append(BALANCE[i0].std())
        Synch.append(np.abs(SYNCH[i0]).mean())
        sSynch.append(np.abs(SYNCH[i0]).std())
        Fa.append(fa)
    Fa, Fe, Fi, sFe, sFi, Balance, sBalance = np.array(Fa), np.array(Fe), np.array(Fi), np.array(sFe),\
                        np.array(sFi), np.array(Balance), np.array(sBalance)
    

    Fe[Fe<=0.01], sFe[Fe<=0.01] = 0.011, 0
    Fi[Fi<=0.01], sFi[Fi<=0.01] = 0.011, 0
    
    # plotting activity
    fig1, ax = plt.subplots(1, figsize=(2.3,2))
    ax.set_yscale('log'); ax.set_xscale('log')
    
    if Fe_th is not None:
        ax.plot(Fa_th, Fe_th, color=Green, lw=7, alpha=.5, label='num. sim.')
    if Fi_th is not None:
        ax.plot(Fa_th, Fi_th, color=Red, lw=7, alpha=.5, label='mean field')
    ax.errorbar(Fa, Fe, yerr=sFe, color=Green, lw=4, label='$\\nu_e$')
    ax.errorbar(Fa, Fi, yerr=sFi, color=Red, lw=4, label='$\\nu_i$')
    ax.legend(frameon=False)
    set_plot(ax,
             ylabel='rate (Hz)', xlabel='$\\nu_a$ (Hz)',
             xticks=XTICKS,
             xticks_labels=[str(x) for x in XTICKS],
             yticks=[0.01,0.1,1,10], yticks_labels=['<0.01','0.1','1','10'])
    y1, y2 = ax.get_ylim()
    # i0 = np.argmax(Fe==0.01)
    # ax.fill_between([min_level, Fa[i0]], [y1,y1], [y2, y2], color=Grey, alpha=.5)

    # plotting balance
    fig2, ax2 = plt.subplots(1, figsize=(1.8,2))
    if Iratio_th is not None:
        ax2.plot(Fa_th, np.abs(Iratio_th), color='k', lw=7, alpha=.5, label='mean field')
    ax2.errorbar(Fa, Balance, sBalance, lw=3, color='k', label='num. sim.')
    ax2.legend(frameon=False)
    ax2.set_xscale('log')
    set_plot(ax2,
             ylabel=r'$\| I_i $ / $ I_e \|$', xlabel='$\\nu_a$ (Hz)',
             xticks=XTICKS,
             xticks_labels=[str(x) for x in XTICKS],
             yticks=[0.,0.5,1])

    for a in [ax, ax2]: a.set_xlim([Fa.min(), Fa.max()])
    
    if with_axes:
        return ax, ax2, fig1, fig2
    else:
        return fig1, fig2


if __name__=='__main__':
    from data_analysis.IO.hdf5 import load_dict_from_hdf5
    from graphs.my_graph import *

    # plot_act_vs_aff_level('data/varying_AffExc_analyzed.npy')
    # sas_data = load_dict_from_hdf5('data/sas.h5')
    # bs_data = load_dict_from_hdf5('data/bs.h5')
    # BA_theory = dict(np.load('data/BA_theory.npz').items())
    # SA_theory = dict(np.load('data/SA_theory.npz').items())
    # fig = Vm_autocorrel([sas_data, bs_data]);

    FAFF2, FE, FI, Iratio = np.load('data/varying_AffExc_theory.npy')
    FE[FE<0.01], FI[FI<0.01] = 0.011, 0.011 # same thing than for numerical sim.
    fig1, fig2 = plot_act_vs_aff_level('data/varying_AffExc_analyzed.npy',
                                          Iratio_th=Iratio, Fe_th=FE, Fi_th=FI, Fa_th=FAFF2)
    
    # fig.savefig('temp.svg')
    show()
    
