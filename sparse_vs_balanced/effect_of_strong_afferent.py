import numpy as np
from itertools import product
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk
from sparse_vs_balanced.running_2pop_model import run_2pop_ntwk_model
from sparse_vs_balanced.running_3pop_model import run_3pop_ntwk_model
from data_analysis.IO.hdf5 import load_dict_from_hdf5
from graphs.my_graph import *
from matplotlib import ticker
# everything stored within a zip file
from scipy.special import erf
from data_analysis.processing.signanalysis import gaussian_smoothing
from graphs.plot_export import put_list_of_figs_to_svg_fig
from matplotlib.cm import copper
import zipfile

Blue, Orange, Green, Red, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

def waveform(t, Model):
    return Model['Faff0']+Model['Faff1']*(1+erf((t-Model['T1'])/Model['DT1']))*\
        (1+erf(-(t-Model['T2'])/Model['DT2']))/4

def get_scan(Model, with_png_export=False,
             filename=None):

    if filename is None:
        filename=str(Model['zip_filename'])
    zf = zipfile.ZipFile(filename, mode='r')
    
    data = zf.read(filename.replace('.zip', '_Model.npz'))
    with open(filename.replace('.zip', '_Model.npz'), 'wb') as f: f.write(data)
    Model = dict(np.load(filename.replace('.zip', '_Model.npz')).items())
    
    seeds = Model['SEEDS']
    
    DATA = []
    for j in range(len(seeds)):
        fn = Model['FILENAMES'][j]
        data = zf.read(fn)
        with open(fn, 'wb') as f: f.write(data)
        with open(fn, 'rb') as f: data = load_dict_from_hdf5(fn)
        DATA.append(data)
    return Model, seeds, DATA

def one_Vm_fig(data, Model,
               iNVm_Exc=np.arange(3), iNVm_Inh=0, iNVm_DsInh=0,
               dV=30, vpeak=-43, smoothing=10e-3, Gl=10,
               FIGSIZE=(6,4),
               XTICKS=[0, 500, 1000],
               t0=600., tstart=1100., tdur=300, tspkdiscard=10.):
    
    tstop = Model['T2']+2.*Model['DT2']
    t = np.arange(int(tstop/Model['dt']))*Model['dt']
    
    fig1, ax = plt.subplots(1, figsize=FIGSIZE)
    plt.subplots_adjust(top=.99, bottom=.2, left=.01, right=.99)
    # exc
    for i, k in enumerate(iNVm_Exc):
        cond = (t>t0)
        ax.plot(t[cond]-Model['T1']+Model['DT1'], data['VMS_RecExc'][k][cond]+i*dV, '-', lw=1, color=Green)
        tspikes = data['tRASTER_RecExc'][np.argwhere(data['iRASTER_RecExc']==k).flatten()]
        for ts in tspikes[tspikes>t0]:
            ax.plot([ts-Model['T1']+Model['DT1'], ts-Model['T1']+Model['DT1']],
                    [data['RecExc_Vthre']+i*dV, vpeak+i*dV], '--', lw=1, color=Green)
    # inh
    ax.plot(t[cond]-Model['T1']+Model['DT1'], data['VMS_RecInh'][iNVm_Inh][cond]-dV, '-', lw=1, color=Red)
    tspikes = data['tRASTER_RecInh'][np.argwhere(data['iRASTER_RecInh']==iNVm_Inh).flatten()]
    for ts in tspikes[tspikes>t0]:
        ax.plot([ts-Model['T1']+Model['DT1'], ts-Model['T1']+Model['DT1']], [data['RecInh_Vthre']-dV, vpeak-dV], '--', color=Red, lw=1)
    # dsinh
    ax.plot(t[cond]-Model['T1']+Model['DT1'], data['VMS_DsInh'][iNVm_DsInh][cond]-2*dV, '-', lw=1, color=Purple)
    tspikes = data['tRASTER_DsInh'][np.argwhere(data['iRASTER_DsInh']==iNVm_DsInh).flatten()]
    for ts in tspikes[tspikes>t0]:
        ax.plot([ts-Model['T1']+Model['DT1'], ts-Model['T1']+Model['DT1']],
                [data['DsInh_Vthre']-2*dV, vpeak-2*dV], '--', color=Purple, lw=1)
    ax.plot([0,0], [-70, -60], color='gray', lw=5)
    ax.annotate('10 mV', (0, -55))
    set_plot(ax, ['bottom'], xticks=XTICKS, yticks=[], xlabel='time (ms)')
    
    return fig1

def conductance_time_course(DATA, Model, t0=400., smoothing=10,
                            FIGSIZE=(4,2),
                            YTICKS=[0, 4, 8], XTICKS=[-400, 0, 400, 800]):
    
    fig2, ax = plt.subplots(1, figsize=FIGSIZE)

    tstop = Model['T2']+2.*Model['DT2']
    t = np.arange(int(tstop/Model['dt']))*Model['dt']
    ismooth = int(smoothing/Model['dt'])
    MEAN_GE, STD_GE = np.zeros((len(DATA), len(t))), np.zeros((len(DATA), len(t)))
    MEAN_GI, STD_GI = np.zeros((len(DATA), len(t))), np.zeros((len(DATA), len(t)))
    
    for n in range(len(DATA)):

        data = DATA[n].copy()
        # smoothing conductances
        for i, gsyn in enumerate(data['GSYNe_RecExc']):
            data['GSYNe_RecExc'][i] = gaussian_smoothing(gsyn, ismooth)
        for i, gsyn in enumerate(data['GSYNi_RecExc']):
            data['GSYNi_RecExc'][i] = gaussian_smoothing(gsyn, ismooth)
            
        MEAN_GE[n, :] = np.array(data['GSYNe_RecExc']).mean(axis=0)/Model['RecExc_Gl']
        STD_GE[n, :] = np.array(data['GSYNe_RecExc']).std(axis=0)/Model['RecExc_Gl']
        MEAN_GI[n, :] = np.array(data['GSYNi_RecExc']).mean(axis=0)/Model['RecExc_Gl']
        STD_GI[n, :] = np.array(data['GSYNi_RecExc']).std(axis=0)/Model['RecExc_Gl']
        
    cond = (t>t0)
    t -= (Model['T1']-Model['DT1'])
    ax.plot(t[cond], MEAN_GE.mean(axis=0)[cond], color=Green, lw=2, label=r'$G_{e}$')
    ax.plot(t[cond], MEAN_GI.mean(axis=0)[cond], color=Red, lw=2, label=r'$G_{i}$')
    ax.fill_between(t[cond],
                    MEAN_GE.mean(axis=0)[cond]-STD_GE.mean(axis=0)[cond],
                    MEAN_GE.mean(axis=0)[cond]+STD_GE.mean(axis=0)[cond],
                    alpha=.6, color=Green)
    ax.fill_between(t[cond],
                    MEAN_GI.mean(axis=0)[cond]-STD_GI.mean(axis=0)[cond],
                    MEAN_GI.mean(axis=0)[cond]+STD_GI.mean(axis=0)[cond],
                    alpha=.6, color=Red)
    ax.legend(frameon=False)
    set_plot(ax, yticks=YTICKS, xticks=XTICKS,
             ylabel='$G_{syn}$/$g_L$', xlabel='time (ms)')
    return fig2, ax

def pop_act_time_course(DATA, Model, t0=400., smoothing=10,
                            FIGSIZE=(4,2),
                            YTICKS=[0, 15, 30], XTICKS=[0, 500, 1000]):
    
    tstop = Model['T2']+2.*Model['DT2']
    t = np.arange(int(tstop/Model['dt']))*Model['dt']
    ismooth = int(smoothing/Model['dt'])
    MEAN_FE, MEAN_FI, MEAN_FD = [np.zeros((len(DATA), len(t))) for i in range(3)]
    
    for n, data in enumerate(DATA):
        
        MEAN_FE[n, :] = gaussian_smoothing(data['POP_ACT_RecExc'], ismooth)
        MEAN_FI[n, :] = gaussian_smoothing(data['POP_ACT_RecInh'], ismooth)
        MEAN_FD[n, :] = gaussian_smoothing(data['POP_ACT_DsInh'], ismooth)

    faff = waveform(t, Model)
    cond = (t>t0)
    t -= (Model['T1']-Model['DT1'])
    
    fig3, ax = plt.subplots(1, figsize=FIGSIZE)

    for vec, label, color in zip([MEAN_FE, MEAN_FI, MEAN_FD],
                                 ['$\\nu_e$', '$\\nu_i$', '$\\nu_d$'],
                                 [Green, Red, Purple]):
        plt.plot(t[cond], vec.mean(axis=0)[cond], color=color, label=label, lw=3)
        plt.fill_between(t[cond],
                         vec.mean(axis=0)[cond]-vec.std(axis=0)[cond],
                         vec.mean(axis=0)[cond]+vec.std(axis=0)[cond],
                         color=color, alpha=.5)

    ax.legend(frameon=False)
    set_plot(ax, yticks=YTICKS, xticks=XTICKS,
             ylabel='rate (Hz)', xlabel='time (ms)')
    return fig3

def aff_pop_time_course(Model, t0=400.,
                        FIGSIZE=(4,2),
                        YTICKS=[0, 5, 10], XTICKS=[0, 500, 1000]):
    
    tstop = Model['T2']+2.*Model['DT2']
    t = np.arange(int(tstop/Model['dt']))*Model['dt']
    faff = waveform(t, Model)
    cond = (t>t0)
    t -= (Model['T1']-Model['DT1'])
    
    fig3, ax = plt.subplots(1, figsize=FIGSIZE)
    plt.plot(t[cond], faff[cond], lw=4, color=Grey, label='$\\nu_a$')
    set_plot(ax, yticks=YTICKS, xticks=XTICKS,
             ylabel='$\\nu_a$ (Hz)', xlabel='time (ms)')
    return fig3

def hist_of_Vm_pre_post(DATA, Model,
                        tbefore=-100, tstart=100, tdur=400, tspkdiscard=20.,
                        FIGSIZE=(2,2), nbin=30,
                        YTICKS=[0, 5, 10], XTICKS=[0, 500, 1000]):
    
    fig4, ax = plt.subplots(1, figsize=FIGSIZE)
    tstop = Model['T2']+2.*Model['DT2']
    t = np.arange(int(tstop/Model['dt']))*Model['dt']
    t2 = t-(Model['T1']-Model['DT1'])

    plt.subplots_adjust(bottom=.4)
    ## Vm during stimulus
    Vm = np.empty(0)
    for data in DATA:
        for key in ['RecExc', 'RecInh']:
            for i in range(len(data['VMS_'+key])):
                cond = (t2>tstart) & (t2<tstart+tdur)
                # then removing spikes
                tspikes = data['tRASTER_'+key][np.argwhere(data['iRASTER_'+key]==i).flatten()]
                for ts in tspikes:
                    cond = cond & np.invert((t>=ts-2.*Model['dt']) & (t<=(ts+tspkdiscard)))
                Vm = np.concatenate([Vm, data['VMS_'+key][i][cond]])
    hist, be = np.histogram(Vm, bins=nbin, normed=True)
    # ax.bar(.5*(be[1:]+be[:-1]), hist, color=Orange, alpha=.8, width=be[1]-be[0],
    ax.bar(be[1:], hist, color=Orange, alpha=.8, width=be[1]-be[0],
           label='t$\in$['+str(tstart)+','+str(tstart+tdur)+']ms')
    std_stim = np.std(Vm)
    
    ## Vm during spontaneous activity
    Vm = np.empty(0)
    for data in DATA:
        for key in ['RecExc', 'RecInh']:
            for i in range(len(data['VMS_'+key])):
                cond = (t2<tbefore)
                # then removing spikes
                tspikes = data['tRASTER_'+key][np.argwhere(data['iRASTER_'+key]==i).flatten()]
                for ts in tspikes:
                    cond = cond & np.invert((t>=ts) & (t<=(ts+tspkdiscard)))
                Vm = np.concatenate([Vm, data['VMS_'+key][i][cond]])
    hist, be = np.histogram(Vm, bins=nbin, normed=True)
    std_pre = np.std(Vm)
    
    # ax.bar(.5*(be[1:]+be[:-1]), hist, color=Blue, alpha=.9, width=be[1]-be[0], label='t<-100ms')
    ax.bar(be[1:], hist, color=Blue, alpha=.9, width=be[1]-be[0], label='t<-100ms')
    ax.legend(loc=(1., .2), frameon=False)
    set_plot(ax, ['bottom'], xlabel='$V_m$ (mV)', xticks=[-70, -60, -50], yticks=[])
    return fig4, [std_stim, std_pre]

def run_sim(Model, tstop=2500., SEED=4):

    NTWK = run_3pop_ntwk_model(Model,
                               faff_waveform_func=waveform,
                               with_Vm=10,
                               filename=Model['filename'], tstop=tstop, SEED=SEED)
    return NTWK

def run_scan(Model):
    
    zf = zipfile.ZipFile(Model['zip_filename'], mode='w')

    seeds = Model['SEEDS']
    Model['FILENAMES'] = np.empty(len(seeds), dtype=object)
    
    for j in range(len(seeds)):
        fn = Model['data_folder']+str(seeds[j])+\
             '_'+str(np.random.randint(100000))+'.h5'
        Model['FILENAMES'][j] = fn
        print('running configuration ', fn)
        run_3pop_ntwk_model(Model,
                            faff_waveform_func=waveform,
                            with_Vm=5,
                            filename=fn, tstop=Model['T2']+2.*Model['DT2'],
                            SEED=seeds[j])
        zf.write(fn)
        
    # writing the parameters
    np.savez(Model['zip_filename'].replace('.zip', '_Model.npz'), **Model)
    zf.write(Model['zip_filename'].replace('.zip', '_Model.npz'))

    zf.close()

if __name__=='__main__':
    
    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import *

    # parameters of the input
    
    parser.add_argument('--Faff0', type=float, default=5.)    
    parser.add_argument('--Faff1', type=float, default=12.)    
    parser.add_argument('--T1', type=float, default=1000)    
    parser.add_argument('--DT1', type=float, default=100)    
    parser.add_argument('--T2', type=float, default=1700)    
    parser.add_argument('--DT2', type=float, default=300)
    
    parser.add_argument('--SEEDS', nargs='+', help='various seeds', type=int,
                        default=np.arange(4))    
    # additional stuff
    parser.add_argument('-df', '--data_folder', help='Folder for data', default='data/')    
    parser.add_argument("--zip_filename", '-f', help="filename for the zip file",type=str,
                        default='data/time_varying_input.zip')
    parser.add_argument("-a", "--analyze", help="perform analysis of params space",
                        action="store_true")
    parser.add_argument("-rm", "--run_multiple_seeds", help="run with multiple seeds",
                        action="store_true")
    parser.add_argument("--debug", help="debug", action="store_true")
    
    args = parser.parse_args()
    Model = vars(args)

    if args.analyze:
        analyze_sim(Model)
        ntwk.show()
    elif args.run_multiple_seeds:
        Model['filename'] = 'sparse_vs_balanced/data/time_varying_input.h5'
        run_scan(Model)
    elif args.debug:
        Model, seeds, DATA = get_scan({}, filename='sparse_vs_balanced/data/time_varying_input.zip')
        hist_of_Vm_pre_post(DATA, Model, nbin=30)
        # one_Vm_fig(DATA[0], Model)
        ntwk.show()
    else:
        Model['filename'] = 'data/time_varying_input_debug.h5'
        NTWK = run_sim(Model)
        Nue, Nui, Nud = NTWK['POP_ACT'][0].rate/ntwk.Hz, NTWK['POP_ACT'][1].rate/ntwk.Hz,\
                        NTWK['POP_ACT'][2].rate/ntwk.Hz
        ntwk.plot(NTWK['POP_ACT'][0].t/ntwk.ms, gaussian_smoothing(Nue,int(20./0.1)))
        ntwk.plot(NTWK['POP_ACT'][0].t/ntwk.ms, gaussian_smoothing(Nui,int(20./0.1)))
        ntwk.plot(NTWK['POP_ACT'][0].t/ntwk.ms, gaussian_smoothing(Nud,int(20./0.1)))
        ntwk.show()

