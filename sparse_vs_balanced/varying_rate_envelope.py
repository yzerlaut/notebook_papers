import numpy as np
from itertools import product
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from data_analysis.IO.hdf5 import load_dict_from_hdf5
from graphs.my_graph import *
# everything stored within a zip file
import zipfile
import matplotlib.cm as cm
from graphs.plot_export import put_list_of_figs_to_svg_fig
from sparse_vs_balanced.running_3pop_model import run_3pop_ntwk_model
from scipy.special import erf
import neural_network_dynamics.main as ntwk
from graphs.my_graph import *
from data_analysis.processing.signanalysis import gaussian_smoothing

Blue, Orange, Green, Red, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

def waveform(t, Model):
    return Model['Faff0']+Model['Faff1']*(1+erf((t-Model['T1'])/Model['DT1']))*\
        (1+erf(-(t-Model['T2'])/Model['DT2']))/4

def get_scan(Model, filename=None):

    if filename is None:
        filename=str(Model['filename'])
        
    zf = zipfile.ZipFile(filename, mode='r')
    DATA = []
    
    model_file = filename.replace('.zip', '_Model.npz')
    data = zf.read(model_file)
    with open(model_file, 'wb') as f: f.write(data)
    Model = dict(np.load(model_file).items())
    
    F_aff = np.linspace(Model['faff0'], Model['faff1'], Model['ngrid'])
    seeds = Model['SEEDS']

    for i, j in product(range(len(F_aff)), range(len(seeds))):
        fn = Model['FILENAMES'][i,j]
        data = zf.read(fn)
        with open(fn, 'wb') as f: f.write(data)
        with open(fn, 'rb') as f: data = load_dict_from_hdf5(fn)
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        data['t'] = t
        Model['Faff0'] = Model['faff_bsl']
        Model['Faff1'] = F_aff[i]
        data['faff'] = waveform(t, Model)
        DATA.append(data)

    return F_aff, seeds, Model, DATA


def analyze_scan(Model, smooth=10, filename=None):

    F_aff, seeds, Model, DATA = get_scan(Model, filename=filename)

    fig1, ax1 = plt.subplots(1, figsize=(4,2.5))
    plt.subplots_adjust(left=.3, bottom=.4)
    fig2, ax2 = plt.subplots(1, figsize=(4,2.5))
    plt.subplots_adjust(left=.3, bottom=.4)

    for i in range(len(F_aff)):
        TRACES = []
        for j in range(len(seeds)):
            data = DATA[i*len(seeds)+j]
            # TRACES.append(data['POP_ACT_RecExc'])
            TRACES.append(gaussian_smoothing(data['POP_ACT_RecExc'],int(smooth/data['dt'])))
        mean = np.array(TRACES).mean(axis=0)
        std = np.array(TRACES).std(axis=0)
        # smoothing
        cond = data['t']>350
        ax1.plot(data['t'][cond], mean[cond], color=cm.copper(i/len(F_aff)), lw=2)
        ax1.fill_between(data['t'][cond], mean[cond]-std[cond], mean[cond]+std[cond],
                         color=cm.copper(i/len(F_aff)), alpha=.5)
        ax2.plot(data['t'][cond], data['faff'][cond], color=cm.copper(i/len(F_aff)))
    set_plot(ax1, xlabel='time (ms)', ylabel='rate (Hz)')
    set_plot(ax2, xlabel='time (ms)', ylabel='rate (Hz)')
    # put_list_of_figs_to_svg_fig([fig1, fig2])
    return fig1, fig2

def make_single_resp_fig(F_aff, seeds, Model, DATA_SA, DATA_BA,
                         smoothing = 30, t0 = 430, XTICKS=np.arange(4)*250,
                         tstart_vis = 350, tend_vis=700, t_after_stim_for_bsl=1400.):
    
    fig1, ax1 = plt.subplots(1, figsize=(4,2.5))
    plt.subplots_adjust(left=.3, bottom=.4)
    fig2, ax2 = plt.subplots(1, figsize=(4,2.5))
    plt.subplots_adjust(left=.3, bottom=.4)
    fig3, ax3 = plt.subplots(1, figsize=(4,2.5))
    plt.subplots_adjust(left=.3, bottom=.4)

    for i in range(len(F_aff)):
        SA_TRACES, BA_TRACES = [], []
        for j in range(len(seeds)):
            data = DATA_SA[i*len(seeds)+j]
            v = gaussian_smoothing(data['POP_ACT_RecExc'],int(smoothing/Model['dt']))
            SA_TRACES.append(v)
            data = DATA_BA[i*len(seeds)+j]
            v = gaussian_smoothing(data['POP_ACT_RecExc'],int(smoothing/Model['dt']))
            BA_TRACES.append(v)
        SA_mean, SA_std = np.array(SA_TRACES).mean(axis=0), np.array(SA_TRACES).std(axis=0)
        BA_mean, BA_std = np.array(BA_TRACES).mean(axis=0), np.array(BA_TRACES).std(axis=0)
        data = DATA_SA[i*len(seeds)+j]
        cond, cond2 = (data['t']>tstart_vis) & (data['t']<tend_vis) ,data['t']>t_after_stim_for_bsl
        ax2.plot(data['t'][cond]-t0,
                 SA_mean[cond]-SA_mean[cond2].mean(), color=cm.viridis(i/len(F_aff)), lw=2)
        ax2.fill_between(data['t'][cond]-t0,
                         SA_mean[cond]-SA_std[cond]-SA_mean[cond2].mean(),
                         SA_mean[cond]+SA_std[cond]-SA_mean[cond2].mean(),
                         color=cm.viridis(i/len(F_aff)), alpha=.5)
        ax3.plot(data['t'][cond]-t0,
                 BA_mean[cond]-BA_mean[cond2].mean(), color=cm.viridis(i/len(F_aff)), lw=2)
        ax3.fill_between(data['t'][cond]-t0,
                         BA_mean[cond]-BA_std[cond]-BA_mean[cond2].mean(),
                         BA_mean[cond]+BA_std[cond]-BA_mean[cond2].mean(),
                         color=cm.viridis(i/len(F_aff)), alpha=.5)
        ax1.plot(data['t'][cond]-t0,
                 data['faff'][cond]-data['faff'][cond2].mean(), color=cm.viridis(i/len(F_aff)), lw=3)
    set_plot(ax1, xlabel='time (ms)', ylabel='$\delta \\nu_a$  (Hz)', xticks=XTICKS)
    set_plot(ax2, xlabel='time (ms)', ylabel='$\delta \\nu_e$ (Hz)', xticks=XTICKS)
    set_plot(ax3, xlabel='time (ms)', ylabel='$\delta \\nu_e$ (Hz)', xticks=XTICKS)
    return fig1, fig2, fig3
    
def compare_two_regimes(Model, smooth=20):

    for f, color in zip(['sas.zip', 'bs.zip'], [Blue, Orange]):
        Model['filename'] = f
        F_aff, DATA = get_scan(Model)
        F_out = []
        for i, data in enumerate(DATA):
            cond = data['t']>350
            Fout.append(gaussian_smoothing(data['POP_ACT_RecExc'],
                                           int(smooth/data['dt']))[cond].max())
        ax.plot(F_aff, F_out, color=color)
    
    set_plot(ax, xlabel=' $\delta \\nu_a $ (Hz)', ylabel=' $\\delta \nu_e$ (Hz)')

def run_scan(Model, SEED=4):
    
    zf = zipfile.ZipFile(Model['filename'], mode='w')

    F_aff = np.linspace(Model['faff0'], Model['faff1'], Model['ngrid'])
    seeds = Model['SEEDS']

    Model['FILENAMES'] = np.empty((len(F_aff), len(seeds)), dtype=object)
    
    for i, j in product(range(len(F_aff)), range(len(seeds))):
        fn = Model['data_folder']+str(Model['faff_bsl'])+str(F_aff[i])+'_'+str(seeds[j])+\
             '_'+str(np.random.randint(100000))+'.h5'
        Model['FILENAMES'][i,j] = fn
        print('running configuration:', fn)
        Model['Faff0'] = Model['faff_bsl']
        Model['Faff1'] = F_aff[i]
        NTWK = run_3pop_ntwk_model(Model,
                                   faff_waveform_func=waveform,
                                   filename=fn,
                                   tstop=Model['tstop'], SEED=seeds[j])
        zf.write(fn)
        
    # writing the parameters
    np.savez(Model['filename'].replace('.zip', '_Model.npz'), **Model)
    zf.write(Model['filename'].replace('.zip', '_Model.npz'))
    
    zf.close()

    
if __name__=='__main__':
    
    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import *

    parser.add_argument('--T1', type=float, default=400)    
    parser.add_argument('--DT1', type=float, default=100)    
    parser.add_argument('--T2', type=float, default=1100)    
    parser.add_argument('--DT2', type=float, default=300)
    
    parser.add_argument("-N",'--ngrid', help="discretization of the grid", type=int, default=2)
    parser.add_argument('--SEEDS', nargs='+', help='various seeds', type=int,
                        default=np.arange(1))    

    # input
    parser.add_argument("--faff_bsl",help="Hz", type=float, default=4.)
    parser.add_argument("--faff0",help="Hz", type=float, default=1.)
    parser.add_argument("--faff1",help="Hz", type=float, default=9.)

    parser.add_argument('-df', '--data_folder', help='Folder for data', default='sparse_vs_balanced/data/')    
    parser.add_argument("--filename", '-f', help="filename", type=str, default='data.zip')
    parser.add_argument("-a", "--analyze", help="perform analysis of params space",
                        action="store_true")
    parser.add_argument("-c", "--compare_regimes",
                        help="perform analysis of params space",
                        action="store_true")
    
    
    args = parser.parse_args()
    Model = vars(args)

    if args.analyze:
        analyze_scan(Model)
        ntwk.show()
    elif args.compare_regimes:
        compare_two_regimes(Model)
        ntwk.show()
    else:
        run_scan(Model)
