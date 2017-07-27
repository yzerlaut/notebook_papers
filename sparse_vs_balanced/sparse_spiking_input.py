import numpy as np
from itertools import product
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk
from data_analysis.IO.hdf5 import load_dict_from_hdf5
from graphs.my_graph import *
# everything stored within a zip file
import zipfile
from scipy.special import erf
from data_analysis.processing.signanalysis import gaussian_smoothing
import matplotlib.cm as cm
from graphs.plot_export import put_list_of_figs_to_svg_fig
import pymuvr # for spike train metrics
from sparse_vs_balanced.running_3pop_model import run_3pop_ntwk_model

Blue, Orange, Green, Red, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

def get_scan(args, filename=None):

    if filename is None:
        filename=str(args.filename)
        
    zf = zipfile.ZipFile(filename, mode='r')
    DATA = []
    
    model_file = filename.replace('.zip', '_Model.npz')
    data = zf.read(model_file)
    with open(model_file, 'wb') as f: f.write(data)
    Model = dict(np.load(model_file).items())

    PATTERNS = Model['PATTERNS']
    seeds = Model['SEEDS']

    for i, j in product(range(len(PATTERNS)), range(len(seeds))):
        fn = Model['FILENAMES'][i,j]
        data = zf.read(fn)
        with open(fn, 'wb') as f: f.write(data)
        with open(fn, 'rb') as f: data = load_dict_from_hdf5(fn)
        t = np.arange(int(data['tstop']/data['dt']))*data['dt']
        data['t'] = t
        DATA.append(data)

    return PATTERNS, seeds, Model, DATA


def sum_up_fig(args):

    fig, ax = plt.subplots(1, figsize=(2,2))
    plt.subplots_adjust(left=.4, bottom=.2)
    ## SAS
    SAS = []
    for i in range(1, 6):
        args.filename = 'sas'+str(i)+'.zip'
        args2, DATA = get_scan(args)
        mean, std = get_spike_train_similarity(DATA,\
                            args.t0, args.time_span, args.N_target)
        SAS.append(100*mean)
    ax.bar([0], [np.array(SAS).mean()], yerr=[np.array(SAS).std()])
    ## BS
    BS = []
    for i in range(1, 6):
        args.filename = 'bs'+str(i)+'.zip'
        args2, DATA = get_scan(args)
        mean, std = get_spike_train_similarity(DATA,\
                            args.t0, args.time_span, args.N_target)
        BS.append(100*mean)
    ax.bar([1], [np.array(BS).mean()], yerr=[np.array(BS).std()])
    set_plot(ax, xticks=[0, 1.],
             xticks_labels=['SAS', 'BS'], ylabel='spike pattern \n similarity (%)')
    fig.savefig('fig.svg')
    ntwk.show()
    
def get_spike_train_distance(DATA, t0, time_span, nmax, cos=0.1, tau=2):

    SPIKES = []
    for i, data in enumerate(DATA):
        Trial = []
        ## EXCITATORY SPIKES
        cond = (data['tRASTER_RecExc']>t0) & (data['iRASTER_RecExc']<nmax) &\
               (data['tRASTER_RecExc']<t0+time_span)
        for ii in np.arange(nmax):
            Nrn = []
            i0 = np.argwhere(data['iRASTER_RecExc'][cond]==ii).flatten()
            Nrn += list(data['tRASTER_RecExc'][cond][i0].flatten())
            Trial.append(Nrn)
        SPIKES.append(Trial)

    DS_matrix = np.array(pymuvr.square_dissimilarity_matrix(SPIKES,\
                                            cos, tau, 'distance'))
    single_pairwise = np.tril(DS_matrix, -1).flatten()
    mean = np.mean(single_pairwise[single_pairwise>0])
    std = np.std(single_pairwise[single_pairwise>0])
    return mean, std

def get_spike_train_similarity(DATA, t0, time_span, nmax, cos=0., tau=3):

    SPIKES = []
    for i, data in enumerate(DATA):
        Trial = []
        ## EXCITATORY SPIKES
        cond = (data['tRASTER_RecExc']>t0) & (data['iRASTER_RecExc']<nmax) &\
               (data['tRASTER_RecExc']<t0+time_span)
        for ii in np.arange(nmax):
            Nrn = [0]
            i0 = np.argwhere(data['iRASTER_RecExc'][cond]==ii).flatten()
            Nrn += list(data['tRASTER_RecExc'][cond][i0].flatten()-t0) 
            Trial.append(Nrn)
        SPIKES.append(Trial)

    DS_matrix = np.array(pymuvr.square_dissimilarity_matrix(SPIKES,\
                                            cos, tau, 'inner product'))
    print(DS_matrix)
    for i in range(DS_matrix.shape[0]):
        for j in range(i):
                   DS_matrix[i,j]/=np.sqrt(DS_matrix[i,i]*DS_matrix[j,j])
    print(DS_matrix)
            
    single_pairwise = np.tril(DS_matrix, -1).flatten()
    print(single_pairwise)
    mean = np.mean(single_pairwise[single_pairwise>0])
    std = np.std(single_pairwise[single_pairwise>0])
    return mean, std

def analyze_scan(args, smooth=10):

    PATTERNS, seeds, Model, DATA = get_scan({}, filename=args.filename)

    fig1, ax1 = plt.subplots(1, figsize=(4,2.5))
    plt.subplots_adjust(bottom=.3, right=.95)
    fig2, ax2 = plt.subplots(1, figsize=(4,2.5))
    plt.subplots_adjust(bottom=.3, right=.95)
    COLORS = [Brown, Cyan, Pink, Purple, Pink, Grey, Kaki]
    # COLORS = [Cyan, Pink, Brown, Purple, Pink, Grey, Kaki]
    MARKERS = ['o', 'd', 'x']
    SIZES = [5, 3, 6]

    nmax_pre = min([args.Nvis_pre, args.N_aff_stim])
    ax1.fill_between([0,args.time_span], [0,0], [nmax_pre,nmax_pre], color='k', alpha=.1)
    nmax_post = min([args.Nvis_post, args.N_target])
    ax2.fill_between([0,args.time_span], [0,0], [nmax_post,nmax_post], color='k', alpha=.1)
    for i, data in enumerate(DATA):
        ii, j = 0, 0
        ## AFFERENT SPIKES
        cond = (data['tRASTER_PRE_in_terms_of_Pre_Pop'][j]>args.t0-args.pre_window) &\
               (data['tRASTER_PRE_in_terms_of_Pre_Pop'][j]<args.t0+args.time_span+args.pre_window) &\
               (data['iRASTER_PRE_in_terms_of_Pre_Pop'][j]<nmax_pre)
        if j==0:
            ax1.plot(data['tRASTER_PRE_in_terms_of_Pre_Pop'][j][cond]-args.t0,
                     data['iRASTER_PRE_in_terms_of_Pre_Pop'][j][cond]+ii, MARKERS[i],
                     ms=SIZES[i], color=COLORS[i], label='trial '+str(i+1))
        else:
            ax1.plot(data['tRASTER_PRE_in_terms_of_Pre_Pop'][j][cond]-args.t0,
                     data['iRASTER_PRE_in_terms_of_Pre_Pop'][j][cond]+ii, MARKERS[i],
                     ms=SIZES[i], color=COLORS[i])
        ## EXCITATORY SPIKES
        cond = (data['tRASTER_RecExc']>args.t0-args.pre_window) & (data['iRASTER_RecExc']<nmax_post) &\
               (data['tRASTER_RecExc']<args.t0+args.time_span+args.pre_window)
        if j==0:
            ax2.plot(data['tRASTER_RecExc'][cond]-args.t0, data['iRASTER_RecExc'][cond]+ii, MARKERS[i],
                     ms=SIZES[i], color=COLORS[i])
        else:
            ax2.plot(data['tRASTER_RecExc'][cond]-args.t0, data['iRASTER_RecExc'][cond]+ii, MARKERS[i],
                     ms=SIZES[i], color=COLORS[i])

    ax1.plot((-args.pre_window)*np.ones(2), [0,int(nmax_pre/4)-1], 'k-', lw=3)
    fig1.text(0.05, 0.5, str(int(nmax_pre/4))+' neurons', rotation=90, fontsize=14)
    ax2.plot((-args.pre_window)*np.ones(2), [0,int(nmax_post/4)-1], 'k-', lw=3)
    fig2.text(0.05, 0.5, str(int(nmax_post/4))+' neurons', rotation=90, fontsize=14)
    ax1.legend()
    set_plot(ax1, ['bottom'], xlabel='time from stim. onset (ms)',
             xlim=[ax1.get_xlim()[0], args.time_span+args.pre_window], yticks=[])
    ax1.xaxis.label.set_size(13)
    set_plot(ax2, ['bottom'], xlabel='time from stim. onset (ms)',
             xlim=[ax2.get_xlim()[0], args.time_span+args.pre_window], yticks=[])
    ax2.xaxis.label.set_size(13)
    put_list_of_figs_to_svg_fig([fig1, fig2], fig_name=args.filename.replace('.zip', '.svg'))
    ntwk.show()


def run_scan(Model, SEED=4):
    
    zf = zipfile.ZipFile(Model['filename'], mode='w')

    Model['PATTERNS'] = np.arange(1, Model['nspk_patterns']+1)
    
    Model['tstop'] = Model['t0']+Model['time_span']+Model['pre_window']
    Model['F_AffExc'] = Model['faff_bsl']

    Model['FILENAMES'] = np.empty((len(Model['PATTERNS']), len(Model['SEEDS'])), dtype=object)
    
    for i, j in product(range(len(Model['PATTERNS'])), range(len(Model['SEEDS']))):
        fn = Model['data_folder']+str(Model['faff_bsl'])+\
             str(Model['PATTERNS'][i])+'_'+str(Model['SEEDS'][j])+\
             '_'+str(np.random.randint(100000))+'.h5'
        Model['FILENAMES'][i,j] = fn
        print('running configuration:', fn)
        Model['Pattern_Seed'] = i
        NTWK = run_3pop_ntwk_model(Model,
                                   filename=fn, with_Vm=2,
                                   tstop=Model['tstop'], SEED=Model['SEEDS'][j],\
                                   with_synchronous_input=True)
        zf.write(fn)
        
    # writing the parameters
    np.savez(Model['filename'].replace('.zip', '_Model.npz'), **Model)
    zf.write(Model['filename'].replace('.zip', '_Model.npz'))
    
    zf.close()


if __name__=='__main__':
    
    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import *

    parser.add_argument("-N",'--nspk_patterns', help="discretization of the grid", type=int, default=1)
    parser.add_argument('--SEEDS', nargs='+', help='various seeds', type=int,
                        default=np.arange(1))
    # input
    parser.add_argument("--faff_bsl",help="Hz", type=float, default=5.)
    parser.add_argument("--N_aff_stim",help="unitless", type=int, default=100)
    parser.add_argument("--N_target",help="unitless", type=int, default=100)
    parser.add_argument("--N_duplicate",help="unitless", type=int, default=10)
    parser.add_argument("--faff1",help="Hz", type=float, default=20.)
    parser.add_argument("--t0",help="ms", type=float, default=400)
    parser.add_argument("--time_span",help="ms", type=float, default=500.)
    parser.add_argument("--pre_window",help="ms", type=float, default=200.)
    parser.add_argument("--Nvis_pre",help="number of visualized neurons", type=int, default=500)
    parser.add_argument("--Nvis_post",help="number of visualized neurons", type=int, default=5000)

    parser.add_argument('-df', '--data_folder', help='Folder for data', default='sparse_vs_balanced/data/')    
    parser.add_argument("--filename", '-f', help="filename", type=str, default='data.zip')
    parser.add_argument("-a", "--analyze", help="perform analysis of params space",
                        action="store_true")
    parser.add_argument("-c", "--compare_regimes",
                        help="perform analysis of params space",
                        action="store_true")
    
    
    args = parser.parse_args()
    Model = vars(args)

    Model['p_AffExc_DsInh'] = 0.07
    
    if args.analyze:
        analyze_scan(args)
        ntwk.show()
    elif args.compare_regimes:
        compare_two_regimes(Model)
        ntwk.show()
    else:
        run_scan(Model)

