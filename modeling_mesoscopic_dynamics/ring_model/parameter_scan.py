import numpy as np
import matplotlib.pylab as plt
import itertools
# everything stored within a zip file
import zipfile
import sys, pathlib, os
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ring_model.compare_to_model import get_data, get_residual
from ring_model.model import Euler_method_for_ring_model

FACTOR_FOR_MUVN_NORM = abs((-54.+58.)/58.) # ~6% corresponding to a ~5mV wrt to rest level

def to_filename(vc, se, ecr, icr, t2, t1):
    return 'data/scan_'+str(vc)+'_'+str(se)+'_'+str(ecr)+'_'+str(icr)+'_'+str(t2)+'_'+str(t1)+'.npy'

def cmd(vc, se, ecr, icr, t2, t1):
    fn = to_filename(vc, se, ecr, icr, t2, t1)
    return fn, 'python modeling_mesoscopic_dynamics/ring_model/single_trial.py '+\
        ' --conduction_velocity_mm_s '+str(vc)+\
        ' --sX '+str(se)+\
        ' --exc_connect_extent '+str(ecr)+\
        ' --inh_connect_extent '+str(icr)+\
        ' --Tau2 '+str(t2)+' --Tau1 '+str(t1)+' -f '+fn+\
        ' --no_plot --X_extent 36 --X_discretization 30'

def create_grid_scan_bash_script(args):
    
    VC = np.linspace(args.vc[0], args.vc[1], args.N)
    SE = np.linspace(args.stim_extent[0], args.stim_extent[1], args.N)
    ECR = np.linspace(args.Econn_radius[0], args.Econn_radius[1], args.N)
    ICR = np.linspace(args.Iconn_radius[0], args.Iconn_radius[1], args.N)
    TAU2 = np.linspace(args.Tau2[0], args.Tau2[1], args.N)
    TAU1 = np.linspace(args.Tau1[0], args.Tau1[1], args.N)

    f = open('modeling_mesoscopic_dynamics/ring_model/bash_parameter_scan.sh', 'w')
    FILENAMES = []
    n = 0 # sim counter
    for vc, se, ecr, icr, t2, t1 in itertools.product(VC, SE, ECR, ICR, TAU2, TAU1):
        fn, c = cmd(vc, se, ecr, icr, t2, t1)
        n+=1
        if (t1<TAU1[-1]) or (t2<TAU2[-1]):
            c += ' & \n'
        else:
            c += ' \n'
        if (args.force==True) or (os.path.isfile(fn)==False):
            f.write(c)
        else:
            print('existing datafile')
        FILENAMES.append(fn)
    f.close()
    np.save('data/scan_data.npy', [VC, SE, ECR, ICR, TAU2, TAU1, np.array(FILENAMES)])
    print(n, 'simulations to be performed')


def zip_data(args):
    zf = zipfile.ZipFile(args.zip_filename, mode='w')
    # writing the parameters
    zf.write('data/scan_data.npy')
    VC, SE, ECR, ICR, TAU2, TAU1, FILENAMES = np.load('data/scan_data.npy')
    for fn in FILENAMES:
        zf.write(fn)
    zf.close()

def unzip_data(args):
    zf = zipfile.ZipFile(args.zip_filename, mode='r')
    # writing the parameters
    data = zf.read('data/scan_data.npy')
    with open('data/scan_data.npy', 'wb') as f: f.write(data)
    VC, SE, ECR, ICR, TAU2, TAU1, FILENAMES = np.load('data/scan_data.npy')
    for fn in FILENAMES:
        data = zf.read(fn)
        with open(fn, 'wb') as f: f.write(data)
    zf.close()
    
def analyze_scan(args):
    
    VC, SE, ECR, ICR, TAU2, TAU1, FILENAMES = np.load('data/scan_data.npy')
    
    Residuals = []
    vcFull, seFull, ecrFull, icrFull, t2Full, t1Full = [], [], [], [], [], []
    
    ## loading data for time residual
    new_time, space, new_data = get_data(args.vsd_data_filename,
                                         t0=args.t0, t1=args.t1)

    for vc, se, ecr, icr, t2, t1 in itertools.product(VC, SE, ECR, ICR, TAU2, TAU1):
        fn = to_filename(vc, se, ecr, icr, t2, t1)
        try:
            res = get_residual(new_time, space, new_data,
                               model_normalization_factor=FACTOR_FOR_MUVN_NORM,
                               fn=fn)
            Residuals.append(res)
            t2Full.append(t2)
            t1Full.append(t1)
            vcFull.append(vc)
            seFull.append(se)
            ecrFull.append(ecr)
            icrFull.append(icr)
        except (IOError, OSError):
            print('missing', fn)

    np.save('data/residuals_data.npy',
            [np.array(Residuals),
             np.array(vcFull), np.array(seFull),
             np.array(ecrFull), np.array(icrFull),
             np.array(t2Full), np.array(t1Full)])

def fix_missing(args):
    
    VC, SE, ECR, ICR, TAU2, TAU1, FILENAMES = np.load('data/scan_data.npy')
    f = open('bash_parameter_scan.sh', 'w')
    n = 0
    for vc, se, ecr, icr, t2, t1 in itertools.product(VC, SE, ECR, ICR, TAU2, TAU1):
        fn, c = cmd(vc, se, ecr, icr, t2, t1)
        try:
            _ = np.load(fn)
        except (IOError, OSError):
            n+=1
            if (n%args.simultaneous_sims==(args.simultaneous_sims-1)):
                c+=' \n'
            else:
                c+=' & \n'
            if not os.path.isfile(fn):
                print('missing:', fn)
            else:
                print('broken:', fn)
            f.write(c)
    f.close()
    print('need to redo', n, 'sim over the ', len(FILENAMES), 'simulation grid')
    
def plot_analysis(args):
    
    Residuals,\
        vcFull, seFull, ecrFull, icrFull,\
        t2Full, t1Full = np.load(\
            'data/residuals_data.npy')

    i0 = np.argmin(Residuals)
    Residuals/=Residuals[i0] # normalizing
    
    fig, AX = plt.subplots(1, 6, figsize=(9,2.))
    plt.subplots_adjust(bottom=.3, left=.15)
    for ax, vec, label in zip(AX,
                              [vcFull, seFull, ecrFull, icrFull, t2Full, t1Full],\
                              ['$v_c (mm/s)$','$l_{stim}$ (mm)',
                               '$l_{exc}$ (mm)', '$l_{inh}$ (mm)',
                               '$\\tau_2$ (ms)', '$\\tau_1$ (ms)']):
        ax.plot(vec, Residuals, 'o')
        ax.plot([vec[i0]], [Residuals[i0]], 'ro')
        ax.set_yscale('log')
        if ax==AX[0]:
            set_plot(ax, xlabel=label, ylabel='Residual (norm.)',
                     yticks=[1, 2, 5, 10, 20], yticks_labels=['1', '2', '5', '10', '20'])
        else:
            set_plot(ax, xlabel=label, yticks=[1, 5, 10, 20], yticks_labels=[])

    new_time, space, new_data = get_data(args.vsd_data_filename,
                                         Nsmooth=args.Nsmooth,
                                         t0=args.t0, t1=args.t1)

    if args.force:
        fn = 'data/model_data.npy'
        t, X, Fe_aff, Fe, Fi, muVn =\
                                 Euler_method_for_ring_model(\
                                                             'RS-cell', 'FS-cell',\
                                                             'CONFIG1', 'RING1', 'CENTER',\
                                        custom_ring_params={\
                                                            'X_discretization':args.X_discretization,
                                                            'X_extent':args.X_extent,
                                                            'conduction_velocity_mm_s':vcFull[i0],
                                                            'exc_connect_extent':ecrFull[i0],
                                                            'inh_connect_extent':icrFull[i0]},
                                        custom_stim_params={\
                                                            'sX':seFull[i0], 'amp':15.,
                                                            'Tau1':t1Full[i0], 'Tau2':t2Full[i0]})
        np.save(fn, [args, t, X, Fe_aff, Fe, Fi, muVn])
    else:
        _, _, _, _, _, _, FILENAMES = np.load('data/scan_data.npy')
        fn = FILENAMES[i0]
    
    res = get_residual(new_time, space, new_data,
                       Nsmooth=args.Nsmooth,
                       fn=fn, with_plot=True)
    

def get_minimum_params(args):
    Residuals,\
        vcFull, seFull, ecrFull, icrFull,\
        t2Full, t1Full = np.load(\
            'data/residuals_data.npy')

    i0 = np.argmin(Residuals)
    Residuals/=Residuals[i0] # normalizing
    return vcFull[i0], seFull[i0], ecrFull[i0], icrFull[i0], t2Full[i0], t1Full[i0]

def full_analysis(args):

    DATA = get_dataset()
    for i in range(len(DATA)):
        print('analyzing cell ', i, ' [...]')
        args.data_index = i
        analyze_scan(args)

from scipy.stats import ttest_rel, pearsonr

def full_plot(args):

    DATA = get_dataset()
    VC, SE, ECR, ICR, TAU2, TAU1, DUR, MONKEY = [], [], [], [], [], [], [], []
    for i in range(len(DATA)):
        args.data_index = i
        params = get_minimum_params(args)
        for vec, VEC in zip(params, [VC, SE, ECR, ICR, TAU2, TAU1]):
            VEC.append(vec)
        DUR.append(DATA[i]['duration'])
        MONKEY.append(DATA[i]['Monkey'])

    # vc
    fig1, ax1 = plt.subplots(1, figsize=(1.5,2.3));plt.subplots_adjust(bottom=.4, left=.6)
    ax1.fill_between([-1., 1.], np.ones(2)*args.vc[0], np.ones(2)*args.vc[1],
                       color='lightgray', alpha=.8, label=r'$\mathcal{D}$ domain')
    ax1.bar([0], [np.array(VC).mean()], yerr=[np.array(VC).std()],
               color='lightgray', edgecolor='k', lw=3)
    ax1.legend(frameon=False)
    print('Vc = ', round(np.array(VC).mean()), '+/-', round(np.array(VC).std()), 'mm/s')
    set_plot(ax1, ['left'], xticks=[], ylabel='$v_c$ (mm/s)')
    # connectivity
    fig2, ax2 = plt.subplots(1, figsize=(2.,2.3));plt.subplots_adjust(bottom=.4, left=.6)
    ax2.bar([0], [np.array(ECR).mean()], yerr=[np.array(ECR).std()],
               color='lightgray', edgecolor='g', lw=3, label='$l_{exc}$')
    print('Ecr=', round(np.array(ECR).mean(),1), '+/-', round(np.array(ECR).std(),1), 'mm/s')
    ax2.bar([1.5], [np.array(ICR).mean()], yerr=[np.array(ICR).std()],
               color='lightgray', edgecolor='r', lw=3, label='$l_{inh}$')
    print('Icr=', round(np.array(ICR).mean(),1), '+/-', round(np.array(ICR).std(),1), 'mm/s')
    ax2.fill_between([-1., 2.5], np.ones(2)*args.Econn_radius[0],
                       np.ones(2)*args.Econn_radius[1],
                       color='lightgray', alpha=.8)
    ax2.legend(frameon=False)
    ax2.annotate("p=%.1e" % ttest_rel(ECR, ICR).pvalue, (0.1, .1), xycoords='figure fraction')
    set_plot(ax2, ['left'], xticks=[], ylabel='extent (mm)')
    # stim extent
    fig3, ax3 = plt.subplots(1, figsize=(1.5,2.3));plt.subplots_adjust(bottom=.4, left=.6)
    ax3.bar([0], [np.array(SE).mean()], yerr=[np.array(SE).std()],
               color='lightgray', edgecolor='k', lw=3)
    print('Ecr=', round(np.array(SE).mean(),1), '+/-', round(np.array(SE).std(),1), 'mm/s')
    ax3.fill_between([-1., 1.], np.ones(2)*args.stim_extent[0], np.ones(2)*args.stim_extent[1],
                       color='lightgray', alpha=.8)
    set_plot(ax3, ['left'], xticks=[], ylabel='$l_{stim}$ (mm)', yticks=[0,1,2])

    DUR, TAU1, TAU2 = np.array(DUR), 1e3*np.array(TAU1), 1e3*np.array(TAU2)
    
    fig4, ax4 = plt.subplots(1, figsize=(2.5,2.3));plt.subplots_adjust(bottom=.4, left=.6)
    for d in np.unique(DUR):
        ax4.errorbar([d], [TAU1[DUR==d].mean()], yerr=[TAU1[DUR==d].std()], marker='o', color='k')
    ax4.plot([DUR.min(), DUR.max()],
             np.polyval(np.polyfit(DUR, TAU1, 1), [DUR.min(), DUR.max()]), 'k--', lw=0.5)
    ax4.fill_between([DUR.min(), DUR.max()],
                     1e3*np.ones(2)*args.Tau1[0], 1e3*np.ones(2)*args.Tau1[1],
                       color='lightgray', alpha=.8)
    ax4.annotate("c=%.1e" % pearsonr(DUR, TAU1)[0], (0.1, .2), xycoords='figure fraction')
    ax4.annotate("p=%.1e" % pearsonr(DUR, TAU1)[1], (0.1, .1), xycoords='figure fraction')
    set_plot(ax4, xticks=[10, 50, 100],
             xlabel='$T_{stim}$ (ms)', ylabel='$\\tau_1$ (ms)', yticks=[0, 25, 50])
    
    fig5, ax5 = plt.subplots(1, figsize=(2.5,2.3));plt.subplots_adjust(bottom=.4, left=.6)
    for d in np.unique(DUR):
        ax5.errorbar([d], [TAU2[DUR==d].mean()], yerr=[TAU2[DUR==d].std()], marker='o', color='k')
    ax5.plot([DUR.min(), DUR.max()],
             np.polyval(np.polyfit(DUR, TAU2, 1), [DUR.min(), DUR.max()]), 'k--', lw=0.5)
    ax5.fill_between([DUR.min(), DUR.max()],
                     1e3*np.ones(2)*args.Tau2[0], 1e3*np.ones(2)*args.Tau2[1],
                       color='lightgray', alpha=.8)
    ax5.annotate("c=%.1e" % pearsonr(DUR, TAU2)[0], (0.1, .2), xycoords='figure fraction')
    ax5.annotate("p=%.1e" % pearsonr(DUR, TAU2)[1], (0.1, .1), xycoords='figure fraction')
    set_plot(ax5, xticks=[10, 50, 100],
             xlabel='$T_{stim}$ (ms)', ylabel='$\\tau_2$ (ms)', yticks=[40, 120, 200])

    
    
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=
            """
            runs a single trial with all options possible
            """,
            formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--vc", nargs=2, type=float, default=[50., 600.])
    parser.add_argument("--stim_extent", nargs=2, type=float, default=[0.2, 2.])
    parser.add_argument("--Econn_radius", nargs=2, type=float, default=[1., 7.])
    parser.add_argument("--Iconn_radius", nargs=2, type=float, default=[1., 7.])
    parser.add_argument("--Tau1", nargs=2, type=float, default=[5e-3, 50e-3])
    parser.add_argument("--Tau2", nargs=2, type=float, default=[50e-3, 200e-3])
    parser.add_argument("--N", type=int, default=2)
    parser.add_argument("--zip_filename", '-f', type=str, default='data/data.zip')
    # data
    parser.add_argument("--vsd_data_filename", default='data/VSD_data_session_example.mat')
    parser.add_argument("--t0", type=float, default=-100.)
    parser.add_argument("--t1", type=float, default=300.)
    parser.add_argument("--X_discretization", type=int, default=30) # PUT 100 for HD
    parser.add_argument("--X_extent", type=float, default=36.)
    parser.add_argument("--Nsmooth", help="for data plots", type=int, default=2)
    parser.add_argument("--simultaneous_sims",
                        help="Number of sims launched simultaneoulsy on the cluster",
                        type=int, default=10)
    parser.add_argument("--Nlevels", type=int, default=20)
    # script function
    parser.add_argument("-s", "--save", help="save fig", action="store_true")
    parser.add_argument("-a", "--analyze", help="analyze", action="store_true")
    parser.add_argument("-p", "--plot", help="plot analysis", action="store_true")
    parser.add_argument("-z", "--zip", help="zip datafiles", action="store_true")
    parser.add_argument("-uz", "--unzip", help="unzip datafiles", action="store_true")
    parser.add_argument("-d", "--debug", help="with debugging", action="store_true")
    parser.add_argument("--force", help="force simulation", action="store_true")
    parser.add_argument("--full", help="full analysis", action="store_true")
    parser.add_argument("--full_plot", help="plot of full analysis", action="store_true")
    parser.add_argument("--fix_missing", help="plot of full analysis", action="store_true")
    
    args = parser.parse_args()
    if args.analyze:
        analyze_scan(args)
    elif args.plot:
        plot_analysis(args)
    elif args.zip:
        zip_data(args)
    elif args.unzip:
        unzip_data(args)
    elif args.full:
        full_analysis(args)
    elif args.full_plot:
        full_plot(args)
    elif args.fix_missing:
        fix_missing(args)
    else:
        create_grid_scan_bash_script(args)
