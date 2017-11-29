from scipy.io import loadmat
import numpy as np
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
# from data_analysis.processing.signanalysis import * #gaussian_smoothing
from scipy.signal import convolve2d
import matplotlib.cm as cm
from scipy.optimize import minimize

from scipy.ndimage.filters import gaussian_filter1d
def gaussian_smoothing(signal, idt_sbsmpl=10):
    """Gaussian smoothing of the data"""
    return gaussian_filter1d(signal, idt_sbsmpl)

def get_time_max(t, data, debug=False, Nsmooth=1):
    spatial_average = np.mean(data, axis=0)
    smoothed = gaussian_smoothing(spatial_average, Nsmooth)[:-int(Nsmooth)]
    i0 = np.argmax(smoothed)
    t0 = t[:-int(Nsmooth)][i0]
    if debug:
        plt.plot(t, spatial_average)
        plt.plot(t[:-int(Nsmooth)], smoothed)
        plt.plot([t0], [smoothed[i0]], 'D')
    return t0

def get_stim_center(time, space, data,
                    Nsmooth=4, debug=False, tmax=0., window=100.):
    """ we smoothe the average over time and take the x position of max signal"""
    temporal_average = np.mean(\
                data[:,(time>tmax-window) & (time<tmax+window)], axis=1)
    smoothed = gaussian_smoothing(temporal_average, Nsmooth)[:-int(Nsmooth)]
    i0 = np.argmax(smoothed)
    x0 = space[:-int(Nsmooth)][i0]
    if debug:
        plt.plot(space, temporal_average)
        plt.plot(space[:-int(Nsmooth)], smoothed)
        plt.plot([x0], [smoothed[i0]], 'D')
    return x0

def get_data(filename,
             t0=-150, t1=100, debug=False,\
             Nsmooth=2,
             smoothing=None):

    # loading data
    delay = 0 
    f = loadmat(filename)
    data = 1e3*f['matNL'][0]['stim1'][0]
    data[np.isnan(data)] = 0 # blanking infinite data
    time = f['matNL'][0]['time'][0].flatten()
    space = f['matNL'][0]['space'][0].flatten()
    if smoothing is None:
        smoothing = np.ones((Nsmooth, Nsmooth))/Nsmooth**2
    smooth_data = convolve2d(data, smoothing, mode='same')
    # smooth_data = data  # REMOVE DATA SMOOTHING
    # apply time conditions
    cond = (time>t0-delay) & (time<t1-delay)
    new_time, new_data = np.array(time[cond]), np.array(smooth_data[:,cond])
    # get onset time
    tmax = get_time_max(new_time, new_data, debug=debug)
    x_center = get_stim_center(new_time, space, new_data, debug=debug,
                               tmax=tmax)
    return new_time-tmax, space-x_center, new_data

def reformat_model_data_for_comparison(model_data_filename,
                                       time_exp, space_exp, data_exp,
                                       model_normalization_factor=None,
                                       with_global_normalization=False,
                                       with_local_normalization=False):
    """

    """
    # loading model and centering just like in the model
    args2, t, X, Fe_aff, Fe, Fi, muVn = np.load(model_data_filename) # we load the data file
    t*=1e3 # bringing to ms
    X -= args2.X_extent/2.+args2.X_extent/args2.X_discretization/2.
    Xcond = (X>=space_exp.min()) & (X<=space_exp.max())
    space, new_muVn = X[Xcond], muVn.T[Xcond,:]
    t -= get_time_max(t, new_muVn)  # centering over time in the same than for data
    
    # let's construct the spatial subsampling of the data that
    # matches the spatial discretization of the model
    exp_data_common_sampling = np.zeros((len(space), len(time_exp)))
    for i, nx in enumerate(space):
        i0 = np.argmin(np.abs(nx-space_exp)**2)
        exp_data_common_sampling[i, :] = data_exp[i0,:]

    # let's construct the temporal subsampling of the model that
    # matches the temporal discretization of the data
    dt_exp = time_exp[1]-time_exp[0]
    model_data_common_sampling = np.zeros((len(space), len(time_exp)))
    for i, nt in enumerate(time_exp):
        i0 = np.argwhere(np.abs(t-nt)<dt_exp)
        if len(i0)>0:
            model_data_common_sampling[:, i] = new_muVn[:, i0[0][0]]
    
    if with_global_normalization:
        if model_normalization_factor is None:
            model_normalization_factor = model_data_common_sampling.max()
        model_data_common_sampling /= model_normalization_factor
        exp_data_common_sampling /= exp_data_common_sampling.max()
    elif with_local_normalization:
        # normalizing by local maximum over time
        for i, nx in enumerate(space):
            model_data_common_sampling[i, :] /= model_data_common_sampling[i,:].max()
            exp_data_common_sampling[i, :] /= exp_data_common_sampling[i,:].max()
            
    return time_exp, space, model_data_common_sampling, exp_data_common_sampling
        

def get_residual(new_time, space, new_data,
                 Nsmooth=2, Nlevels=10,
                 fn='example_data.npy',
                 model_normalization_factor=None,
                 with_plot=False):

    new_time, space,\
        model_data_common_sampling,\
        exp_data_common_sampling =\
                reformat_model_data_for_comparison(fn,
                    new_time, space, new_data,
                    model_normalization_factor=model_normalization_factor,
                    with_global_normalization=True)
    
    if with_plot:

        fig, AX = plt.subplots(2, figsize=(4.5,5))
        plt.subplots_adjust(bottom=.23, top=.97, right=.85, left=.3)
        plt.axes(AX[0])
        c = AX[0].contourf(new_time, space, exp_data_common_sampling,
           np.linspace(exp_data_common_sampling.min(), exp_data_common_sampling.max(), Nlevels),
                           cmap=cm.viridis)
        plt.colorbar(c, label='norm. VSD',
                     ticks=.5*np.arange(3))
        set_plot(AX[0], xticks_labels=[], ylabel='space (mm)')
        plt.axes(AX[1])

        # to have the zero at the same color level
        factor = np.abs(exp_data_common_sampling.min()/exp_data_common_sampling.max())
        model_data_common_sampling[-1,-1] = -factor*model_data_common_sampling.max()

        c2 = AX[1].contourf(new_time, space, model_data_common_sampling,
          np.linspace(model_data_common_sampling.min(), model_data_common_sampling.max(), Nlevels),
                            cmap=cm.viridis)
        
        plt.colorbar(c2, label='norm. $\\delta V_N$',
                     ticks=.5*np.arange(3))
        set_plot(AX[1], xlabel='time (ms)', ylabel='space (mm)')

    return np.sum((exp_data_common_sampling-model_data_common_sampling)**2)

if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description=
            """
            """,
            formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("--Nsmooth", help="for data plots", type=int, default=1)
    parser.add_argument("-s", "--save", help="save fig", action="store_true")
    parser.add_argument("-a", "--analyze", help="analyze", action="store_true")
    parser.add_argument("-p", "--plot", help="plot analysis", action="store_true")
    parser.add_argument("--space", help="space residual", action="store_true")
    parser.add_argument("--time", help="temporal residual", action="store_true")
    parser.add_argument("--model_filename", '-f', type=str, default='data/example_data.npy')
    parser.add_argument("--data_filename", default='data/VSD_data_session_example.mat')
    parser.add_argument("--t0", type=float, default=-np.inf)
    parser.add_argument("--t1", type=float, default=np.inf)
    parser.add_argument("--Nlevels", type=int, default=20)
    args = parser.parse_args()

    new_time, space, new_data = get_data(args.data_filename,
                                         Nsmooth=args.Nsmooth,
                                         t0=args.t0, t1=args.t1)
    print(get_residual(new_time, space, new_data,
                       Nsmooth=args.Nsmooth,
                       fn=args.model_filename,
                       with_plot=True))
