import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from analysis.phase_latency import find_latencies_over_space

def find_latencies_over_space_simple(t, X, signal,\
                              signal_criteria=0.01,\
                              baseline=0, discard=20,\
                              amp_criteria=1./4.):
    signal2 = np.abs(signal)-np.abs(signal[1,:]).mean()
    i_discard = int(discard/(t[1]-t[0]))
    t = t[i_discard:]
    signal2 = signal2[i_discard:,:]-baseline
    XX, TT = [], []
    for i in range(signal2.shape[1]):
        imax = np.argmax(signal2[:,i])
        if signal2[imax,i]>=signal_criteria*signal2.max():
            ii = np.argmin(np.abs(signal2[:imax,i]-amp_criteria*signal2[imax,i]))
            XX.append(X[i])
            TT.append(t[ii]+t[i_discard])
    return TT, XX

def space_time_vsd_style_plot(t, X, array,
                              Xwindow = 30., tzoom=[-150, 300.],
                              zlabel='rate (Hz)',\
                              xlabel='time (ms)', ylabel='cortical space', title='',
                              zticks = None,
                              zlim=None, with_latency_analysis=False,
                              bar_mm=2, Nlevels=10,
                              params={'pixels_per_mm':1.},
                              xzoom=None, yzoom=None):
    """
    takes an array of shape (t, X, Y) and plots its value in 2d for different times !
    at 8 different times
    it returns the figure
    """

    if yzoom is None:
        yzoom = [0, array.shape[1]]
    else:
        yzoom = np.array(yzoom)*params['pixels_per_mm']
    if xzoom is None:
        xzoom = [t[0], t[-1]]

    iXcenter = np.argmax(np.mean(array, axis=0))
    iTcenter = np.argmax(np.mean(array, axis=1))

    fig, ax = plt.subplots(1, figsize=(3.8,2.))
    # plt.suptitle(title, fontsize=22)
    plt.subplots_adjust(bottom=.23, top=.97, right=.85, left=.1)
    c = ax.contourf(t, X, array.T,
                    np.linspace(min([0,array.min()]), array.max(), Nlevels),
                    cmap=mpl.cm.viridis)
    if zticks is None:
        zticks = np.round(np.linspace(min([0,array.min()]), array.max(), 5),1)
    plt.colorbar(c, label=zlabel, ticks=zticks)
    ax.annotate(str(int(bar_mm))+'mm', (-.1, 0.1),
                xycoords='axes fraction', rotation=90, fontsize=13) # ba
    
    if with_latency_analysis:
        TT, XX = find_latencies_over_space_simple(t, X,
                                                  array, signal_criteria=5e-2,\
                                                  amp_criteria=1./5., discard=20)
        ax.plot(TT, XX, 'w--', lw=1)

    # bar annotation
    # now in pixel coordinates
    bar_mm *= params['pixels_per_mm']
    ax.annotate('50ms', (0.2, -0.1), xycoords='axes fraction', fontsize=13)
    set_plot(ax, [], xticks=[], yticks=[],
             ylim=[X[iXcenter]-Xwindow/2., X[iXcenter]+Xwindow/2.],
                   xlim=xzoom)
    ax.plot([xzoom[0], xzoom[0]], ax.get_ylim()[0]+np.array([0, bar_mm]), 'k-', lw=3)
    ax.plot([xzoom[0], xzoom[0]+50], ax.get_ylim()[0]*np.ones(2), 'k-', lw=3)
    
    return ax, fig
        
