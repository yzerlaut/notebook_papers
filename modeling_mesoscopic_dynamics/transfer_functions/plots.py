import numpy as np
import matplotlib.pylab as plt
import matplotlib
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from graphs.my_graph import *
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from transfer_functions.theoretical_tools import *

def make_exc_inh_fig(DATA, P=None):
    
    MEANfreq, SDfreq, Fe_eff, fiSim, params = np.load(DATA)
    Fe_eff, Fout = np.array(Fe_eff), np.array(MEANfreq)
    fiSim = np.meshgrid(np.zeros(Fe_eff.shape[1]), fiSim)[1]
    levels = np.unique(fiSim) # to store for colors
    
    if P is not None:
        params['P']=P
        
    # # #### FIGURE AND COLOR GRADIENT STUFF
    
    fig1 = plt.figure(figsize=(6,4))
    plt.subplots_adjust(bottom=.2, left=.15, right=.85, wspace=.2)
    ax = plt.subplot2grid((1,8), (0,0), colspan=7)
    ax_cb = plt.subplot2grid((1,8), (0,7))

    
    # -- Setting up a colormap that's a simple transtion
    mymap = get_linear_colormap(color1='k', color2='gray')
    # mymap = get_linear_colormap()
    build_bar_legend(np.round(levels,1), ax_cb, mymap,label='$\\nu_i$ inh. freq. (Hz)')
    
    for i in range(levels.size):

        SIMvector = MEANfreq[i][:]
        SDvector = SDfreq[i][:]
        feSim = Fe_eff[i][:]
        feth = np.linspace(feSim.min(), feSim.max(), 1e2)
        fi = fiSim[i][0]

        r = (float(levels[i])-levels.min())/(levels.max()-levels.min())
        ax.errorbar(feSim, SIMvector, yerr=SDvector,\
                    color=mymap(r,1),marker='D',ms=5, capsize=3, elinewidth=1, lw=0)
        if 'P' in params.keys():
            Fout_th = TF_my_template(feth, fi, *pseq_params(params))
            ax.plot(feth, Fout_th, color=mymap(r,1), lw=5, alpha=.5)

    set_plot(ax, ['bottom', 'left'], xlabel='$\\nu_e$ exc. freq. (Hz)',\
             ylabel='$\\nu_{out}$   output. freq. (Hz)')

if __name__=='__main__':

    # First a nice documentation 
    parser=argparse.ArgumentParser(description=
     """ 
     '=================================================='
     '=====> PLOT of the transfer function =============='
     '=================================================='
     """,
              formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-f', "--FILE",help="file name of numerical TF data",\
                        default='data/example_data.npy')
    parser.add_argument("--With_Square",help="Add the square terms in the TF formula"+\
                        "\n then we have 7 parameters",\
                         action="store_true")
    args = parser.parse_args()

    try:
        P = np.load(args.FILE.replace('.npy', '_fit.npy'))
    except IOError:
        P=None
    make_exc_inh_fig(args.FILE, P=P)
    plt.show()
