import numpy as np
import matplotlib.pylab as plt
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from heterogeneous_firing_responses.numerical_simulations.models import models, get_model_params
from heterogeneous_firing_responses.numerical_simulations.simulations import single_experiment

t = np.arange(0,700,1e-2)*1e-3

def make_model_figure(MODEL, I0=100e-12,
                      savefig=False, for_title=None):

    params = get_model_params(MODEL, {})
    
    Inorm = np.array([1. if (tt>100e-3 and tt<500e-3) else 0 for tt in t])
    if 'Istep' in params:
        I0 = params['Istep']
    I = I0*Inorm
        
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(111, frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    plt.title(params['name']+'\n', fontsize=25)
    
    if MODEL.split('-')[0]=='aEIF':
        [v1, va1], spikes = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                                  MODEL=MODEL, full_I_traces=I)
        [v2, va2], spikes2 = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                                  MODEL=MODEL, full_I_traces=-I)
        plt.plot(1e3*t, 1e3*v1, 'k', label='soma')
        plt.plot(1e3*t, 1e3*va1, 'r', label='axon')
        plt.plot(1e3*t, -85+10*Inorm, 'k')
        plt.plot(1e3*t, 1e3*v2, 'k--')
        plt.plot(1e3*t, 1e3*va2, 'r--')
        plt.plot(1e3*t, -85-10*Inorm, 'k--')
        plt.legend(loc='upper right', frameon=False)
        
    elif MODEL.split('-')[0]=='iLIF' or MODEL.split('-')[0]=='iAdExp':
        v1, va1, spikes = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                     MODEL=MODEL, full_I_traces=I, return_threshold=True)
        v2, va2, spikes2 = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                     MODEL=MODEL, full_I_traces=-I, return_threshold=True)
        plt.plot(1e3*t, 1e3*v1, 'k', label=r'$V_\mathrm{m}$')
        plt.plot(1e3*t, 1e3*va1, 'r', label=r'$\theta$')
        plt.plot(1e3*t, -85+10*Inorm, 'k')
        plt.plot(1e3*t, 1e3*v2, 'k--')
        plt.plot(1e3*t, -85-10*Inorm, 'k--')
        plt.legend(loc='upper right', frameon=False)
    else:
        v1, spikes = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                                  MODEL=MODEL, full_I_traces=I)
        v2, spikes2 = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                                  MODEL=MODEL, full_I_traces=-I)
        plt.plot(1e3*t, 1e3*v1, 'k')
        plt.plot(1e3*t, -85+10*Inorm, 'k')
        plt.plot(1e3*t, 1e3*v2, 'k--')
        plt.plot(1e3*t, -85-10*Inorm, 'k--')

    if MODEL is not 'Wang-Buszaki':
        for s in spikes:
            plt.plot([1e3*s,1e3*s], [5e3*params['delta_v']+\
                                     1e3*params['vthresh'],20], 'k:')
        for s in spikes2:
            plt.plot([1e3*s,1e3*s], [5e3*params['delta_v']+\
                                     1e3*params['vthresh'],20], 'k:')

    plt.tight_layout()
    plt.plot([10,10],[-25,-15], 'gray', lw=3)
    plt.plot([10,60],[-25,-25], 'gray', lw=3)
    plt.annotate('10mV', (16,-10), textcoords='data', size=13)
    plt.annotate(str(int(1e12*I0))+'pA', (16,-20), textcoords='data',size=13)
    plt.annotate('50ms', (17,-40), textcoords='data', size=13)
    if savefig==True:
        fig.savefig('../figures/'+MODEL+'_step_response.svg',\
                    format='svg', transparent=True)
    return fig
    
def make_2models_figure(ax, MODEL1, MODEL2, I0=300e-12,\
                        color1='r', color2='b'):

    params = get_model_params(MODEL1, {})
    
    Inorm = np.array([1. if (tt>100e-3 and tt<500e-3) else 0 for tt in t])
    if 'Istep' in params:
        I0 = params['Istep']
    I = I0*Inorm

    if (MODEL1.split('-')[0]=='iAdExp') or (MODEL1.split('-')[0]=='iLIF'):
        v1, va1, spikes1 = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                              MODEL=MODEL1, full_I_traces=I, return_threshold=True)
        v2, va2, spikes2 = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                                  MODEL=MODEL2, full_I_traces=I, return_threshold=True)
    else:
        v1, spikes1 = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                                  MODEL=MODEL1, full_I_traces=I)
        v2, spikes2 = single_experiment(t, 0., 0., 0., 0., 1e9, -70e-3, params,\
                                  MODEL=MODEL2, full_I_traces=I)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.plot(1e3*t, 1e3*v1, color=color1)
    ax.plot(1e3*t, -85+10*Inorm, 'k')
    ax.plot(1e3*t, 1e3*v2+30, color=color2)
    if (MODEL1.split('-')[0]=='iAdExp') or (MODEL1.split('-')[0]=='iLIF'):
        ax.plot(1e3*t, 1e3*va1, 'k:')
        ax.plot(1e3*t, 1e3*va2+30, 'k:')
    ax.plot([1e3*t[0], 1e3*t[0]], [1e3*v1[0], 1e3*v2[0]+30], 'k>', ms=10)
    ax.annotate('-70mV', (.1,.3), xycoords='axes fraction')


    # plt.tight_layout()
    ax.plot([10,10],[-25,-15], 'gray', lw=3)
    ax.plot([10,60],[-25,-25], 'gray', lw=3)
    ax.annotate('10mV', (16,-10), textcoords='data', size=13)
    ax.annotate(str(int(abs(1e12*I0)))+'pA', (16,-20), textcoords='data',size=13)
    ax.annotate('50ms', (17,-40), textcoords='data', size=13)
    ax.annotate('-70mV', (-10, -70), textcoords='data', size=13)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('MODEL', default='LIF')
    parser.add_argument('--I0', type=float, default=100., help='value of current input')
    args = parser.parse_args()

    if args.MODEL!='full':
        fig = make_model_figure(args.MODEL, I0=args.I0*1e-12, savefig=False)
        plt.show()
    else:
        for m in models('all_models'):
            fig = make_model_figure(m, I0=args.I0*1e-12, savefig=True)


