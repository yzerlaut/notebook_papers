import numpy as np
import matplotlib.pylab as plt
import time
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from numerical_simulations.models import get_model_params
# import numba
"""
the numba implementation doesn't work anymore with python 3, no idea why, but it works without, it's just a bit slower...
"""

####################################################################
############ Functions for the spiking dynamics ###########
####################################################################

def generate_conductance_shotnoise(freq, t, N, Q, Tsyn, g0=0, seed=0):
    """
    generates a shotnoise convoluted with a waveform
    frequency of the shotnoise is freq,
    K is the number of synapses that multiplies freq
    g0 is the starting value of the shotnoise
    """
    if freq==0:
        # print("problem, 0 frequency !!! ---> freq=1e-9 !!")
        freq=1e-9
    upper_number_of_events = max([int(3*freq*t[-1]*N),1]) # at least 1 event
    np.random.seed(seed=seed)
    spike_events = np.cumsum(np.random.exponential(1./(N*freq),\
                             upper_number_of_events))
    g = np.ones(t.size)*g0 # init to first value
    dt, t = t[1]-t[0], t-t[0] # we need to have t starting at 0
    # stupid implementation of a shotnoise
    event = 0 # index for the spiking events
    for i in range(1,t.size):
        g[i] = g[i-1]*np.exp(-dt/Tsyn)
        while spike_events[event]<=t[i]:
            g[i]+=Q
            event+=1
    return g

### ================================================
### ======== iAdExp model (general) ================
### == extension of LIF, iLIF, EIF, AdExp, ...
### ================================================

def pseq_iAdExp(cell_params):

    El, Gl = cell_params['El'], cell_params['Gl']
    Cm = cell_params['Cm']
    
    vthresh, vreset, vspike, vpeak =\
                 cell_params['vthresh'], cell_params['vreset'],\
                 cell_params['vspike'], cell_params['vpeak']

    # adaptation variables
    a, b, tauw = cell_params['a'],\
                     cell_params['b'], cell_params['tauw']

    # spike variables
    trefrac, delta_v = cell_params['trefrac'], cell_params['delta_v']

    # inactivation variables
    Vi, Ti = cell_params['Vi'], cell_params['Ti']
    Ai = cell_params['Ai']

    return El, Gl, Cm, vthresh, vreset, vspike, vpeak,\
                     trefrac, delta_v, a, b, tauw, Vi, Ti, Ai
                     
# @numba.jit('u1[:](f8[:], f8[:], f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8, f8)')
def iAdExp_sim(t, I, Gs, muV,
         El, Gl, Cm, vthresh, vreset, vspike, vpeak,\
         trefrac, delta_v, a, b, tauw, Vi, Ti, Ai):
    """ functions that solve the membrane equations for the
    adexp model for 2 time varying excitatory and inhibitory
    conductances as well as a current input
    returns : v, spikes
    """

    if delta_v==0: # i.e. Integrate and Fire
        one_over_delta_v = 0
    else:
        one_over_delta_v = 1./delta_v
        
    vspike=vthresh+5.*delta_v # practical threshold detection
            
    last_spike = -np.inf # time of the last spike, for the refractory period
    V, spikes = vreset*np.ones(len(t), dtype=np.float), []
    theta=vthresh*np.ones(len(t), dtype=np.float) # initial adaptative threshold value
    dt = t[1]-t[0]

    w, i_exp = 0., 0. # w and i_exp are the exponential and adaptation currents

    for i in range(len(t)-1):
        w = w + dt/tauw*(a*(V[i]-El)-w) # adaptation current
        i_exp = Gl*delta_v*np.exp((V[i]-vthresh)*one_over_delta_v) 
        
        if (t[i]-last_spike)>trefrac: # only when non refractory
            ## Vm dynamics calculus
            V[i+1] = V[i] + dt/Cm*(I[i] + i_exp - w +\
                                Gl*(El-V[i]) + Gs*(muV-V[i]) )

        # then threshold
        theta_inf_v = vthresh + Ai*0.5*(1+np.sign(V[i]-Vi))*(V[i]-Vi)
        theta[i+1] = theta[i] + dt/Ti*(theta_inf_v - theta[i])
        
        if V[i+1] >= theta[i+1]+5.*delta_v:
            
            V[i+1] = vreset # non estethic version
            
            w = w + b # then we increase the adaptation current
            last_spike = t[i+1]
            spikes.append(t[i+1])

    return V, theta, np.array(spikes)


def params_variations_calc(muGn, muV, sV, Ts_ratio, params):
    """
    input should be numpy arrays !!

    We solve the equations:
    Ts = Tv - \frac{C_m}{\mu_G}
    Q \, T_S (\nu_e + \nu_i) = \mu_G
    Q \, T_S (\nu_e E_e + \nu_i E_i) = \mu_G \mu_V - g_L E_L
    Q^2 \, T_S^2 \, big( \nu_e (E_e-\mu_V)^2 +
        \nu_i (E_i - \mu_V)^2 \big) = 2 \mu_G^2 \tau_V \sigma_V^2

    return numpy arrays !!
    """

    Gl, Cm, El = params['Gl'], params['Cm'], params['El']
    Tm0 = Cm/Gl
    Ts = Ts_ratio*Tm0
    DV = params['Driving_Force']
    muG = muGn*Gl
    Gs = muG-Gl # shunt conductance
    Tv = Ts+Tm0/muGn
    I0 = Gl*(muV-El) # current to bring at mean !
    f = 2000.+0*I0 #Hz
    Q = muG*sV*np.sqrt(Tv/f)/Ts/DV
    
    return I0, Gs, f, Q, Ts

def single_experiment(t, I0, Gs, f, Q, Ts, muV,\
                      params, MODEL='SUBTHRE', seed=0, full_I_traces=None,\
                      return_threshold=False):

    params = params.copy()
    
    Ge = generate_conductance_shotnoise(f, t, 1.,\
                            Q, Ts, g0=0, seed=seed)
    Gi = generate_conductance_shotnoise(f, t, 1.,\
                            Q, Ts, g0=0, seed=seed**2+1)

    # current input !!!
    if full_I_traces is None:
        I = np.ones(len(t))*I0+(Ge-Gi)*params['Driving_Force']
    else:
        I= full_I_traces

    params = get_model_params(MODEL, params)
    
    v, theta, spikes = iAdExp_sim(t, I, Gs, muV, *pseq_iAdExp(params))

    if return_threshold:
        return v, theta, spikes
    else:
        return v, spikes
        


