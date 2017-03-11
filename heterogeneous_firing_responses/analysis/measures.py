#!/usr/bin/python
# Filename: measures.py

import numpy as np
from scipy.optimize import curve_fit

def autocorrel_func(signal, tmax, dt):
    """
    argument : signal (np.array), tmax and dt (float)
    tmax, is the maximum length of the autocorrelation that we want to see
    returns : autocorrel (np.array), time_shift (np.array)
    take a signal of time sampling dt, and returns its autocorrelation
     function between [0,tstop] (normalized) !!
    """
    steps = int(tmax/dt) # number of steps to sum on
    signal = (signal-signal.mean())/signal.std()
    cr = np.correlate(signal[steps:],signal)/steps
    time_shift = np.arange(len(cr))*dt
    return cr/cr.max(),time_shift


def measuring_subthre_dynamics(v, spikes, dt,\
                               discard=100e-3,refrac1=2e-3, refrac2=7e-3,
                               autocorrel_window=100e-3, Tm0=30e-3):
    """
    we discard the first 100ms
    then we need to hande the fact that you have spikes !!!
    so we discard a first refrac1 period followed by the refrac2 spike period

    So we take temporal slices defined by this criteria and then we sum their contribution
    associated with their representative weights
    """
    spikes = spikes[spikes>discard] # we discard the first xxx ms
    
    ir1, ir2 = int(refrac1/dt), int(refrac2/dt) # refractory steps
    i_start_slice, i_end_slice = [], []
    i_start_slice.append(int(discard/dt)) # we start at discard

    if len(spikes)>0:
        i_spikes = np.array(spikes/dt, dtype='int')
        for i_s in i_spikes:
            # we give 0 length if two spikes overlap
            i_end_slice.append(max([i_s-ir1,i_start_slice[-1]]))
            i_start_slice.append(i_s+ir2)
    # then need to handle the final case
    i_end_slice.append(len(v)-1)

    total_weight, ac_weight = 0,0 # weight for the all the contributions
    muV_sum, sV_sum, Tv_trace_sum = 0, 0, np.zeros(int(autocorrel_window/dt)+1)
    
    for it0, it1 in zip(i_start_slice, i_end_slice):

        if it1>it0:
            weight = (it1-it0)/1000. # weight as step number over 1000.
            muV_sum += weight*v[it0:it1].mean()
            sV_sum += weight*v[it0:it1].std()
            total_weight += weight
        
        if (it1-it0)>int(autocorrel_window/dt): # if bigger than the window
            v_acf, t_shift = autocorrel_func(v[it0:it1], autocorrel_window, dt)
            Tv_trace_sum += v_acf*weight
            ac_weight += weight

    if total_weight>0:
        muV_sum /= total_weight
        sV_sum /= total_weight

    if ac_weight>0:
        Tv_trace_sum /= ac_weight
        # then fit of a typical time
        exp_f = lambda t, tau: np.exp(-t/tau) # exponential function
        P, pcov = curve_fit(exp_f, t_shift, Tv_trace_sum)
        Tv_typical = P[0]
    else:
        print("not enough subthreshold time")
        Tv_typical = 1e-9
        nn = int(autocorrel_window/dt)
        t_shift, Tv_trace_sum = np.arange(nn)*dt, np.ones(nn)*.5
        
    return muV_sum, sV_sum, Tv_typical/Tm0
