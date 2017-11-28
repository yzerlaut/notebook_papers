"""
Loads the parameters of the stimulus and experiment !!
"""

import numpy as np

default_params = {\
                  'sX':1.5, # extension of the stimulus (gaussian in space)
                  'X1':-1, # center of first stim
                  'X2':1., # center of second stim
                  'dt':5e-4,
                  'BIN':5e-3, # for markovian formalism
                  'tstop':400e-3,
                  'tstart':150e-3,
                  'amp':15.,
                  'Tau1':50e-3,
                  'Tau2':150e-3}

am_params = {
    'stimuli_shift':8.1, # 2deg by default
    'delay':50e-3
}

def heaviside(x):
    return 0.5*(1+np.sign(x))

def triple_gaussian(t, X, t0, T1, T2, X0, sX, amplitude):
    return amplitude*(\
                      np.exp(-(t-t0)**2/2./T1**2)*heaviside(-(t-t0))+\
                      np.exp(-(t-t0)**2/2./T2**2)*heaviside(t-t0))*\
                      np.exp(-(X-X0)**2/2./sX**2)
                      # heaviside(X-X0+sX/2.)*heaviside(X0+sX/2.-X)

def quadruple_gaussian(t, X, t0, T1, T2, X1, X2, sX, amplitude):
    space_dep = heaviside(X1-X)*np.exp(-(X-X1)**2/2/sX**2)+\
                heaviside(X2-X)*heaviside(X-X1)+\
                heaviside(X-X2)*np.exp(-(X-X2)**2/2/sX**2)
    temp_dep = np.exp(-(t-t0)**2/2./T1**2)*heaviside(-(t-t0))+\
               np.exp(-(t-t0)**2/2./T2**2)*heaviside(t-t0)
    return amplitude*space_dep*temp_dep

                      
def get_stimulation(X, MODEL, return_print=False, custom={}):

    BASE = MODEL.split('-')[0]
    if len(MODEL.split('-'))==2:
        ARG1, ARG2 = MODEL.split('-')[1], ''
    elif len(MODEL.split('-'))==3:
        ARG1, ARG2 = MODEL.split('-')[1], MODEL.split('-')[2]
    else:
        ARG1, ARG2 = '', ''
        
    params = default_params
    for key, val in custom.items():
        params[key] = val

    if type(MODEL)==dict:
        print('Stimulation not taken from library, manually set')
        params = MODEL
        X0 = X[int(len(X)/2.)]
        t = np.arange(int((params['tstop'])/params['dt']))*params['dt'] # time array
        X1, t1 = np.meshgrid(X, t)
        nu_e_aff = quadruple_gaussian(\
                                      t1, X1, params['tstart'],\
                                      params['Tau1'], params['Tau2'],\
                                      X0+params['dX1'], X0+params['dX2'],
                                      params['sX'], params['amp'])
    else:

        if BASE=='CENTER':
            X0 = X[int(len(X)/2.)]
            # check if stimulation temporal boundaris is ok, making it longer if stim not fully develloped
            params['tstart'] = np.max([3.*params['Tau1'], params['tstart']])
            params['tstop'] = np.max([params['tstart']+3.*params['Tau2'], params['tstop']])
            t = np.arange(int((params['tstop'])/params['dt']))*params['dt'] # time array
            X1, t1 = np.meshgrid(X, t)
            nu_e_aff = triple_gaussian(\
                                       t1, X1, params['tstart'],\
                                       params['Tau1'], params['Tau2'],\
                                       X0, params['sX'], params['amp'])

        elif BASE=='FIRST_STIM':

            if ARG1=='1deg':
                stimuli_shift = am_params['stimuli_shift']/2.
            else:
                stimuli_shift = am_params['stimuli_shift']

            X0 = X[int(len(X)/2.)]-stimuli_shift/2.
            t = np.arange(int((params['tstop'])/params['dt']))*params['dt'] # time array
            X1, t1 = np.meshgrid(X, t)
            nu_e_aff = triple_gaussian(\
                                       t1, X1, params['tstart'],\
                                       params['Tau1'], params['Tau2'],\
                                       X0, params['sX'], params['amp'])

        elif BASE=='SECOND_STIM':

            if ARG1=='1deg':
                stimuli_shift = am_params['stimuli_shift']/2.
            else:
                stimuli_shift = am_params['stimuli_shift']

            X0 = X[int(len(X)/2.)]+stimuli_shift/2.
            t = np.arange(int((params['tstop'])/params['dt']))*params['dt'] # time array
            X1, t1 = np.meshgrid(X, t)
            nu_e_aff = triple_gaussian(\
                                       t1, X1, params['tstart']+am_params['delay'],\
                                       params['Tau1'], params['Tau2'],\
                                       X0, params['sX'], params['amp'])

        elif BASE=='AM':

            if ARG1=='1deg':
                stimuli_shift = am_params['stimuli_shift']/2.
            else:
                stimuli_shift = am_params['stimuli_shift']

            # first stimulus
            X0 = X[int(len(X)/2.)]-stimuli_shift/2.
            t = np.arange(int((params['tstop'])/params['dt']))*params['dt'] # time array
            X1, t1 = np.meshgrid(X, t)
            nu_e_aff1 = triple_gaussian(\
                                       t1, X1, params['tstart'],\
                                       params['Tau1'], params['Tau2'],\
                                       X0, params['sX'], params['amp'])
            # second stimulus
            X0 = X[int(len(X)/2.)]+stimuli_shift/2.
            t = np.arange(int((params['tstop'])/params['dt']))*params['dt'] # time array
            X1, t1 = np.meshgrid(X, t)
            nu_e_aff2 = triple_gaussian(\
                                       t1, X1, params['tstart']+am_params['delay'],\
                                       params['Tau1'], params['Tau2'],\
                                       X0, params['sX'], params['amp'])
            nu_e_aff = nu_e_aff1+nu_e_aff2
        
    if return_print:
        return params
    else:
        return t, nu_e_aff


all_models = ['CENTER']

import pprint                   
if __name__=='__main__':
    for m in all_models:
        p = get_stimulation(MODEL, return_print=True)                
        print("==============================================")
        print("===----", p['name'], "-----===========")
        print("==============================================")
        pprint.pprint(p)

