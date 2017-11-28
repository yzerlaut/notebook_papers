import numpy as np
import sys
sys.path.append('../')
from transfer_functions.load_config import load_transfer_functions
from scipy.integrate import odeint


def build_up_differential_operator_first_order(TF1, TF2, T=5e-3):
    """
    simple first order system
    """
    def A0(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return 1./T*(TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)-V[0])
    
    def A1(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return 1./T*(TF2(V[0]+exc_aff, V[1]+inh_aff)-V[1])
    
    def Diff_OP(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return np.array([A0(V, exc_aff=exc_aff,inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A1(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])
    return Diff_OP
    

def build_up_differential_operator(TF1, TF2,\
                                   Ne=8000, Ni=2000, T=5e-3):
    """
    Implements Equation (3.16) in El BOustani & Destexhe 2009
    in the case of a network of two populations:
    one excitatory and one inhibitory
    
    Each neuronal population has the same transfer function
    this 2 order formalism computes the impact of finite size effects
    T : is the bin for the Markovian formalism to apply

    the time dependent vector vector is V=[fe,fi, sfe, sfi, sfefi]
    the function returns Diff_OP
    and d(V)/dt = Diff_OP(V)
    """
    
    # we have the transfer function, now we also get its derivatives
    # TF, diff_fe, diff_fi, diff2_fe_fe, diff2_fe_fi, diff2_fi_fi, values = \
    #                         get_derivatives_of_TF(params)
    
    def A0(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return 1./T*(\
                .5*V[2]*diff2_fe_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                .5*V[3]*diff2_fe_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                .5*V[3]*diff2_fi_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                .5*V[4]*diff2_fi_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)-V[0])
    
    def A1(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return 1./T*(\
                .5*V[2]*diff2_fe_fe(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                .5*V[3]*diff2_fe_fi(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                .5*V[3]*diff2_fi_fe(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                .5*V[4]*diff2_fi_fi(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                TF2(V[0]+exc_aff, V[1]+inh_aff)-V[1])
    
    def A2(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return 1./T*(\
                1./Ne*TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)*(1./T-TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff))+\
                (TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)-V[0])**2+\
                2.*V[2]*diff_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                2.*V[3]*diff_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                -2.*V[2])
    
    def A3(V, exc_aff=0, inh_aff=0, pure_exc_aff=0): # mu, nu = e,i, then lbd = e then i
        return 1./T*(\
               (TF1(V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)-V[0])*(TF2(V[0]+exc_aff, V[1]+inh_aff)-V[1])+\
                V[2]*diff_fe(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                V[3]*diff_fe(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                V[3]*diff_fi(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                V[4]*diff_fi(TF1, V[0]+exc_aff+pure_exc_aff, V[1]+inh_aff)+\
                -2.*V[3])
    
    def A4(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return 1./T*(\
                1./Ni*TF2(V[0]+exc_aff, V[1]+inh_aff)*(1./T-TF2(V[0]+exc_aff, V[1]+inh_aff))+\
                (TF2(V[0]+exc_aff, V[1]+inh_aff)-V[1])**2+\
                2.*V[3]*diff_fe(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                2.*V[4]*diff_fi(TF2, V[0]+exc_aff, V[1]+inh_aff)+\
                -2.*V[4])
    
    def Diff_OP(V, exc_aff=0, inh_aff=0, pure_exc_aff=0):
        return np.array([A0(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A1(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A2(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A3(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff),\
                         A4(V, exc_aff=exc_aff, inh_aff=inh_aff, pure_exc_aff=pure_exc_aff)])
    return Diff_OP

##### Derivatives taken numerically,
## to be implemented analitically ! not hard...

def diff_fe(TF, fe, fi, df=1e-4):
    return (TF(fe+df/2., fi)-TF(fe-df/2.,fi))/df

def diff_fi(TF, fe, fi, df=1e-4):
    return (TF(fe, fi+df/2.)-TF(fe, fi-df/2.))/df

def diff2_fe_fe(TF, fe, fi, df=1e-4):
    return (diff_fe(TF, fe+df/2., fi)-diff_fe(TF,fe-df/2.,fi))/df

def diff2_fi_fe(TF, fe, fi, df=1e-4):
    return (diff_fi(TF, fe+df/2., fi)-diff_fi(TF,fe-df/2.,fi))/df

def diff2_fe_fi(TF, fe, fi, df=1e-4):
    return (diff_fe(TF, fe, fi+df/2.)-diff_fe(TF,fe, fi-df/2.))/df

def diff2_fi_fi(TF, fe, fi, df=1e-4):
    return (diff_fi(TF, fe, fi+df/2.)-diff_fi(TF,fe, fi-df/2.))/df

def find_fixed_point_first_order(NRN1, NRN2, NTWK,\
                                 Ne=8000, Ni=2000, exc_aff=0.,\
                                 verbose=False):

    TF1, TF2 = load_transfer_functions(NRN1, NRN2, NTWK)
    
    t = np.arange(2000)*1e-4              # time
    
    ### FIRST ORDER ###
    def dX_dt_scalar(X, t=0):
        return build_up_differential_operator_first_order(TF1, TF2, T=5e-3)(X, exc_aff=exc_aff)
    X0 = [1, 10] # need inhibition stronger than excitation
    X = odeint(dX_dt_scalar, X0, t)         # we don't need infodict here
    if verbose:
        print('first order prediction: ', X[-1])
    return X[-1][0], X[-1][1] 

def find_fixed_point(NRN1, NRN2, NTWK, Ne=8000, Ni=2000, exc_aff=0., verbose=False):

    # we start from the first order prediction !!!
    X0 = find_fixed_point_first_order(NRN1, NRN2, NTWK,\
                                      Ne=Ne, Ni=Ni, exc_aff=exc_aff,\
                                      verbose=verbose)
    X0 = [X0[0], X0[1], .5, .5, .5]
    
    TF1, TF2 = load_transfer_functions(NRN1, NRN2, NTWK)
    t = np.arange(200)*1e-4              # time

    ### SECOND ORDER ###
    # def dX_dt_scalar(X, t=0):
    #     return build_up_differential_operator(TF1, TF2,\
    #                                           Ne=Ne, Ni=Ni)(X, exc_aff=exc_aff)
    # X = odeint(dX_dt_scalar, X0, t)         # we don't need infodict here

    # simple euler
    X = X0
    for i in range(len(t)-1):
        X = X + (t[1]-t[0])*build_up_differential_operator(TF1, TF2,Ne=Ne, Ni=Ni)(X, exc_aff=exc_aff)
        last_X = X

    print('Make sure that those two values are similar !!')
    print(X)
    print(last_X)
    
    if verbose:
        print(X)
    if verbose:
        print('first order prediction: ',X[-1])
    
    # return X[-1][0], X[-1][1], np.sqrt(X[-1][2]), np.sqrt(X[-1][3]), np.sqrt(X[-1][4])
    return X[0], X[1], np.sqrt(X[2]), np.sqrt(X[3]), np.sqrt(X[4])

if __name__=='__main__':

    # find_fixed_point('LIF', 'LIF', 'Vogels-Abbott', exc_aff=0., Ne=4000, Ni=1000, verbose=True)
    find_fixed_point('RS-cell', 'FS-cell', 'CONFIG1', exc_aff=4., Ne=8000, Ni=2000, verbose=True)
    

    
