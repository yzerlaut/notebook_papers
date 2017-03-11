import numpy as np
from scipy.optimize import minimize
from scipy.optimize import curve_fit, leastsq
import scipy.special as sp_spec
import statsmodels.api as sm

## NORMALIZING COEFFICIENTS
# needs to be global here, because used both in the function
# and its derivatives
muV0, DmuV0 = -60e-3,10e-3
sV0, DsV0 =4e-3, 6e-3
TvN0, DTvN0 = 0.5, 1.

### CORRECTIONS FOR THE EFFECTIVE THRESHOLD

default_correction = {
    'V0':True,
    'muV_lin':True,
    'sV_lin':True,
    'Tv_lin':True,
    'muG_log':False,
    'muV_square':False,
    'sV_square':False,
    'Tv_square':False,
    'muV_sV_square':False,
    'muV_Tv_square':False,
    'sV_Tv_square':False
}

def final_threshold_func(coeff, muV, sV, TvN, muGn, El,\
                         correction = default_correction):
    full_coeff = np.zeros(len(correction))
    full_coeff[0] = coeff[0] # one threshold by default
    i = 1 # now, we set the others coeff in the order
    if correction['muV_lin']: full_coeff[1] = coeff[i];i+=1
    if correction['sV_lin']: full_coeff[2] = coeff[i];i+=1
    if correction['Tv_lin']: full_coeff[3] = coeff[i];i+=1
    if correction['muG_log']: full_coeff[4] = coeff[i];i+=1
    if correction['muV_square']: full_coeff[5] = coeff[i];i+=1
    if correction['sV_square']: full_coeff[6] = coeff[i];i+=1
    if correction['Tv_square']: full_coeff[7] = coeff[i];i+=1
    if correction['muV_sV_square']: full_coeff[8] = coeff[i];i+=1
    if correction['muV_Tv_square']: full_coeff[9] = coeff[i];i+=1
    if correction['sV_Tv_square']: full_coeff[10] = coeff[i];i+=1

    if not i==len(coeff):
        print('==================================================')
        print('mismatch between coeff number and correction type')
        print('==================================================')
        
    output = full_coeff[0]+\
      full_coeff[1]*(muV-muV0)/DmuV0+\
      full_coeff[2]*(sV-sV0)/DsV0+\
      full_coeff[3]*(TvN-TvN0)/DTvN0+\
      full_coeff[4]*np.log(muGn+1e-12)+\
      full_coeff[5]*((muV-muV0)/DmuV0)**2+\
      full_coeff[6]*((sV-sV0)/DsV0)**2+\
      full_coeff[7]*((TvN-TvN0)/DTvN0)**2+\
      full_coeff[8]*(muV-muV0)/DmuV0*(sV-sV0)/DsV0+\
      full_coeff[9]*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0+\
      full_coeff[10]*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0

    return output

### FUNCTION, INVERSE FUNCTION

def erfc_func(mu, sigma, TvN, Vthre, Gl, Cm):
    return .5/TvN*Gl/Cm*\
      sp_spec.erfc((Vthre-mu)/np.sqrt(2)/sigma)

def effective_Vthre(Y, mVm, sVm, TvN, Gl, Cm):
    Vthre_eff = mVm+np.sqrt(2)*sVm*sp_spec.erfcinv(\
                    Y*2.*TvN*Cm/Gl) # effective threshold
    return Vthre_eff


#######################################################
####### DERIVATIVES OF THE FIRING RATE RESPONSE

def derivative_muV(P, muV, sV, TvN, muGn, El, Gl, Cm,\
                   DmuV0=DmuV0, DsV0=DsV0, DTvN0=DTvN0):
    Vthre = final_threshold_func(P, muV, sV, TvN, muGn, El)
    factor = (1-P[1]/DmuV0)
    exp_term = np.exp(-((Vthre-muV)/np.sqrt(2)/sV)**2)
    denom = np.sqrt(2.*np.pi)*TvN*Cm/Gl*sV
    return factor*exp_term/denom

def derivative_sV(P, muV, sV, TvN, muGn, El, Gl, Cm,\
                   DmuV0=DmuV0, DsV0=DsV0, DTvN0=DTvN0):
    Vthre = final_threshold_func(P, muV, sV, TvN, muGn, El)
    factor = ((Vthre-muV)/sV-P[2]/DsV0)
    exp_term = np.exp(-((Vthre-muV)/np.sqrt(2)/sV)**2)
    denom = np.sqrt(2.*np.pi)*TvN*Cm/Gl*sV
    return factor*exp_term/denom

def derivative_Tv(P, muV, sV, TvN, muGn, El, Gl, Cm,\
                   DmuV0=DmuV0, DsV0=DsV0, DTvN0=DTvN0):

    Vthre = final_threshold_func(P, muV, sV, TvN, muGn, El)
    Fout = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    factor = P[3]/DTvN0
    exp_term = np.exp(-((Vthre-muV)/np.sqrt(2)/sV)**2)
    denom = np.sqrt(2.*np.pi)*TvN*Cm/Gl*sV
    return -(factor*exp_term/denom+Fout/TvN)

## all derivatives

def derivatives_template(P, muV, sV, TvN, muGn, El, Gl, Cm,\
                         DmuV0=DmuV0, DsV0=DsV0, DTvN0=DTvN0):
    # in common
    Vthre = final_threshold_func(P, muV, sV, TvN, muGn, El)
    exp_term = np.exp(-((Vthre-muV)/np.sqrt(2)/sV)**2)
    denom = np.sqrt(2.*np.pi)*TvN*Cm/Gl*sV
    Fout = erfc_func(muV, sV, TvN, Vthre, Gl, Cm)
    # independent
    factor1 = (1-P[1]/DmuV0)
    factor2 = ((Vthre-muV)/sV-P[2]/DsV0)
    factor3 = P[3]/DTvN0 # TvN needs additional Fout term !!
    return factor1*exp_term/denom, factor2*exp_term/denom,\
        -(factor3*exp_term/denom+Fout/TvN)

#######################################################

#######################################################
####### PRINTING COEFFICIENTS

def print_reduce_parameters(P, with_return=True, correction=default_correction):
    # first we reduce the parameters
    P = 1e3*np.array(P)
    final_string = ''
    keys = ['0', '\mu_V', '\sigma_V', '\tau_V']
    for p, key in zip(P, keys):
        final_string += '$P_{'+key+'}$ ='+str(round(p,1))+'mV, '
    if with_return:
        return final_string
    else:
        print(final_string)
    
#######################################################


#######################################################
####### THEN FITTING ALGORITHMS

def determination_of_muGn_dependency(\
            Fout, muV, sV, TvN, muGn, Gl, Cm,
            maxiter=1e9, xtol=1e-12):

    # we start by comuting the threshold
    i_non_zeros = np.nonzero(Fout)
    muV2, sV2, TvN2, muGn2, Fout2 = \
      muV[i_non_zeros], sV[i_non_zeros],\
       TvN[i_non_zeros], muGn[i_non_zeros],\
       Fout[i_non_zeros]
    
    vthre = effective_Vthre(Fout2, muV2, sV2, TvN2, Gl, Cm)

    # initial guess !!! mean and linear regressions !
    p1, p0 = np.polyfit(np.log(muGn2), vthre, 1)
    P = [p0, p1]

    def Res(p):
        to_minimize = (Fout-erfc_func(muV, sV, TvN,\
                p[0]+p[1]*np.log(muGn), Gl, Cm))**2
        return np.mean(to_minimize)
    
    plsq = minimize(Res,P, method='nelder-mead',\
            options={'xtol': xtol, 'maxiter':maxiter})
    P = plsq.x
    print(plsq)

    return P


def determination_of_TvN_dependency(\
            Fout, muV, sV, TvN, muGn, Gl, Cm,
            maxiter=1e9, xtol=1e-12):

    # we start by comuting the threshold
    i_non_zeros = np.nonzero(Fout)
    muV2, sV2, TvN2, muGn2, Fout2 = \
      muV[i_non_zeros], sV[i_non_zeros],\
       TvN[i_non_zeros], muGn[i_non_zeros],\
       Fout[i_non_zeros]
    
    vthre = effective_Vthre(Fout2, muV2, sV2, TvN2, Gl, Cm)

    # initial guess !!! mean and linear regressions !
    p1, p0 = np.polyfit((TvN2-TvN0)/DTvN0, vthre, 1)
    print(1e3*p1, 1e3*p0)
    P = [p0, p1]

    def Res(p):
        to_minimize = (Fout-erfc_func(muV, sV, TvN,\
                p[0]+p[1]*(TvN-TvN0)/DTvN0, Gl, Cm))**2
        return np.mean(to_minimize)
    
    plsq = minimize(Res,P, method='nelder-mead',\
            options={'xtol': xtol, 'maxiter':maxiter})
    P = plsq.x
    print(plsq)

    return P

def determination_of_muGn_TvN_dependency(\
            Fout, muV, sV, TvN, muGn, Gl, Cm,
            maxiter=1e9, xtol=1e-12):

    # we start by comuting the threshold
    i_non_zeros = np.nonzero(Fout)
    muV2, sV2, TvN2, muGn2, Fout2 = \
      muV[i_non_zeros], sV[i_non_zeros],\
       TvN[i_non_zeros], muGn[i_non_zeros],\
       Fout[i_non_zeros]
    
    vthre = effective_Vthre(Fout2, muV2, sV2, TvN2, Gl, Cm)
    
    # initial guess !!! mean and linear regressions !
    p1, p0 = np.polyfit((TvN2-TvN0)/DTvN0, vthre, 1)
    P = [p0, p1]
    
    def Res(p):
        to_minimize = (Fout-erfc_func(muV, sV, TvN,\
                p[0]+p[1]*(TvN-TvN0)/DTvN0, Gl, Cm))**2
        return np.mean(to_minimize)
    
    plsq = minimize(Res,P, method='nelder-mead',\
            options={'xtol': xtol, 'maxiter':maxiter})
    P = plsq.x
    print(plsq)
    return P


def determination_of_muV_sV_dependency(\
            Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
            maxiter=1e9, xtol=1e-12):


    # we start by comuting the threshold
    i_non_zeros = np.nonzero(Fout)
    muV2, sV2, TvN2, muGn2, Fout2 = \
      muV[i_non_zeros], sV[i_non_zeros],\
       TvN[i_non_zeros], muGn[i_non_zeros],\
       Fout[i_non_zeros]
    
    vthre = effective_Vthre(Fout2, muV2, sV2, TvN2, Gl, Cm)
    
    # initial guess !!! mean and linear regressions !
    P = [vthre.mean(),\
         np.polyfit(muV2, vthre, 1)[0], np.polyfit(sV2, vthre, 1)[0]]

    def Res(p):
        threshold2 = first_order_pol(p, muV, sV)+\
          p[1]*(muV-muV0)/DmuV0+p[2]*(sV-sV0)/DsV0
        to_minimize = (Fout-erfc_func(muV, sV, TvN,threshold2, Gl, Cm))**2
        return np.mean(to_minimize)

    plsq = minimize(Res,P, method='nelder-mead', tol=xtol, options={'maxiter':maxiter})

    P = plsq.x
    print(plsq)
    return P


def linear_fitting_of_threshold_with_firing_weight(\
            Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
            maxiter=1e5, xtol=1e-18,
            correction=default_correction,\
            print_things=True):

    # we start by comuting the threshold
    i_non_zeros = np.nonzero(Fout)
    muV2, sV2, TvN2, muGn2, Fout2 = \
      muV[i_non_zeros], sV[i_non_zeros],\
       TvN[i_non_zeros], muGn[i_non_zeros],\
       Fout[i_non_zeros]
    
    vthre = effective_Vthre(Fout2, muV2, sV2, TvN2, Gl, Cm)

    # initial guess !!! mean and linear regressions !
    i = 0
    for val in correction.values():
        if val: i+=1
    P = np.zeros(i)
    P[0] = -45e-3 # just threshold in the right range

    def Res(p):
        threshold2 = final_threshold_func(p, muV2, sV2, TvN2, muGn2, El,\
                                          correction=correction)
        to_minimize = (vthre-threshold2)**2
        return np.mean(to_minimize)/len(threshold2)

    # bnds = ((-90e-3, -10e-3), (None,None), (None,None), (None,None), (None,None),\
    #         (None,None),(-2e-3, 3e-3), (-2e-3, 3e-3))
    # plsq = minimize(Res,P, method='SLSQP', bounds=bnds, tol=xtol,\
    #         options={'maxiter':maxiter})

    plsq = minimize(Res,P, tol=xtol, options={'maxiter':maxiter})
            
    P = plsq.x
    if print_things:
        print(plsq)
    return P

def fitting_Vthre_then_Fout(Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
        maxiter=1e5, ftol=1e-15,\
        correction=default_correction, print_things=True,\
        return_chi2=False):

    P = linear_fitting_of_threshold_with_firing_weight(\
            Fout, muV, sV, TvN, muGn, Gl, Cm, El,\
            maxiter=maxiter, xtol=ftol,\
            correction=correction,\
            print_things=print_things)

    def Res(p, muV, sV, TvN, muGn, Fout):
        return (Fout-erfc_func(muV, sV, TvN,\
           final_threshold_func(p, muV, sV, TvN, muGn, El,\
                                correction=correction), Gl, Cm))
                                
    if return_chi2:
        P,cov,infodict,mesg,ier = leastsq(
            Res,P, args=(muV, sV, TvN, muGn, Fout),\
            full_output=True)
        ss_err=(infodict['fvec']**2).sum()
        ss_tot=((Fout-Fout.mean())**2).sum()
        rsquared=1-(ss_err/ss_tot)
        return P, rsquared
    else:
        P = leastsq(Res, P, args=(muV, sV, TvN, muGn, Fout))[0]
        return P
        
