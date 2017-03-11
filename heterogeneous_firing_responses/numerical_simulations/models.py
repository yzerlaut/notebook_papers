"""
Loads the model with constants as previously published
if it already has the key (e.g. 'Gl'), it does not change it,
else it gives the default value
"""
import numpy as np

default_keys = ['El', 'Gl', 'Cm', 'vthresh', 'vpeak', 'vreset',\
                'Driving_Force', 'vspike', 'a', 'b', 'tauw',
                'trefrac', 'delta_v', 'Ai', 'Vi', 'Ti', 'Tm_var']
default_values = [-70e-3, 2.5e-9, 80e-12, -47e-3, 20e-3, -70e-3, 30e-3,\
                  -0e-3, 0e-9, 0e-12, 500e-3, 5e-3, 0e-3, 0., -54e-3,\
                  5e-3, 12e-3]


def get_model_params(MODEL, params):

    ## we start with the passive parameters, by default
    ## they will be overwritten by some specific models
    ## (e.g. Adexp-Rs, Wang-Buszaki) and not by others (IAF, EIF)
    for i in range(len(default_keys)):
        if default_keys[i] not in params:
            params[default_keys[i]]=default_values[i]

    ### VARYING THE MEMBRANE TIME CONSTANTS
    if MODEL.split('__')[-1]=='minus':
        params['Cm'] *= (1-params['Tm_var']*params['Gl']/params['Cm'])
    if MODEL.split('__')[-1]=='plus':
        params['Cm'] *= (1+params['Tm_var']*params['Gl']/params['Cm'])
    MODEL2 = MODEL.split('_')[0]
    
    # values by default
    params['RANGE_FOR_3D'] = [-66e-3, -54e-3, 4e-3, 9e-3, 1e-3, 5e-3, .15]

    """ ======== SUBTHRESHOLD RC CIRCUIT ============ """
    if MODEL2=='SUBTHRE':
        # no spiking dynamics
        params['vspike'], params['vthresh'], params['delta_v'] =100, 100, 0
        params['Ai'], params['a'], params['b'] = 0, 0, 0 # no non linear mech
        params['name'] = 'RC-circuit'

        """ ======== INTEGRATE AND FIRE ============ """
    elif MODEL2.split('-')[0]=='LIF':

        params['RANGE_FOR_3D'] = [params['vthresh']-18e-3, params['vthresh']-6e-3, 4e-3, 9e-3, 1e-3, 5e-3, .15]
        # params by default are already LIF
        params['name'] = 'LIF'
        params['Ai'], params['a'], params['b'] = 0, 0, 0 # no non linear mech
        params['Istep'] = 100e-12

        if MODEL2.split('-')[-1]!='LIF': # means other-argument

            ### --- VARYING THE THRESHOLD
            if MODEL2.split('-')[1]=='Vthre': # keyword for varying threshold
                ex_vthre = 1e3*params['vthresh'] # in mV
                vthre = float(MODEL2.split('-')[2]) # value is the last arg
                params['vthresh'] = -1e-3*vthre
                params['name'] = 'LIF $V_\mathrm{thre}$=-'+\
                                 MODEL2.split('-')[2]+'mV'
                # shift because change of excitability
                params['RANGE_FOR_3D'][0] -= (vthre+ex_vthre)*1e-3
                params['RANGE_FOR_3D'][1] -= (vthre+ex_vthre)*1e-3
                

            ### --- VARYING THE REFRACTORY PERIOD
            elif MODEL2.split('-')[1]=='Tref': # keyword for varying threshold
                Tref = float(MODEL2.split('-')[2]) # value is the last arg
                params['trefrac'] = 1e-3*Tref
                params['name'] = 'LIF $\\tau_{ref}$='+\
                                 MODEL2.split('-')[2]+'ms'


        """ ======== EXPONENTIAL INTEGRATE AND FIRE ============ """
    elif MODEL2.split('-')[0]=='EIF':
        
        # default values for EIF
        # params['vthresh'] = -47e-3
        params['delta_v'] = 2e-3
        params['name'] = 'EIF'
        params['Ai'], params['a'], params['b'] = 0, 0, 0 # no non linear mech
        # increased range
        params['RANGE_FOR_3D'] = [params['vthresh']-18e-3, params['vthresh']-4e-3, 4e-3, 9e-3, 1e-3, 6e-3, .15]
        params['Istep'] = 100e-12

        if MODEL2.split('-')[-1]!='EIF': # means other-argument

            ### --- VARYING THE ACTIVATION SHARPNESS
            if MODEL2.split('-')[1]=='ka': # keyword for varying threshold
                dv = float(MODEL2.split('-')[2]) # value is the last arg
                params['delta_v'] = 1e-3*dv
                params['name'] = 'EIF $k_a$='+\
                                 MODEL2.split('-')[2]+'mV'
                  
            ### --- VARYING THE THRESHOLD
            elif MODEL2.split('-')[1]=='Vthre': # keyword for varying threshold
                vthre = float(MODEL2.split('-')[2]) # value is the last arg
                params['vthresh'] = -1e-3*vthre
                params['name'] = 'EIF $V_\mathrm{thre}$=-'+\
                                 MODEL2.split('-')[2]+'mV'


    elif MODEL2.split('-')[0]=='sfaLIF': # default for all Adexp models

        # params['vthresh'] = -50e-3
        # by default AdExp, Regular Spiking chractristics !!
        params['b'] = 20e-12          # (pA->A)   : increment of adaptation
        params['delta_v'] = 0e-3           # (mV)   : steepness of exponential approach to threshold
        params['a'], params['Ai'] = 0, 0 # no other non-linear mech
        params['name']='sfa-LIF'
        params['Istep'] = 100e-12
        
        # needs more variance at depolarized levels
        params['RANGE_FOR_3D'] = [params['vthresh']-18e-3+\
                                  5e-4*params['b']/10e-12,\
                                  params['vthresh']-5e-3+\
                                  5e-4*params['b']/10e-12,\
                                  4e-3, 9e-3, 1e-3, 5.5e-3, .15]


        # then we can change some values depending on further options
        if MODEL2.split('-')[-1]!='sfaLIF': # means other-argument

            ### --- VARYING THE SPIKE FREQUENCY ADAPTATION
            if MODEL2.split('-')[1]=='b': # keyword for this strength
                b = float(MODEL2.split('-')[2]) # value is the last arg
                params['b'] = b*1e-12 #  pA
                params['name'] = 'sfa-LIF $b$='+MODEL2.split('-')[2]+'pA'
                params['RANGE_FOR_3D'] = [params['vthresh']-18e-3+1e-3*params['b']/40e-12,\
                                          params['vthresh']-4e-3+1e-3*params['b']/40e-12,\
                                          4e-3, 9e-3, 1e-3, 6e-3, .15]

    elif MODEL2.split('-')[0]=='sbtaLIF': # default for all Adexp models

        # params['vthresh'] = -45e-3
        # by default AdExp, Regular Spiking chractristics !!
        params['a'] = 3e-9 #  pA
        params['delta_v'] = 0e-3           # (mV)   : steepness of exponential approach to threshold
        params['b'], params['Ai'] = 0, 0 # no other non-linear mech
        params['name']='sbta-LIF'
        params['Istep'] = 100e-12
        
        # needs more variance at depolarized levels
        params['RANGE_FOR_3D'] = [params['vthresh']-18e-3,\
                params['vthresh']-6e-3, 4e-3, 9e-3, 1e-3, 5e-3, .15]


        # then we can change some values depending on further options
        if MODEL2.split('-')[-1]!='sbtaLIF': # means other-argument

            ### --- VARYING THE SPIKE FREQUENCY ADAPTATION
            if MODEL2.split('-')[1]=='a': # keyword for this strength
                a = float(MODEL2.split('-')[2]) # value is the last arg
                params['a'] = a*1e-9 #  pA
                params['name'] = 'sbta-LIF $a$='+MODEL2.split('-')[2]+'nS'
                params['RANGE_FOR_3D'] = [params['vthresh']-18e-3,\
                        params['vthresh']-4e-3+3e-3*a/10., 4e-3, 9e-3, 1e-3, 6e-3, .15]

        """ ======== INTEGRATE AND FIRE with voltage dependent inactivation ============ """
    elif MODEL2.split('-')[0]=='iLIF':

        # other spiking dynamics params, set within main file
        # params['vthresh'] = -49e-3
        params['Vi'] = params['vthresh']-8e-3
        params['Ai'] = 0.6 # actually ki/ka, 
        params['Ti'] = 5e-3
        params['trefrac'] = 5e-3
        params['name'] = 'iLIF'
        params['a'], params['b'] = 0, 0 # no other non linear mech
        params['Istep'] = 100e-12
        
        # needs more variance at depolarized levels
        params['RANGE_FOR_3D'] = [params['vthresh']-16e-3, params['vthresh']-2e-3, 4e-3, 9e-3, 1e-3, 6e-3, .15]

        if MODEL2.split('-')[-1]!='iLIF': # means other-argument

            ### --- VARYING THE STREGTH OF INACTIVATION
            if MODEL2.split('-')[1]=='Ai': # keyword for this strength
                ai = float(MODEL2.split('-')[2]) # value is the last arg
                params['Ai'] = ai # actually ki/ka
                params['name'] = 'iLIF $a_i$='+MODEL2.split('-')[2]+ ' '
                params['RANGE_FOR_3D'] = [params['vthresh']-18e-3,\
                  params['vthresh']-4e-3+3e-3*ai, 4e-3, 9e-3, 1e-3, 6e-3, .15]

            ### --- VARYING THE THRESHOLD
            if MODEL2.split('-')[1]=='Vthre': # keyword for varying threshold
                vthre = float(MODEL2.split('-')[2]) # value is the last arg
                params['vthresh'] = -1e-3*vthre
                params['name'] = 'iLIF $V_\mathrm{thre}$=-'+\
                                 MODEL2.split('-')[2]+'mV'

        """ ======== AdExp models ============ """
    elif MODEL2.split('-')[0]=='AdExp': # default for all Adexp models

        # by default AdExp, Regular Spiking chractristics !!
        params['b'] = 20e-12          # (pA->A)   : increment of adaptation
        params['a'] = 4e-9 #  pA
        params['delta_v'] = 2e-3           # (mV)   : steepness of exponential approach to threshold
        params['name']='AdExp-RS'
        params['Ai'] = 0
        # params['vthresh']=-48.178614766e-3
        params['RANGE_FOR_3D'] = [-65e-3, -53e-3, 4e-3, 9e-3, 1e-3, 6e-3, .15]

                
        # then we can change some values depending on further options
        if MODEL2.split('-')[-1]!='AdExp': # means other-argument

            ### --- VARYING BOTH THE SPIKE SHARPNESS AND SPIKE FREQUENCY ADAPTATION
            if (MODEL2.split('-')[1]=='comod'):
                n = 10 # needs to correspond to the one in get_models
                B = np.linspace(0., 40., n)
                Ka = np.linspace(0., 4., n)
                A = np.ones(n)*4. # np.linspace(0, 7., n) # CST a !!!!
                i = int(MODEL2.split('-')[2])
                ka, b, a = Ka[i], B[i], A[i]
                params['delta_v'] = 1e-3*ka
                params['b'] = b*1e-12 #  pA
                params['a'] = a*1e-9 #  nS
                params['name'] = 'AdExp $k_a$='+str(ka)+'mV, $b$='+str(b)+'pA, $a$='+str(a)+'nS'
                params['RANGE_FOR_3D'] = [-62e-3, -55e-3+4e-3*i/(n-1),\
                                          4e-3, 9e-3, 1e-3, 7e-3, .15]

            ### --- VARYING THE SPIKE SHARPNESS
            elif MODEL2.split('-')[1]=='ka': # keyword for varying threshold
                ka = float(MODEL2.split('-')[2]) # value is the last arg
                params['delta_v'] = 1e-3*ka
                params['name'] = 'AdExp $V_\mathrm{thre}$=-'+\
                                 MODEL2.split('-')[2]+'mV'

            ### --- VARYING THE SPIKE FREQUENCY ADAPTATION
            elif MODEL2.split('-')[1]=='b': # keyword for this strength
                b = float(MODEL2.split('-')[2]) # value is the last arg
                params['b'] = b*1e-12 #  pA
                params['name'] = 'AdExp $b$='+MODEL2.split('-')[2]+'pA'

            ### --- VARYING SUBTHRESHOLD ADAPTATION
            elif MODEL2.split('-')[1]=='a': # keyword for this strength
                a = float(MODEL2.split('-')[2]) # value is the last arg
                params['a'] = a*1e-9 #  pA
                params['name'] = 'AdExp $a$='+MODEL2.split('-')[2]+'nS'

            ### --- VARYING THE THRESHOLD
            if MODEL2.split('-')[1]=='Vthre': # keyword for varying threshold
                vthre = float(MODEL2.split('-')[2]) # value is the last arg
                params['vthresh'] = -1e-3*vthre
                params['name'] = 'AdExp $V_\mathrm{thre}$=-'+\
                                 MODEL2.split('-')[2]+'mV'

        elif MODEL2.split('-')[-1]=='LTS': # model of LOW THRESHOLD SPIKE neuron
            params['a'] = 0.02e-6       #  (uS->S)   : level of subthreshold adaptation
            params['name']='AdExp-LTS'
        
        """ ======== iAdExp models ============ """
    elif MODEL2.split('-')[0]=='iAdExp': 

        # by default AdExp, Regular Spiking chractristics !!

        # params['vthresh'] = -49e-3
        params['Vi'] = params['vthresh']-8e-3
        params['Ai'] = 0.6 # actually ki/ka, 
        params['Ti'] = 5e-3
        params['trefrac'] = 5e-3
        params['name'] = 'iAdExp'
        params['b'] = 6e-12          # (pA->A)   : increment of adaptation
        params['delta_v'] = 1e-3          # (pA->A)   : increment of adaptation
        params['a'] = 0e-9 #  pA
        params['Istep'] = 100e-12
        
        params['RANGE_FOR_3D'] = [params['vthresh']+\
                                  .5*params['delta_v']+params['Ai']*.5e-3\
                                  -15e-3+5e-4*params['b']/10e-12,\
                                  params['vthresh']+.5*params['delta_v']+\
                                  params['Ai']*.5e-3-4e-3+\
                                  +5e-4*params['b']/10e-12,\
                                  3e-3, 8e-3, 1e-3, 6e-3, .15]
                
        # to then adapt the domain
        ex_vthre = params['vthresh'] # in mV

        # then we can change some values depending on further options
        if MODEL2.split('-')[-1]!='iAdExp': # means other-argument

            PARAMS = ['Vthre', 'Ka', 'Ai', 'A', 'B']
            VALUE = ['vthresh', 'delta_v', 'Ai', 'a', 'b']
            UNITS = [-1e-3, 1e-3, 1., 1e-9, 1e-12]
            for pp, vv, uu in zip(PARAMS, VALUE, UNITS):
                if len(MODEL2.split(pp+'-'))>1:
                    params[vv] = uu*float(MODEL2.split(pp+'-')[1].split('-')[0])
            params['Vi'] = params['vthresh']-8e-3 # need to readjst the shift

            # shift of the domain (because change of excitability) to remain in 1-20Hz
            params['RANGE_FOR_3D'] = [params['vthresh']+\
                                      .5*params['delta_v']+params['Ai']*.5e-3\
                                      -15e-3+5e-4*params['b']/10e-12,\
                                      params['vthresh']+.5*params['delta_v']+\
                                      params['Ai']*.5e-3-4e-3+\
                                      +5e-4*params['b']/10e-12,\
                                      3e-3, 8e-3, 1e-3, 6e-3, .15]
                
        
    else:
        params = {}
        print('-------------------------------------------------------------')
        print('==========> ||neuron model not recognized|| <================')
        print('-------------------------------------------------------------')

    return params


def models(name, n=10, get_interpretation=False, get_legend=False,\
           get_input_space=False, get_input_space2=False, get_title=False):

    if name=='typical_models':
        return ['LIF', 'EIF', 'sfaLIF', 'sbtaLIF', 'iLIF', 'iAdExp']

    if name=='LIF_Vthre_models':
        # varying thresholds for LIF :
        params = get_model_params('LIF', {})
        vthre = np.linspace(59, 35, n)
        if get_interpretation:
            return 'higher threshold'
        elif get_title:
            return r'LIF varying $V_\mathrm{thre}$'
        elif get_legend:
            return r'$V_\mathrm{thre}$ (mV)'
        elif get_input_space:
            return -vthre
        else:
            return ['LIF-Vthre-'+str(round(vv,2)) for vv in vthre]

    elif name=='LIF_Tref_models':
        # varying refractory period for LIF :
        tref = np.linspace(1e-4, 10, n)
        if get_interpretation:
            return 'longer refractory period'
        elif get_title:
            return r'LIF varying $\tau_\mathrm{ref}$'
        elif get_legend:
            return r'$\tau_\mathrm{ref}$ (ms)'
        else:
            return ['LIF-Tref-'+str(round(tt,1)) for tt in tref]

    elif name=='EIF_ka_models':
        # varying sharpness for EIF :
        dv = np.linspace(0., 3.5, n)
        if get_interpretation:
            return 'smoother Na activation'
        elif get_title:
            return r'EIF varying $k_\mathrm{a}$'
        elif get_legend:
            return r'$k_\mathrm{a}$ (mV)'
        elif get_input_space:
            return dv
        else:
            return ['EIF-ka-'+str(round(vv,1)) for vv in dv]

    elif name=='aEIF_Xa_models':
        # varying sharpness for EIF :
        XA = np.arange(1,n+1)*10
        if get_interpretation:
            return 'further initiation'
        else:
            return ['aEIF-Xa-'+str(round(vv,1)) for vv in XA]

    elif name=='sfaLIF_b_models':
        # varying spike frequency adaptation for AdExp :
        B = np.linspace(0., 35., n)
        if get_interpretation:
            return 'stronger spike freq. adaptation'
        elif get_title:
            return r'sfaLIF varying $b$'
        elif get_input_space:
            return B
        elif get_legend:
            return r'$b$ (pA)'
        else:
            return ['sfaLIF-b-'+str(round(vv,1)) for vv in B]

    elif name=='iLIF_Ai_models':
        # varying the strength of spike frequency adaptation for AdExp :
        ai = np.linspace(0., 0.58, n) # actually ki / ka
        if get_interpretation:
            return 'stronger inactivation'
        elif get_title:
            return r'iLIF varying $a_i$'
        elif get_input_space:
            return ai
        elif get_legend:
            return r'$a_\mathrm{i}$'
        else:
            return ['iLIF-Ai-'+str(round(vv,1)) for vv in ai]

    elif name=='sbtaLIF_a_models':
        # varying spike frequency adaptation for AdExp :
        A = np.linspace(0., 7., n)
        if get_interpretation:
            return 'stronger subthresh. adaptation'
        elif get_legend:
            return r'$a$ (nS)'
        else:
            return ['sbtaLIF-a-'+str(round(vv,1)) for vv in A]

    elif name=='iAdExp_PC1_models':
        params = get_model_params('iAdExp', {})
        Vthre = np.concatenate([[59.], np.linspace(59., 47., n-1)])
        Ka = np.linspace(0., 3.5, n)
        Ai = np.linspace(.58, 0., n) # actually ki / ka
        B = np.linspace(0, 25., n)
        if get_interpretation:
            import matplotlib.pylab as plt
            import sys
            sys.path.append('/home/yann/work/python_library/')
            from my_graph import set_plot
            fig, AXX = plt.subplots(4, figsize=(5,5))
            for axx, X, ylabel in zip(AXX, [-Vthre, Ka, Ai, B],\
                                 [r'$V_\mathrm{thre}$(mV)', r'$k_a$(mV)',\
                                  r'$a_i$', r'$b$(pA)']):
                axx.plot(list(range(len(X))), X, 'kD-', ms=5)
                axx.plot([len(X)-1], X[-1], 'bD', [0], X[0], 'rD', ms=6)
                set_plot(axx, ylabel=ylabel, xticks=[], num_yticks=3,\
                         xlim_enhancment=7, ylim_enhancment=7)
                
            return fig
        elif get_legend:
            return 'params comodulation'
        elif get_title:
            return r'iAdExp'
        elif get_input_space:
            return np.linspace(0, 1, n)
        elif get_input_space2:
            return -Vthre, Ka, Ai, B
        else:
            return ['iAdExp-Vthre-'+str(round(Vthre[i],2))+\
                '-Ka-'+str(round(Ka[i],2))+'-Ai-'+str(round(Ai[i],2))+\
                '-B-'+str(round(B[i],2)) for i in range(len(Vthre))]
                
    elif name=='iAdExp_5D_models':
        params = get_model_params('iAdExp', {})
        n = 7 # more is impossible !!
        Vthre = np.linspace(-2,12,n)-1e3*params['vthresh']
        Ka = np.linspace(0., 3.7, n)
        Ai = np.linspace(0., 0.7, n) # actually ki / ka
        B = np.linspace(0., 40., n)
        A = np.linspace(0., 7., n)
        Vthre, Ka, Ai, B, A = np.meshgrid(Vthre, Ka, Ai, B, A)
        Vthre, Ka, Ai, B, A = Vthre.flatten(), Ka.flatten(), Ai.flatten(), B.flatten(), A.flatten()
        if get_input_space:
            return Vthre, Ka, Ai, B, A
        else:
            return ['iAdExp-Vthre-'+str(Vthre[i])+\
                '-Ka-'+str(Ka[i])+'-Ai-'+str(Ai[i])+\
                '-B-'+str(B[i])+\
                '-A-'+str(A[i]) for i in range(len(Vthre))]

    elif name=='iAdExp_4D_models':
        params = get_model_params('iAdExp', {})
        n = 7 # more is impossible !!
        Vthre = np.linspace(-2,12,n)-1e3*params['vthresh']
        Ka = np.linspace(0., 3., n)
        Ai = np.linspace(0., 3., n) # actually ki / ka
        B = np.linspace(0., 40., n)
        Vthre, Ka, Ai, B = np.meshgrid(Vthre, Ka, Ai, B)
        Vthre, Ka, Ai, B = Vthre.flatten(), Ka.flatten(), Ai.flatten(), B.flatten()
        if get_input_space:
            return Vthre, Ka, Ai, B
        else:
            return ['iAdExp-Vthre-'+str(Vthre[i])+\
                '-Ka-'+str(Ka[i])+'-Ai-'+str(Ai[i])+\
                '-B-'+str(B[i]) for i in range(len(Vthre))]
    elif name=='all_models':
        return all_models
    else:
        return None

### ALL MODELS !!!
all_models = models('typical_models')+models('LIF_Vthre_models')+\
             models('EIF_ka_models')+models('sfaLIF_b_models')+\
             models('iLIF_Ai_models')+models('iAdExp_PC1_models')

import pprint
if __name__=='__main__':
    for m in all_models:
        p = get_model_params(m, {})
        print("==============================================")
        print("===--------", p['name'], "---------===========")
        print("==============================================")
        pprint.pprint(p)


