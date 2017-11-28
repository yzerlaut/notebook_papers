"""
Loads the RING model parameters
"""
import numpy as np

default_params = {\
'X_discretization':30.,
'X_extent':36., # mm
'exc_connect_extent':5., # mm
'inh_connect_extent':1., # mm
'conduction_velocity_mm_s': 300. # mm/s
}

def pixels_per_mm(MODEL):
    """
    """
    params = get_model_params(MODEL)
    return params['X_discretization']/params['X_extent']
    
def mm_per_pixel(MODEL):
    """
    """
    params = get_model_params(MODEL)
    return params['X_extent']/params['X_discretization']

def from_mm_to_discretized_model(params):
    """
    translate all quantities of the ring model from mm to pixels !
    """
    params['mm_per_pixel'] = params['X_extent']/params['X_discretization']
    params['exc_decay_connect'] = params['exc_connect_extent']/params['mm_per_pixel']
    params['inh_decay_connect'] = params['inh_connect_extent']/params['mm_per_pixel']
    # in practice connectivity extends up to 3 std dev.
    params['exc_connected_neighbors'] = int(3.*params['exc_decay_connect']/params['mm_per_pixel'])
    params['inh_connected_neighbors'] = int(3.*params['inh_decay_connect']/params['mm_per_pixel'])
    params['conduction_velocity'] = params['conduction_velocity_mm_s']/params['mm_per_pixel']

    
def get_model_params(MODEL, custom={}):
    """
    we start with the passive parameters, by default
    they will be overwritten by some specific models
    (e.g. Adexp-Rs, Wang-Buszaki) and not by others (IAF, EIF)
    """

    params = default_params
    
    # overiding default params by custom params
    for key, val in custom.items():
        params[key] = val

    # """ ======== INTEGRATE AND FIRE ============ """
    if MODEL=='RING1':
        # params by default 
        params['name'] = MODEL
    elif MODEL=='RING1-hd':
        # params by default 
        params['name'] = MODEL
        params['X_discretization']=100.
    else:
        params = None
        print('==========> ||ring model not recognized|| <================')

    # now we discretize it 
    from_mm_to_discretized_model(params)
    
    return params


def gaussian_connectivity(x, x0, dx):
    return 1./(np.sqrt(2.*np.pi)*(dx+1e-12))*np.exp(-(x-x0)**2/2./(1e-12+dx)**2)


def pseq_ring_params(RING, custom={}):
    """ """
    params = get_model_params(RING, custom=custom)
    exc_connected_neighbors=params['exc_connected_neighbors']
    exc_decay_connect=params['exc_decay_connect']
    inh_connected_neighbors=params['inh_connected_neighbors']
    inh_decay_connect=params['inh_decay_connect']
    conduction_velocity=params['conduction_velocity']
    Xn_exc = np.arange(-exc_connected_neighbors, exc_connected_neighbors+1)
    Xn_inh = np.arange(-inh_connected_neighbors, inh_connected_neighbors+1)
    X = np.linspace(0, params['X_extent'], int(params['X_discretization']), endpoint=True)
    return X, Xn_exc, Xn_inh, exc_connected_neighbors, exc_decay_connect,\
        inh_connected_neighbors, inh_decay_connect, conduction_velocity

all_models = ['RING1', 'RING1-hd']

import sys
if __name__=='__main__':
    import pprint                   
    if len(sys.argv)==1:
        for m in all_models:
            p = get_model_params(m)
            print("==============================================")
            print("===----", p['name'], "-----===========")
            print("==============================================")
            pprint.pprint(p)
    else:
        p = get_model_params(sys.argv[-1])
        print("==============================================")
        print("===----", p['name'], "-----===========")
        print("==============================================")
        pprint.pprint(p)
        import matplotlib.pylab as plt
        sys.path.append('../code')
        from my_graph import set_plot
        fig, ax = plt.subplots(1)
        plt.subplots_adjust(left=.3, bottom=.3, top=.8)
        x = np.linspace(-p['X_extent']/2., p['X_extent']/2., 1e3)
        ax.plot(x, gaussian_connectivity(x, 0., p['inh_connect_extent']), 'r-', lw=3, label='inhibition')
        ax.plot(x, gaussian_connectivity(x, 0., p['exc_connect_extent']), 'b-', lw=3, label='excitation')
        ax.set_title('connectivity \n profile')
        ax.legend(frameon=False)
        set_plot(ax, ylabel='connectivity probability',\
                 xlabel='distance from center pixel (mm)')
        plt.show()
        


