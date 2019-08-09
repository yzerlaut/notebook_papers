import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator

try:
    from .template_and_fitting import erfc_func, fitting_Vthre_then_Fout,\
        final_threshold_func, print_reduce_parameters
except ModuleNotFoundError:
    from template_and_fitting import erfc_func, fitting_Vthre_then_Fout,\
        final_threshold_func, print_reduce_parameters
    
def make_3d_fig(P, Fout, s_Fout, muV, sV, Tv_ratio,\
                muGn, Gl, Cm, El, cell_id, vthre_lim=None,\
                FONTSIZE=18):
    
    font = {'size'   : FONTSIZE}
    mpl.rc('font', **font)

    Tv_ratio = np.round(1000.*Tv_ratio)/10
    sV, muV = np.round(sV), np.round(muV)
    Tv_levels = np.unique(Tv_ratio)

    muV_levels = np.unique(muV)
    DISCRET_muV = len(muV_levels)

    # 3d view - Fout
    fig1 = plt.figure(figsize=(7,4))
    plt.subplots_adjust(left=.1, bottom=.2, right=.78, top=0.95)
    ax = plt.subplot(111, projection='3d')
    ax.set_title(cell_id)
    ax.view_init(elev=20., azim=210.)
    plt.xlabel('\n\n $\mu_V$ (mV)')
    plt.ylabel('\n\n $\sigma_V$ (mV)')
    ax.set_zlabel('\n\n $\\nu_\mathrm{out}$ (Hz)')

    # the colorbar to index the autocorrelation
    ax2 = plt.axes([.82, .1, .02, .8])
    Tv_levels = np.unique(Tv_ratio)
    # levels no more than 5
    mymap = mpl.colors.LinearSegmentedColormap.from_list(\
                        'mycolors',['red','blue'])
    bounds= np.linspace(Tv_levels.min()-10, Tv_levels.max()+10, len(Tv_levels)+1)
    norm = mpl.colors.BoundaryNorm(bounds, mymap.N)
    cb = mpl.colorbar.ColorbarBase(ax2, cmap=mymap, norm=norm,
                                    orientation='vertical')
    cb.set_ticks(np.round(Tv_levels)) 
    cb.set_label('$\\tau_V / \\tau_\mathrm{m}^0$ (%)', fontsize=16)
    
    for TvN in Tv_levels:

        i_repet_Tv = np.where(Tv_ratio==TvN)[0]
        # setting color
        if len(Tv_levels)>1:
            r = np.min([1,(TvN-bounds.min())/(bounds.max()-bounds.min())])
        else:
            r=1

        muV2, sV2 = muV[i_repet_Tv], sV[i_repet_Tv]
        Fout2, s_Fout2 = Fout[i_repet_Tv], s_Fout[i_repet_Tv]
    
        for muV3 in np.unique(muV2):
            
            i_repet_muV = np.where(muV2==muV3)[0]
            i_muV = np.where(muV3==muV_levels)[0]

            sV3 = sV2[i_repet_muV]
            Fout3, s_Fout3 = Fout2[i_repet_muV], s_Fout2[i_repet_muV]

            ax.plot(muV3*np.ones(len(sV3)), sV3, Fout3,\
                     'D', color=mymap(r,1), ms=6, lw=0)
            
            sv_th = np.linspace(0, sV3.max())
            muGn3 =np.ones(len(sv_th))
            Vthre_th = final_threshold_func(P,\
                  1e-3*muV3, 1e-3*sv_th, TvN/100., muGn3, El)
            Fout_th = erfc_func(1e-3*muV3, 1e-3*sv_th,\
                                    TvN/100., Vthre_th, Gl, Cm)
            ax.plot(muV3*np.ones(len(sv_th)), sv_th,\
                    Fout_th, color=mymap(r,1), alpha=.7, lw=3)
                
            for ii in range(len(Fout3)): # then errobar manually
                    ax.plot([muV3, muV3], [sV3[ii], sV3[ii]],\
                        [Fout3[ii]+s_Fout3[ii], Fout3[ii]-s_Fout3[ii]],\
                        marker='_', color=mymap(r,1))
                    
    ax.set_zlim([0., Fout.max()])
    ax.xaxis.set_major_locator( MaxNLocator(nbins = 4,prune='both') )
    ax.yaxis.set_major_locator( MaxNLocator(nbins = 4) )
    ax.zaxis.set_major_locator( MaxNLocator(nbins = 4,prune='lower'))
    
    ax.set_zlim([0., max([1,Fout.max()])])

    ax.xaxis.set_major_locator( MaxNLocator(nbins = 4,prune='both') )
    ax.yaxis.set_major_locator( MaxNLocator(nbins = 4) )
    ax.zaxis.set_major_locator( MaxNLocator(nbins = 4,prune='lower') )

    return fig1

if __name__=='__main__':
    # for spiking properties, what model ?? see models.py
    import argparse
    parser=argparse.ArgumentParser(description=
     """ 
     Stimulate a reconstructed cell with a shotnoise and study Vm dynamics
     """
    ,formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument("NEURON",\
                        help="Choose a cell (e.g. 'cell1') or a model of neuron (e.g. 'LIF')", default='LIF')

    args = parser.parse_args()
    
    data = np.load('data/'+args.NEURON+'.npz')

    ##### FITTING OF THE PHENOMENOLOGICAL THRESHOLD #####
    # two-steps procedure, see template_and_fitting.py
    # need SI units !!!
    P = fitting_Vthre_then_Fout(data['Fout'], 1e-3*data['muV'],\
                                1e-3*data['sV'], data['TvN'],\
                                data['muGn'], data['Gl'], data['Cm'],
                                data['El'], print_things=True)

    ##### PLOTTING #####
    # see plotting_tools.py
    # need non SI units (electrophy units) !!!
    FIG = make_3d_fig(P,\
                      data['Fout'], data['s_Fout'], data['muV'],\
                      data['sV'], data['TvN'], data['muGn'],\
                      data['Gl'], data['Cm'], data['El'], args.NEURON)


    plt.show()
