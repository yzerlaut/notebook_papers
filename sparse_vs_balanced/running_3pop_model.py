import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk

def run_3pop_ntwk_model(Model,
                        tstop=2000.,
                        filename='data/sas.h5',
                        with_raster=True, with_Vm=4,
                        with_pop_act=True,
                        with_synaptic_currents=True,
                        with_synaptic_conductances=True,
                        faff_waveform_func=None,
                        with_synchronous_input=False,
                        verbose=False, SEED=3):

    Model['SEED'], Model['tstop'] = SEED, tstop # adjusting simulation seed and length 
    
    NTWK = ntwk.build_populations(Model, ['RecExc', 'RecInh', 'DsInh'],
                                  AFFERENT_POPULATIONS=['AffExc'],
                                  with_raster=with_raster, with_Vm=with_Vm,
                                  with_pop_act=with_pop_act,
                                  with_synaptic_currents=with_synaptic_currents,
                                  with_synaptic_conductances=with_synaptic_conductances,
                                  verbose=verbose)

    ntwk.build_up_recurrent_connections(NTWK, SEED=SEED, verbose=verbose)

    #######################################
    ########### AFFERENT INPUTS ###########
    #######################################

    t_array = ntwk.arange(int(Model['tstop']/Model['dt']))*Model['dt']

    if faff_waveform_func is None:
       faff_waveform =  Model['F_AffExc']+0.*t_array
    else:
       faff_waveform = faff_waveform_func(t_array, Model)
    # # # afferent excitation onto cortical excitation and inhibition
    for i, tpop in enumerate(['RecExc', 'RecInh', 'DsInh']): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff_waveform,
                                         with_presynaptic_spikes=True,
                                         verbose=verbose,
                                         SEED=int(37*SEED+i)%13)



    if with_synchronous_input:
        step_input = ntwk.array([\
           Model['faff1'] if ((tt>Model['t0']) & (tt<Model['t0']+Model['time_span'])) else 0 for tt in t_array])
        ntwk.construct_feedforward_input_synchronous(NTWK, 'RecExc', 'AffExc',
                                             Model['N_aff_stim'], Model['N_target'], Model['N_duplicate'],
                                             t_array, step_input,
                                             with_presynaptic_spikes=True, SEED=Model['Pattern_Seed'])
    
    ################################################################
    ## --------------- Initial Condition ------------------------ ##
    ################################################################
    ntwk.initialize_to_rest(NTWK)
    
    #####################
    ## ----- Run ----- ##
    #####################
    network_sim = ntwk.collect_and_run(NTWK, verbose=verbose)

    ntwk.write_as_hdf5(NTWK, filename=filename)

    return NTWK

if __name__=='__main__':

    from data_analysis.processing.signanalysis import gaussian_smoothing
    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import *

    args = parser.parse_args()
    Model = vars(args)
    if Model['p_AffExc_DsInh']==0.:
        print('---------------------------------------------------------------------------')
        print('to run the 3 pop model, you need to set an afferent connectivity proba !')
        print('e.g run: \n                python running_3pop_model.py --p_AffExc_DsInh 0.1')
        print('---------------------------------------------------------------------------')
    else:
        NTWK = run_3pop_ntwk_model(Model, tstop=400)
        Nue = NTWK['POP_ACT'][0].rate/ntwk.Hz
        ntwk.plot(NTWK['POP_ACT'][0].t/ntwk.ms, gaussian_smoothing(Nue,int(20./0.1)))
        ntwk.show()
