import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import neural_network_dynamics.main as ntwk

def run_2pop_ntwk_model(Model,
                        tstop=2000.,
                        filename='data/sas.h5',
                        with_raster=True, with_Vm=4,
                        with_pop_act=True,
                        with_synaptic_currents=True,
                        with_synaptic_conductances=True,
                        faff_waveform_func=None,
                        verbose=False, SEED=3):

    Model['SEED'], Model['tstop'] = SEED, tstop # adjusting simulation seed and length 
    
    NTWK = ntwk.build_populations(Model, ['RecExc', 'RecInh'],
                                  AFFERENT_POPULATIONS=['AffExc', 'DsInh'],
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
    for i, tpop in enumerate(['RecExc', 'RecInh']): # both on excitation and inhibition
        ntwk.construct_feedforward_input(NTWK, tpop, 'AffExc',
                                         t_array, faff_waveform,
                                         with_presynaptic_spikes=True,
                                         verbose=verbose,
                                         SEED=int(37*SEED+i)%13)

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

    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import *

    args = parser.parse_args()
    Model = vars(args)
    
    NTWK = run_2pop_ntwk_model(Model, tstop=400)
    ntwk.plot(NTWK['POP_ACT'][1].t/ntwk.ms, NTWK['POP_ACT'][1].rate/ntwk.Hz)
    ntwk.show()
