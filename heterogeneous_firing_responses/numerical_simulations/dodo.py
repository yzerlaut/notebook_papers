
"""
================================================================
generate_data_to_investigate_control_of_fluctuations
================================================================


"""

def Build_task_fluct_ctrl(MODEL):
    return {'actions':['python run_3d_scan_stim.py '+MODEL+' --RANGE_FOR_3D -0.07 -0.05 0.004 0.008 0.001 0.004 0.15 --DISCRET_TvN 8 --DISCRET_muV 8 --SUFFIX_NAME for_fluct_ctrl'],\
            'file_dep':['run_3d_scan_stim.py'],\
            'targets': ['../data/'+MODEL+'_for_fluct_ctrl.npz']}

def task_generate_data_to_investigate_control_of_fluctuations():
    for MODEL in ['LIF', 'SUBTHRE', 'iAdExp']:
        T = Build_task_fluct_ctrl(MODEL)
        T['basename'] = 'Generate data to investigate control of fluctuations : '+MODEL+' model'
        yield T
