import numpy as np
import sys, pathlib, os
sep = os.path.sep # Ms-Win vs UNIX
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
# for running simulations
from sparse_vs_balanced.running_2pop_model import run_2pop_ntwk_model
from sparse_vs_balanced.running_3pop_model import run_3pop_ntwk_model
from itertools import product
# for analysis
import neural_network_dynamics.main as ntwk
from data_analysis.IO.hdf5 import load_dict_from_hdf5
from graphs.my_graph import *
from matplotlib import ticker
from PIL import Image # BITMAP (png, jpg, ...)
from fpdf import FPDF
# everything stored within a zip file
import zipfile
Blue, Orange, Green, Red, Purple, Brown, Pink, Grey,\
    Kaki, Cyan = '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'

def my_logspace(x1, x2, n):
    return np.logspace(np.log(x1)/np.log(10), np.log(x2)/np.log(10), n)

def get_scan(Model, filename=None):

    if filename is None:
        filename=str(Model['zip_filename'])
    zf = zipfile.ZipFile(filename, mode='r')
    
    data = zf.read(filename.replace('.zip', '_Model.npz'))
    with open(filename.replace('.zip', '_Model.npz'), 'wb') as f: f.write(data)
    Model = dict(np.load(filename.replace('.zip', '_Model.npz')).items())
    
    F_aff, seeds = Model['F_AffExc_array'], Model['SEEDS']
    
    DATA = []
    for i, j in product(range(len(F_aff)), range(len(seeds))):
        
        fn = Model['FILENAMES'][i,j]
        data = zf.read(fn)
        with open(fn, 'wb') as f: f.write(data)
        with open(fn, 'rb') as f: data = load_dict_from_hdf5(fn)
        data['faff'], data['seed'] = F_aff[i], seeds[j]
        DATA.append(data)
        
    return Model, F_aff, seeds, DATA
    
def analyze_scan(Model,
                 irreg_criteria=0.01, synch_criteria=0.9,
                 filename=None):

    Model, F_aff, seeds, DATA = get_scan(Model,
                                         filename=filename)
    if filename is None:
        filename=str(Model['zip_filename'])

    # printing parameters
    print('=========================================================')
    print('---------- NETWORK AND SIMULATION PARAMETERS -------------')
    print('=========================================================')
    for i, (key, val) in enumerate(Model.items()):
        print(key, val)
    
    FA, FD = [], []
    SYNCH, IRREG = [], []
    BALANCE, EXC_ACT, INH_ACT = [], [], []
    EXC_II, EXC_IE, INH_II, INH_IE = [], [], [], []

    print('Running analysis [...]')
    for i in range(len(DATA)):
        
        FA.append(DATA[i]['faff'])
        output = ntwk.get_all_macro_quant(DATA[i],
                                          exc_pop_key='RecExc',
                                          inh_pop_key='RecInh', other_pops=['DsInh'])
        FD.append(output['mean_DsInh'])
        SYNCH.append(output['synchrony'])
        IRREG.append(output['irregularity'])
        BALANCE.append(output['balance_Exc'])
        EXC_ACT.append(output['mean_exc'])
        INH_ACT.append(output['mean_inh'])
        EXC_IE.append(output['meanIe_Exc'])
        EXC_II.append(-output['meanIi_Exc']) # absolute value !
        INH_IE.append(output['meanIe_Inh'])
        INH_II.append(-output['meanIi_Inh']) # absolute value !

    print('Done with analysis, now plotting and pdf export [...]')
    
    QUANT = [SYNCH, IRREG, BALANCE, EXC_ACT, INH_ACT, EXC_IE, EXC_II, FD]
    for q in QUANT: q = np.array(q)
    
    np.save(filename.replace('.zip', '_analyzed.npy'), [FA, *QUANT])

def run_scan(Model):
    
    zf = zipfile.ZipFile(Model['zip_filename'], mode='w')

    #####################################################
    ############# GRID OF PARAMETER SPACE ###############
    #####################################################

    F_aff, seeds = Model['F_AffExc_array'], Model['SEEDS']
    
    Model['FILENAMES'] = np.empty((len(F_aff), len(seeds)), dtype=object)
    
    for i, j in product(range(len(F_aff)), range(len(seeds))):
        fn = Model['data_folder']+str(F_aff[i])+'_'+str(seeds[j])+\
             '_'+str(np.random.randint(100000))+'.h5'
        Model['FILENAMES'][i,j] = fn
        print('running configuration ', fn)
        Model['F_AffExc'] = F_aff[i]
        if Model['p_AffExc_DsInh']>0:
            run_3pop_ntwk_model(Model, filename=fn, SEED=seeds[j])
        else:
            run_2pop_ntwk_model(Model, filename=fn, SEED=seeds[j])
        zf.write(fn)
        
    # writing the parameters
    np.savez(Model['zip_filename'].replace('.zip', '_Model.npz'), **Model)
    zf.write(Model['zip_filename'].replace('.zip', '_Model.npz'))

    zf.close()

    
if __name__=='__main__':

    # import the model defined in root directory
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from model import *

    parser.add_argument('--F_AffExc_array', nargs='+', help='Afferent firing rates', type=float,
                        default=my_logspace(1, 25, 2))    
    parser.add_argument('--SEEDS', nargs='+', help='various seeds', type=int,
                        default=np.arange(2))    
    parser.add_argument('-df', '--data_folder', help='Folder for data', default='data'+sep)    
    parser.add_argument("-a", "--analyze", help="ANALYSIS", action="store_true")
    
    # additional stuff
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--zip_filename", '-f', help="filename for the zip file",type=str, default='data.zip')

    args = parser.parse_args()
    Model = vars(args) # overwrite model based on passed arguments
    
    if args.analyze:
        analyze_scan(Model)
    else:
        run_scan(Model)
    
