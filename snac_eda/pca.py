import os

import pandas as pd

from datasciencetools.base import *
from datasciencetools.eda import *
from datasciencetools.plot import *
from datasciencetools.metrics import *
from datasciencetools.modeling import *
from datasciencetools.preprocessing import *
from datasciencetools.multiblock import *
# from datasciencetools.data_filter import *
from data_filter import *
from specific_plot import *
import logging
import time
import argparse
import json
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser('Running datascience on SNAC')
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=60621)
parser.add_argument('-y', '--y_variable', type=str, default='VFA_geq_y', choices=['TAN_relative_error', 'TAN_absolute_error', 'VFA_relative_error', 'VFA_absolute_error',
                                                                                  'VFA_geq_x', 'VFA_geq_y', 'TAN_geq_x', 'TAN_geq_y', 'outlier_VFA'], help='The y variable to predict')
parser.add_argument('-x', '--x_dataset_type', type=str, default='volume', choices=['volume', 'volume_dilutioncorrected', 'conductivity','conductivity_dilutioncorrected', 'custom_criteria', 'mb_volume_conductivity'], help='The X dataset to use for prediction')
parser.add_argument('--json_config', type=str, help='The config as json string')
args = parser.parse_args()

# add a unique run_id :
args.__dict__.update({'run_id':time.strftime("%Y%m%d-%H%M%S")})

if args.json_config:# -y VFA_absolute_error -x custom_criteria --json_config "{\"preprocessing\":{\"activate\":false, \"steps\":{}}, \"normalize\":true, \"plot\":{\"plot_dataset_graphs\":false, \"pearsonr_threshold\":0.8}}"
    args.__dict__.update(json.loads(args.json_config))

if 'VFA' in args.y_variable:
    # config = update_recursively(config, compounds_config['VFA'])
    compounds = 'VFA'
elif 'TAN' in args.y_variable:
    # config = update_recursively(config, compounds_config['VFA'])
    compounds = 'TAN'

test_number_name = 'T18' # specific test name for folder
# other_info_name ='' # to give more detail in the folder name
other_info_name ='' # to give more detail in the folder name

compounds_config = {
    'VFA': {
        'filters': [
            ('VFA_geq_y', [0.2, np.inf]), # labo data
            ('VFA_indip', [1, 1]), # 0: dependent on SNAC analysis - 1: independent | mettre [1,1] quand comparatif SNAc/labo et [0,1] pour epoPLS
            ('VFA_geq_x', [0., np.inf]), # SNAC data
            # ('VFA_geq_x', [True]) # SNAC data - basic pH signal accepted
            ('103_0', [1,1]) # SNAC data - basic conductivity [1,1] = accepted; [0,1] not accepted + accepted
            ],
        'drop': {
           'general': True,
           'manual_conductivity': True,
           'source': [
               ('Project',[
                   # 'Benchmark officiel CCR',
                   # 'Benchmark officiel SRO',
                   # 'Benchmark officieux SRO'
                ])
           ],
        },
        'outliers_filters': ('outlier_VFA', [0, 0]), #outlier already identified in the excel file
        'preprocessing': {
            'activate': True,
            # 'type_data': 'raw',
            'type_data': 'derivative',
            'steps': {
                # 'pH_range': SelectColumnsBasedOnRange(column_range=[0., 12]), # all range
                #  'pH_range': SelectColumnsBasedOnRange(column_range=[3.0, 6.5]), # other
                # 'pH_range': SelectColumnsBasedOnRange(column_range=[3.8, 5.8]), # centré 4.8 - 1.0
                # 'pH_range': SelectColumnsBasedOnRange(column_range=[4.55, 5.05]), # centré 4.8 - 0.25
                'pH_range': SelectColumnsBasedOnRange(column_range=[4.3, 5.3]), # centré 4.8 - 0.5
                # 'pH_range': SelectColumnsBasedOnRange(column_range=[4.25, 4.75]), # centré 4.5 - 0.25
                # 'pH_range': SelectColumnsBasedOnRange(column_range=[4.0, 5.0]), # centré 4.5 - 0.5
                # 'pH_range': SelectColumnsBasedOnRange(column_range=[3.5, 5.5]), # centré 4.5 - 1.0
                # 'pH_range': SelectColumnsBasedOnRange(column_range=[3.8, 5.3]), # decentré
                # 'sg1':SavitzkyGolay(window_length=21, polyorder=2, deriv=1)
            },
            'normalize': False,
            },
        'model_type': {
                # 'linear_regression': { # comp SNAC et labo
                #     'apply': True,
                #     'filter': [
                #         ('VFA_geq_y', [0.2, np.inf]),  # labo data
                #         ('VFA_indip', [1, 1]),
                #         # 0: dependent on SNAC analysis - 1: independent | mettre [1,1] quand comparatif SNAc/labo et [0,1] pour epoPLS
                #         ('VFA_geq_x', [0., np.inf]), # SNAC data
                #     ],
                # },
                    'regression': {
                        'apply': True,
                         # 'type': 'plsr',
                         'type':'mbplsr',
                        # 'type':'epoplsr',
                    },
                    'prediction':{
                        'apply': False,
                        'filter':[
                            # [0.,0.19999],
                            ('VFA_geq_y', [0.2, np.inf]),  # labo data
                            ('VFA_indip', [1, 1]),
                            # 0: dependent on SNAC analysis - 1: independent | mettre [1,1] quand comparatif SNAc/labo et [0,1] pour epoPLS
                            # ('VFA_geq_x', [0., np.inf]),  # SNAC data
                            # ('VFA_geq_x', [np.inf, np.inf]),  # SNAC data
                                               ] }
                        }
        },

    #--------------------------   TAN   ------------------------------------------------------------------

    'TAN': {
        'filters': [
            ('TAN_geq_y', [0., np.inf]), # labo data
            ('TAN_indip', [1, 1]), # 0: dependent on SNAC analysis - 1: independent | mettre [1,1] quand comparatif SNAc/labo et [0,1] pour PLS
            ('TAN_geq_x', [0., np.inf]), # SNAC data
            ('103_0', [1, 1]) # SNAC data - basic conductivity [1,1] = accepted; [0,1] not accepted + accepted
        ],
        'drop': {
            'general': True,
            'manual_conductivity': True,
            'source': [
                ('Project', [
                    # 'Benchmark officiel CCR',
                    # 'Benchmark officiel SRO',
                    # 'Benchmark officieux SRO'
                ])
            ],
        },
        'outliers_filters':
            ('outlier_TAN', [0, 0]) #outlier already identified : 1 correspond to ouliers and 0 to not outliers
        # ('outlier labo TAN', [0,0])
        ,
        'preprocessing': {
            'activate': True,
            # 'type_data': 'raw',
            'type_data': 'derivative',
                'steps': {
                    # 'pH_range': SelectColumnsBasedOnRange(column_range=[0., 12]), # all range
                    # 'pH_range': SelectColumnsBasedOnRange(column_range=[7.5, 11.0]), # other
                    'pH_range': SelectColumnsBasedOnRange(column_range=[8.25, 10.25]), # centré sur 9.25 - 1.0
                    # 'pH_range': SelectColumnsBasedOnRange(column_range=[9.0, 9.5]), # centré sur 9.25 - 0.25
                    # 'pH_range': SelectColumnsBasedOnRange(column_range=[8.75, 9.75]), # centré sur 9.25 - 0.5
                    # 'pH_range': SelectColumnsBasedOnRange(column_range=[9.45, 9.95]), # centré 9.7 - 0.25
                    # 'pH_range': SelectColumnsBasedOnRange(column_range=[9.2, 10.2]), # centré 9.7 - 0.5
                    # 'pH_range': SelectColumnsBasedOnRange(column_range=[8.7, 10.7]), # centré 9.7 - 1.0
                  # 'sg1': SavitzkyGolay(window_length=21, polyorder=2, deriv=1)
                },
                'normalize': False,
            },
        'model_type': {
            'regression': {
                'apply': True,
                # 'type': 'plsr'
                'type':'mbplsr',
                # 'type':'epoplsr'
},
            'prediction': {
                'apply': False,
                'filter': [
                    [0., 0.19999],
                    ('VFA_geq_y', [0.2, np.inf]),  # labo data
                    ('VFA_indip', [1, 1]),
                    # 0: dependent on SNAC analysis - 1: independent | mettre [1,1] quand comparatif SNAc/labo et [0,1] pour epoPLS
                    ('VFA_geq_x', [0., np.inf]),  # SNAC data
                ]}
        }
    } # TAN compounds
} # config

default_config = {
    'project_path':'./projects/snac_eda/',
    'y_variable':'TAN_absolute_error', #'TAN_relative_error', 'TAN_absolute_error', 'VFA_relative_error', 'VFA_absolute_error'
    'x_dataset_type':'custom_criteria', # volume, conductivity, custom_criteria, volume_dilutioncorrected
    'global_data_analysis':{
        'activate': False,
        'x_dataset_type': 'custom_criteria',  # volume, conductivity, custom_criteria, volume_dilutioncorrected
    },
    'filtering' : {'activate':True,
                   'filters':compounds_config[compounds]['filters']
                   },
    'outliers_drop':{'activate':compounds_config[compounds]['drop']['general'],
                     'method':['filter', 'list'],# column # if "list" it is based on the "outliers" list, if "filter" it is based on the "outliers_filters"
                       'outliers':[
                           #'Methavert_20210406_1',
                           #'Methabelair_20200922_1', #excessive VFA level, excessive TAN_pka decalage)
                           #'INRA_V1A_20191128_3', #exessive custom criterias errors
                           #'INRA_V1A_20191128_3', #exessive custom criterias errors
                           #'INRA_V1A_20191205_3', 'INRA_V1A_20191205_4', 'INRA_V1A_20191210_4', 'INRA_V1A_20200728_2', 'INRA_V1A_20200922_1',
                           #'StepNantes_20211015_1'

                           'Medoc_20200728_2'  # outlier pas détecté (on peut améliorer le code SNAC)
                           
                            'INRA_V1A_20191120_5', # labo condition far form reality
                            'INRA_V1A_20191121_1',# labo condition far form reality
                            'INRA_V1A_20191121_2',# labo condition far form reality
                            'INRA_V1A_20191122_2',# labo condition far form reality
                            'INRA_V1A_20191122_5',# labo condition far form reality # mauvvais pour AGV
                            'INRA_V1A_20191125_5',# labo condition far form reality
                            'INRA_V1A_20191125_7',# labo condition far form reality
                            'INRA_V1A_20191128_3',# labo condition far form reality
                            'INRA_V1A_20191204_7',# labo condition far form reality # mauvvais pour TAN
                            'INRA_V1A_20191205_3',# labo condition far form reality
                            'INRA_V1A_20191205_4',# labo condition far form reality # mauvvais pour AGV
                            'INRA_V1A_20191210_2',# labo condition far form reality
                            'INRA_V1A_20191210_4',# labo condition far form reality # mauvais pour AGV et TAN
                            'INRA_V1A_20191223_1',# labo condition far form reality
                            'INRA_V1A_20191223_2' # labo condition far form reality
                                        ],
                     'activate_conductivity_manual':compounds_config[compounds]['drop']['manual_conductivity'],
                     'outliers_conductivity':[
                         'Methabelair_20201109_2',# signal perturbé non identifié comme mauvais
                         'Methavert_20211213_2' # gros pic positif bizarre sur la conductivité
                     ],
                    'source':compounds_config[compounds]['drop']['source'],
                     'outliers_filters':[compounds_config[compounds]['outliers_filters']
                        ]
                       },
    'plot':{'plot_dataset_graphs':True,
            'pearsonr_threshold':0.8,
            'predobs_params':{
                'add_reg_line':True,
                'add_qreg_line':False,
                'reg_params':{'fit_intercept': False},
                'qreg_params': {'fit_intercept': True, 'quantile': 0.95, 'alpha': 0},
            },
            'relatexy_params':{
                'add_reg_line':True,
                'add_qreg_line':True,
                'reg_params':{'fit_intercept': True},
                'qreg_params': {'fit_intercept': True, 'quantile': 0.95, 'alpha': 0}
            },
            'n_axes_per_fig':12,
            'label_outliers':True,
            'mb_params':{
                'loadings_max_n_comp':10,
            },
            },
    'compute_metrics':{'compute_with_indip_samples_only':{'activate':True,
                                                  'type':compounds+'_indip',
                                                  }},
    'preprocessing':compounds_config[compounds]['preprocessing'],
    'normalize':compounds_config[compounds]['preprocessing']['normalize'],
    'reduction':{'apply':False,
                 'type': PCA_analysis(n_components=None),
                 'params':{'n_components':6}
                 },
    'regression': {'plsr': {
                           'apply': True,
                           'type': 'plsr',
                           'params': {'n_components': 1, 'scale': False},
                           'cv': {'test_size': 0.5,
                                  'train_test_split_method': 'duplex_y',  # possible choices:[duplex_y, duplex_x, random]
                                  'cv_splitter': RepeatedKFold(n_splits=5, n_repeats=30, random_state=1),
                                  # 'cv_splitter': LeaveOneOut(),
                                  'gs_params':{'n_components':np.arange(1,10)}
                                  # 'gs_params': None,
                                  }  # np.arange(1,20) range of n_components looked at
                           },
                    'epoplsr': {
                           'apply': True,
                           'type':'epoplsr',
                           'params': {'epo__n_components':1, 'plsr__scale':False, 'plsr__n_components':1},
                           'cv': {'test_size': 0.5,
                                  'train_test_split_method': 'duplex_y',  # possible choices:[duplex_y, duplex_x, random]
                                  'cv_splitter': RepeatedKFold(n_splits=5, n_repeats=30, random_state=1),
                                  # 'cv_splitter': LeaveOneOut(),
                                  # 'gs_params':{'n_components':[3,]}
                                  'gs_params':{'epo__n_components':[0, 1, 2, 3], 'plsr__n_components':np.arange(1,10)}
                                  },  # np.arange(1,20) range of n_components looked at
                           'detrimental_criterias': [
                               (compounds+'_indip', [0, 0]),
                           ],
                           },
                    'mbplsr': {
                           'apply': True,
                           'type': 'mbplsr',
                           'params': {'n_components': 1, 'standardize': True, 'scaler_type':HardBlockVarianceScaling()},
                           'cv': {'test_size': 0.5,
                                  'train_test_split_method': 'duplex_y',
                                  'cv_splitter': RepeatedKFold(n_splits=5, n_repeats=30, random_state=1),
                                  'gs_params':{'n_components':np.arange(1,10)}
                                  }  # np.arange(1,20) range of n_components looked at if gs_params=None
                           },
                    },

    'outlier_detection': {'apply':False},
    'custom_criteria_variables':['503_RMSE_VFA_pka_th', '503_RMSE_pH6_pka_th', '503_RMSE_VFA_pH6_pka_th', '503_RMSE_TAN_pka_th', '503_RMSE_global_pka_th',
                              '503_diff_VFA_pka_th', '503_diff_pH6_pka_th', '503_diff_VFA_pH6_pka_th', '503_diff_TAN_pka_th', '503_diff_global_pka_th',
                              #'503_RMSE_VFA_pka_exp', '503_RMSE_pH6_pka_exp', '503_RMSE_VFA_pH6_pka_exp', '503_RMSE_TAN_pka_exp', '503_RMSE_global_pka_exp',
                              #'503_diff_VFA_pka_exp', '503_diff_pH6_pka_exp', '503_diff_VFA_pH6_pka_exp', '503_diff_TAN_pka_exp', '503_diff_global_pka_exp',
                              '504_distance_pka_VFA', '504_distance_pka_TAN'],
}

config = default_config.copy()
config = update_recursively(config, vars(args))
#config.update(vars(args))


# set paths
input_path = config['project_path'] + 'inputs/'
output_path = config['project_path'] + 'outputs/'
sub_output_path = output_path + args.run_id +'/'

# create folder name
pH_range = str(compounds_config[compounds]['preprocessing']['steps']['pH_range'].column_range[0]).replace(".","") +'-'+ str(compounds_config[compounds]['preprocessing']['steps']['pH_range'].column_range[1]).replace(".","")
stat_model = compounds_config[compounds]['model_type']['regression']['type']
signal = args.x_dataset_type.replace("_","-")
folder_name = test_number_name +'_'+ compounds +'_'+ pH_range +'_'+ stat_model +'_'+ signal
if other_info_name != '':
    folder_name = folder_name +'_' + other_info_name
sub_output_path = output_path + folder_name +'/'
sub_output_path_data_article = sub_output_path + 'Data_article/'

# create folders
for p in [input_path, output_path]:
       if not os.path.exists(p):
              os.makedirs(p)
for p in [sub_output_path, sub_output_path_data_article]:
    if os.path.exists(p):
        import shutil
        shutil.rmtree(p)
        os.makedirs(p)
    else :
        os.makedirs(p)

    # set log
logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)-15s][%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            handlers=(
                # logging.FileHandler(config['project_path'] + "logs/log_"+args.run_id+".log", mode='w'),
                logging.FileHandler(sub_output_path + "/log_"+args.run_id+".log", mode='a'),
                logging.StreamHandler()
            ))

logging.info('Running EDA for: %s'%compounds)
logging.info('Running EDA using following config: %s'%config)


    # check that are no incohernece in the config
check_config(compounds, compounds_config, config)


    # save run description in runs registry:
runs_registry = pd.read_csv(output_path + 'runs_registry.csv', sep=';', decimal='.', index_col=0)
cur_run = pd.Series(index=runs_registry.columns, name=config['run_id'],  data=[config['x_dataset_type'], config['y_variable'], config['filtering']['activate'], config['filtering']['filters'], config['outliers_drop']['activate'], config['outliers_drop']['outliers'], config['preprocessing']['activate'], config['preprocessing']['steps'], config['normalize']])
runs_registry = runs_registry.append(cur_run)
runs_registry.to_csv(output_path + 'runs_registry.csv', sep=';', decimal='.')

    # read data
logging.info('# Importing data')
measurements_cond = pd.read_csv(input_path + '/Res_stat_modified/Res_conductivity_processed.csv', index_col=-1, header=0, sep=',', decimal='.').T.sort_index(axis=1)
measurements_vol = pd.read_csv(input_path + '/Res_stat_modified/Res_titrant_volume_processed.csv', index_col=-1, header=0, sep=',', decimal='.').T.sort_index(axis=1)
measurements_cond_derive = pd.read_csv(input_path + '/Res_stat_modified/Res_dconductivity_dpH_processed.csv', index_col=-1, header=0, sep=',', decimal='.').T.sort_index(axis=1)
measurements_vol_derive = pd.read_csv(input_path + '/Res_stat_modified/Res_buffer_capacity_processed.csv', index_col=-1, header=0, sep=',', decimal='.').T.sort_index(axis=1)

# We chose which X use for stat model
type_signal = compounds_config[compounds]['preprocessing']['type_data']
if type_signal == 'raw':
    logging.info('# Raw volume/conductivity signal(s) used')
    X_vol = measurements_vol
    X_cond = measurements_cond
elif type_signal == 'derivative':
    logging.info('# Derivative volume/conductivity signal(s) used')
    X_vol = measurements_vol_derive
    X_cond = measurements_cond_derive

if config['plot']['plot_dataset_graphs']:
    logging.info('plotting '+type_signal+' conductivity and volume')
    fig, axes = plt.subplots(1,2, figsize=(10,6))
    X_cond.T.plot(ax=axes[0], lw=.8, grid=True, title='Conductivity '+type_signal, legend=False)
    X_vol.T.plot(ax=axes[1], lw=.8, grid=True, title='Volume '+type_signal, legend=False)
    fig.savefig(sub_output_path + 'All_'+type_signal+'_cond_vol.png')

X_vol.columns = X_vol.columns.astype(str)
X_cond.columns = X_cond.columns.astype(str)

analyse_labo = pd.read_csv(input_path + '/Analyse_labo.csv', index_col=None, header=0, sep=';', decimal=',')
analyse_labo.set_index(['Nom'], inplace=True)
analyse_labo.loc[:,['VFA_geq', 'TAN_geq']] = analyse_labo.loc[:,['VFA_geq', 'TAN_geq']].astype(np.float64) # nan values were set as strings..

Res = pd.read_csv(input_path + '/Res.csv', index_col=0, header=0, sep=';', decimal='.')
Res['VFA_geq'] = Res.apply(lambda x:x['VFA_geq']/1000 if x['VFA_geq']>100 else x['VFA_geq'], axis=1) #todo: attention
Res['TAN_geq'] = Res.apply(lambda x:x['TAN_geq']/1000 if x['TAN_geq']>100 else x['TAN_geq'], axis=1) #todo: attention

logging.info('Cond %s, Volume %s, Res %s, analyse_labo %s'%(X_cond.shape, X_vol.shape, Res.shape, analyse_labo.shape))
logging.info('# Merging lab/snac analyses, defining custom criteria')
merged_Res_analyselabo_raw = Res.merge(analyse_labo, how='left', left_index=True, right_index=True)

    # filter labo data
custom_criteria = merged_Res_analyselabo_raw.loc[:, config['custom_criteria_variables']].copy()# todo: we could change the logic of custom criteria
merged_Res_analyselabo, custom_criteria, indip_samples_list = data_filtering(merged_Res_analyselabo_raw, config, custom_criteria, args)
# merged_Res_analyselabo, indip_samples_list = data_filtering(merged_Res_analyselabo_raw, config, args)

    # filter and process raw data
X, y = data_raw_filter(merged_Res_analyselabo, X_vol, X_cond, config, custom_criteria, sub_output_path)

# Global data analysis based on expert criteria
if config['global_data_analysis']['activate']:
    logging.info('# Global data analysis active')

    # defining custom_criteria variables:
    logging.info('Custom_criteria variables used: %s' % config['custom_criteria_variables'])
    logging.info('merged_Res_analyselabo_raw: %s, custom_criteria: %s' % (
    merged_Res_analyselabo_raw.shape, custom_criteria.shape))

    logging.info('# Compute all errors (lab vs. snac) - relative/absolute')

    def relative_error(y_obs, y_pred):
        return np.abs(y_pred - y_obs) / y_obs if y_obs else np.nan
    def absolute_error(y_obs, y_pred):
        return (y_pred - y_obs)

    merged_Res_analyselabo_raw['TAN_relative_error'] = merged_Res_analyselabo_raw.loc[:,['TAN_geq_x', 'TAN_geq_y']].apply(
        lambda row: relative_error(row['TAN_geq_x'], row['TAN_geq_y']), axis=1)
    merged_Res_analyselabo_raw['TAN_absolute_error'] = merged_Res_analyselabo_raw.loc[:,['TAN_geq_x', 'TAN_geq_y']].apply(
        lambda row: absolute_error(row['TAN_geq_x'], row['TAN_geq_y']), axis=1)
    merged_Res_analyselabo_raw['VFA_relative_error'] = merged_Res_analyselabo_raw.loc[:,['VFA_geq_x', 'VFA_geq_y']].apply(
        lambda row: relative_error(row['VFA_geq_x'], row['VFA_geq_y']), axis=1)
    merged_Res_analyselabo_raw['VFA_absolute_error'] = merged_Res_analyselabo_raw.loc[:,['VFA_geq_x', 'VFA_geq_y']].apply(
        lambda row: absolute_error(row['VFA_geq_x'], row['VFA_geq_y']), axis=1)

    # plot
    plot_dataset_graphs(merged_Res_analyselabo, compounds, config, custom_criteria, sub_output_path)
    plot_snac_labo(merged_Res_analyselabo, compounds, config, custom_criteria, sub_output_path)

    if config['x_dataset_type'] == 'custom_criteria':
        logging.info('# custom criteria chosen: running a simple and quantile regression parameter per parameter')

        fid=-1
        for i,xcol in enumerate(X.columns):
            if i%config['plot']['n_axes_per_fig']==0:
                fig, axes = plt.subplots(*get_aspect_ratio(config['plot']['n_axes_per_fig']), figsize=(16, 10))
                axes =axes.ravel()
                fid+=1
            x = X.loc[:,[xcol]]
            if config['plot']['label_outliers']:
                fig, axes[i-fid*config['plot']['n_axes_per_fig']] = relate_xy(x=x.values, y=y.values, fig=fig, ax=axes[i-fid*config['plot']['n_axes_per_fig']],
                                                                              add_reg_line=config['plot']['relatexy_params']['add_reg_line'], add_qreg_line=config['plot']['relatexy_params']['add_qreg_line'], labels=x.index, label_only_outliers=True,
                                                                              reg_params=config['plot']['relatexy_params']['reg_params'], qreg_params=config['plot']['relatexy_params']['qreg_params'])
            else:
                fig, axes[i - fid * config['plot']['n_axes_per_fig']] = relate_xy(x=x.values, y=y.values, fig=fig, ax=axes[i - fid * config['plot']['n_axes_per_fig']],
                                                                                  add_reg_line=config['plot']['relatexy_params']['add_reg_line'], add_qreg_line=config['plot']['relatexy_params']['add_qreg_line'], #labels=x.index, label_only_outliers=True,
                                                                                  reg_params=config['plot']['relatexy_params']['reg_params'], qreg_params=config['plot']['relatexy_params']['qreg_params'])
            axes[i-fid*config['plot']['n_axes_per_fig']].set_xlabel(xcol)
            axes[i-fid*config['plot']['n_axes_per_fig']].set_ylabel(config['y_variable'])

            if ((i+1)%config['plot']['n_axes_per_fig']==0) | (i==X.shape[1]-1):
                #fig.tight_layout()
                fig.savefig(sub_output_path + 'custom_criteria_simplereg_f%s.png'%(fid))



                ##############################
                #######   Stat model   #######
                ##############################
regression_type = compounds_config[compounds]['model_type']['regression']['type']
# run reduction EDA (PCA)
if config['reduction']['apply']:
    if regression_type == 'mnbplsr':
        logging.error('Cannot apply reduction with multiblock methods MBPLSR') # todo: treat this case
    else:
        reductor = run_PCA(X, config)
        scores = reductor.transform(X) # needed to activate all the properties
        loadings = reductor.components_ # needed to activate all the properties
        # plot PCA
        plot_PCA(X, y, reductor, config, sub_output_path)


# PLS - EPO - MBPLS
if compounds_config[compounds]['model_type']['regression']['apply']:
    if regression_type == 'plsr':
        logging.info('# Running PLSR')
        estimator = PLS_analysis()
        estimator.set_params(**config['regression'][regression_type]['params'])
    elif regression_type == 'epoplsr':
        logging.info('# Running EPO-PLSR')
        logging.info('# Defining X_detrimental and y_detrimental, based on detrimental criterias: %s'%config['regression'][regression_type]['detrimental_criterias'])
        merged_Res_analyselabo_detrimental = apply_filter(merged_Res_analyselabo, config['regression'][regression_type]['detrimental_criterias'])
        # for f in config['regression'][regression_type]['detrimental_criterias']:
        #     merged_Res_analyselabo_detrimental = merged_Res_analyselabo[(merged_Res_analyselabo[f[0]] >= f[1][0]) & (merged_Res_analyselabo[f[0]] <= f[1][1])]
        detrimental_ids = merged_Res_analyselabo_detrimental.index.unique()

        X_detrimental = X.loc[X.index.isin(detrimental_ids),:]
        X = X.loc[~X.index.isin(detrimental_ids),:]
        y_detrimental = y.loc[y.index.isin(detrimental_ids), :]
        y = y.loc[~y.index.isin(detrimental_ids), :]
        logging.info('X_detrimental:%s, y_detrimental:%s, X:%s, y:%s' % (X_detrimental.shape, y_detrimental.shape, X.shape, y.shape))

        # center samples from same client/serie
        X_detrimental = X_detrimental.merge(merged_Res_analyselabo_detrimental, left_index=True, right_index=True, how='left').groupby(['client', 'Serie'])[X_detrimental.columns].apply(lambda x:x - x.mean())

        epo = EPO(X_detrimental=X_detrimental)
        plsr = PLS_analysis()
        estimator = Pipeline(steps=[('epo', epo), ('plsr', plsr)])
        estimator.set_params(**config['regression'][regression_type]['params'])

    elif regression_type == 'mbplsr':
        logging.info('# Running MBPLSR')
        estimator = MBPLSRegression()
        estimator.set_params(**config['regression'][regression_type]['params'])
    else:
        logging.error('specify regression type, "%s" unknown'%regression_type)

    # train model
    if config['y_variable'] in [compounds+'_geq_y']: # case where model is used to directly predict VFA_geq_y or TAN_geq_y instead of error
        y_ref_pred_snac = config['y_variable'].replace('_y','_x')

        if isinstance(estimator, MBPLSRegression):
            gscv, (fig1, ax1), (fig2, ax2, ax3), train_ids, test_ids = run_regression_MB(X=X, y=y, model=estimator, test_size=config['regression'][regression_type]['cv']['test_size'],
                                                                                      train_test_split_method=config['regression'][regression_type]['cv']['train_test_split_method'],
                                                                                      cv=config['regression'][regression_type]['cv']['cv_splitter'],
                                                                                      gs_params=config['regression'][regression_type]['cv']['gs_params'],
                                                                                      y_ref_pred=y.merge(merged_Res_analyselabo.loc[:,[y_ref_pred_snac]].drop_duplicates().dropna(),
                                                                                      how='left', left_index=True, right_index=True).loc[:,[y_ref_pred_snac]], ignored_samples=indip_samples_list)

            ax2.set_title('With %s' % regression_type)
            ax3.set_title('With SNAC')

            # run stacking mbplr
            y_train, y_test, y_train_pred_stacking, y_test_pred_stacking, stacking = run_stacking_MB(X,y, y_ref_pred_snac, merged_Res_analyselabo, train_ids, test_ids, config,regression_type, sub_output_path, gscv, MBPLSRegression)

            # plot stacking figures
            plot_stacking_MB(y_train, y_test, y_train_pred_stacking, y_test_pred_stacking, stacking, sub_output_path)


        else: # PLS or EPO
            gscv, (fig1, ax1), (fig2, ax2, ax3), train_ids, test_ids = run_regression(X=X, y=y, model=estimator, test_size=config['regression'][regression_type]['cv']['test_size'],
                                                                 train_test_split_method= config['regression'][regression_type]['cv']['train_test_split_method'],
                                                                cv = config['regression'][regression_type]['cv']['cv_splitter'],
                                                                gs_params = config['regression'][regression_type]['cv']['gs_params'],
                                                                y_ref_pred =y.merge(merged_Res_analyselabo.loc[:,[y_ref_pred_snac]].drop_duplicates().dropna(),
                                                                how='left', left_index=True, right_index=True).loc[:,[y_ref_pred_snac]], ignored_samples=indip_samples_list)
            ax2.set_title('With %s'%regression_type)
            ax3.set_title('With SNAC')

            # plot train/test histograms:
            fig, ax = plt.subplots(1, 1)
            ax, bins_saved = plot_histogram_ax(ax=ax, y=y.iloc[train_ids,:].values,xlabel=config['y_variable'], facecolor='blue', text_xy=(0.95, 0.95))
            ax = plot_histogram_ax(ax=ax, y=y.iloc[test_ids,:].values, xlabel=config['y_variable'], facecolor='orange', text_xy=(0.95, 0.75), bins=bins_saved)
            ax.grid()
            fig.savefig(sub_output_path + '%s_ytraintest_histograms.png' % regression_type)

            # predict new data if necessary
            if compounds_config[compounds]['model_type']['prediction']['apply']:
                # filter raw data
                logging.info('plotting pred/obs plots with y filtered within a range: %s'% compounds_config[compounds]['model_type']['prediction']['filter'])
                merged_Res_analyselabo_filtered = apply_filter(merged_Res_analyselabo_raw, compounds_config[compounds]['model_type']['prediction']['filter'])
                merged_Res_analyselabo_filtered = merged_Res_analyselabo_filtered.loc[pd.isnull(merged_Res_analyselabo_filtered['VFA_geq_x']) == True] # take all Nan (not computed by SNAC)
                X_regression, y_regression = data_raw_filter(merged_Res_analyselabo_filtered, X_vol, X_cond, config, custom_criteria, sub_output_path)
                fig3, ax1, ax2 = plot_specific_SRO(
                                            X=X_regression,
                                            y=y_regression,
                                            y_ref_pred = y.merge(merged_Res_analyselabo.loc[:,[y_ref_pred_snac]].drop_duplicates().dropna(), how='left', left_index=True, right_index=True).loc[:,[y_ref_pred_snac]],
                                            gscv = gscv,
                                            train_ids = None,
                                            test_ids = None,
                                            y_filter_range = None)
                ax1.set_title('With %s' %regression_type)
                ax2.set_title('With SNAC')
                fig3.savefig(sub_output_path + '%s_predobs_yrangefiltered.png'%regression_type)

            if regression_type == 'plsr' :  # only case where to do stacking for the moment:

                # run stacking plr
                y_train, y_test, y_train_pred_stacking, y_test_pred_stacking, stacking = run_stacking_PLS(X, y, y_ref_pred_snac,merged_Res_analyselabo,
                                                                                       train_ids, test_ids, config,regression_type, sub_output_path,
                                                                                       gscv, MBPLSRegression)

                # plot stacking figures
                plot_stacking_PLS(y_train, y_test, y_train_pred_stacking, y_test_pred_stacking, stacking, sub_output_path)

    else: # case where model is used to predict error (absolute/relative)
        gscv, (fig1, ax1), (fig2, ax2), train_ids, test_ids = run_regression(X=X, y=y, model=estimator,
                                                        test_size=config['regression'][regression_type]['cv']['test_size'],
                                                        train_test_split_method= config['regression'][regression_type]['cv']['train_test_split_method'],
                                                        cv=config['regression'][regression_type]['cv']['cv_splitter'],
                                                        gs_params=config['regression'][regression_type]['cv']['gs_params'],
                                                        ignored_samples=indip_samples_list
                                                        )

    plot_pls_epo(config, X, sub_output_path, regression_type, gscv, fig1, ax1, fig2, ax2, train_ids, test_ids)
    plt.close('all')

    # save data for article
    model_and_snac_data = pd.DataFrame()
    raw_X_PLS_data = pd.DataFrame()
    model_coeff = pd.DataFrame()
    stacking_coeff = pd.DataFrame(columns=['model', 'SNAC'])

    # conductivity raw signals
    if args.x_dataset_type == 'volume_dilutioncorrected':
        X_vol.loc[X.index, :].to_csv(sub_output_path_data_article + 'x_model_all_vol.csv', sep=';', decimal='.')
        X.to_csv(sub_output_path_data_article + 'x_model_pHfiltered_vol.csv', sep=';', decimal='.')
    elif args.x_dataset_type == 'conductivity_dilutioncorrected':
        X_cond.loc[X.index, :].to_csv(sub_output_path_data_article + 'x_model_all_cond.csv', sep=';', decimal='.')
        X.to_csv(sub_output_path_data_article + 'x_model_pHfiltered_cond.csv', sep=';', decimal='.')
    elif args.x_dataset_type == 'mb_volume_conductivity':
        X_vol.loc[X[0].index, :].to_csv(sub_output_path_data_article + 'x_model_all_vol.csv', sep=';', decimal='.')
        X_cond.loc[X[0].index, :].to_csv(sub_output_path_data_article + 'x_model_all_cond.csv', sep=';', decimal='.')
        X[0].to_csv(sub_output_path_data_article + 'x_model_pHfiltered_vol.csv', sep=';', decimal='.')
        X[1].to_csv(sub_output_path_data_article + 'x_model_pHfiltered_cond.csv', sep=';', decimal='.')

    if regression_type == 'plsr':
        X_train = X.iloc[train_ids, :]
        X_test = X.iloc[test_ids, :]
        model_and_snac_data['fichier_SNAC'] = merged_Res_analyselabo['0_1'][X.index]
        model_and_snac_data[compounds + '_labo'] = merged_Res_analyselabo['Labo'][X.index]
        model_and_snac_data[compounds + '_method'] = merged_Res_analyselabo['Method_'+compounds][X.index]
        model_and_snac_data['y_' + compounds + '_labo'] = merged_Res_analyselabo[compounds + '_geq_y'][X.index]
        model_and_snac_data['y_' + compounds + '_SNAC'] = merged_Res_analyselabo[compounds + '_geq_x'][X.index]
        model_and_snac_data['y_' + compounds + '_model_test'] = pd.DataFrame(index=X_test.index, columns=['test'],
                                                                             data=gscv.predict(X_test))
        model_and_snac_data['y_' + compounds + '_model_train'] = pd.DataFrame(index=X_train.index, columns=['train'],
                                                                              data=gscv.predict(X_train))
        model_and_snac_data['y_' + compounds + '_stacking_test'] = pd.DataFrame(index=X_test.index, columns=['test'],
                                                                             data=stacking.predict(X_test))
        model_and_snac_data['y_' + compounds + '_stacking_train'] = pd.DataFrame(index=X_train.index, columns=['train'],
                                                                              data=stacking.predict(X_train))
        model_and_snac_data.to_csv(sub_output_path_data_article + 'model_and_SNAC_data.csv', sep=';', decimal='.')

        model_coeff['pH'] = X.columns.values
        model_coeff['bcoeff_model_signal'] = gscv.best_estimator_.coef_
        model_coeff.to_csv(sub_output_path_data_article + 'model_coeff.csv', sep=';', decimal='.')

        stacking_coeff['model'] = stacking.final_estimator_.coef_[0]
        stacking_coeff['SNAC'] = stacking.final_estimator_.coef_[1]
        stacking_coeff.to_csv(sub_output_path_data_article + 'stacking_coeff.csv', sep=';', decimal='.')

    if regression_type == 'mbplsr':
        X_train = [X[0].iloc[train_ids, :],X[1].iloc[train_ids, :]] # [0] or [1] is the same since I just use the index afterwords
        X_test = [X[0].iloc[test_ids, :],X[1].iloc[test_ids, :]] # [0] or [1] is the same since I just use the index afterwords
        model_and_snac_data['fichier_SNAC'] = merged_Res_analyselabo['0_1'][X[0].index]
        model_and_snac_data[compounds + '_labo'] = merged_Res_analyselabo['Labo'][X[0].index]
        model_and_snac_data[compounds + '_method'] = merged_Res_analyselabo['Method_'+compounds][X[0].index]
        model_and_snac_data['y_' + compounds + '_labo'] = merged_Res_analyselabo[compounds + '_geq_y'][X[0].index] # [0] or [1] is the same since I just use the index
        model_and_snac_data['y_' + compounds + '_SNAC'] = merged_Res_analyselabo[compounds + '_geq_x'][X[0].index] # [0] or [1] is the same since I just use the index
        model_and_snac_data['y_' + compounds + '_model_test'] = pd.DataFrame(index=X_test[0].index, columns=['test'],# [0] or [1] is the same since I just use the index
                                                                             data=gscv.predict(X_test))
        model_and_snac_data['y_' + compounds + '_model_train'] = pd.DataFrame(index=X_train[0].index, columns=['train'],# [0] or [1] is the same since I just use the index
                                                                              data=gscv.predict(X_train))
        model_and_snac_data['y_' + compounds + '_stacking_test'] = pd.DataFrame(index=X_test[0].index, columns=['test'],# [0] or [1] is the same since I just use the index
                                                                             data=y_test_pred_stacking)
        model_and_snac_data['y_' + compounds + '_stacking_train'] = pd.DataFrame(index=X_train[0].index, columns=['train'],# [0] or [1] is the same since I just use the index
                                                                              data=y_train_pred_stacking)
        model_and_snac_data.to_csv(sub_output_path_data_article + 'model_and_SNAC_data.csv', sep=';', decimal='.')


        model_coeff['pH'] = X[0].columns.values
        beta_coeff = np.array_split(gscv.best_estimator_.beta_, 2)
        # model_coeff['bcoeff_model_vol'] = model_coeff.apply(lambda x:1,axis=1)
        model_coeff['bcoeff_model_vol'] = beta_coeff[0]
        model_coeff['bcoeff_model_cond'] = beta_coeff[1]
        model_coeff.to_csv(sub_output_path_data_article + 'model_coeff.csv', sep=';', decimal='.')

        stacking_coeff['model'] = stacking.final_estimator_.coef_[0]
        stacking_coeff['SNAC'] = stacking.final_estimator_.coef_[1]
        stacking_coeff.to_csv(sub_output_path_data_article + 'stacking_coeff.csv', sep=';', decimal='.')


    # Eporting model
    logging.info('Exporting models (pickling)')
    best_estimator = gscv.best_estimator_
    output_fn = '%s%s_estimator.pk' % (sub_output_path, regression_type)
    logging.info('estimator to pickle:%s' % (output_fn))
    import pickle as pk
    with open(output_fn, 'wb') as f:
        f.write(pk.dumps(best_estimator))

    if isinstance(best_estimator, PLSRegression):
        logging.info('Exporting model parameters (json)')
        output_fn = '%s%s_estimator_params_%s.json'%(sub_output_path, regression_type,compounds)
        logging.info('bcoefficients, _x_mean, _x_std, _y_mean to json:%s'%(output_fn))
        json_dict = dict()
        json_dict['coef_'] = best_estimator.coef_.flatten().tolist()
        json_dict['_x_mean'] = best_estimator._x_mean.flatten().tolist()
        json_dict['_x_std'] = best_estimator._x_std.flatten().tolist()
        json_dict['_y_mean'] = best_estimator._y_mean.flatten().tolist()
        json_dict['x_rotations_'] = best_estimator.x_rotations_.tolist()
        json_dict['x_loadings_'] = best_estimator.x_loadings_.tolist()
        json_dict['n_components'] = best_estimator.n_components.flatten().tolist()
        json_dict['scores_std'] = best_estimator.scores_std.flatten().tolist()
        json_dict['cutoff_od'] = best_estimator.cutoff_od.flatten().tolist()
        json_dict['cutoff_sd'] = best_estimator.cutoff_sd.flatten().tolist()
        json_dict['pH_range'] = [X.columns[0],X.columns[-1]] # better have it as a parameter because it could change
        json_dict['pH_vector_size'] = [X.columns.size] # need it for good interpolation of raw data
        json_dict['model'] = [compounds_config[compounds]['model_type']['regression']['type']] # better have it as a parameter because it could change
        json_dict['signal'] = [config['x_dataset_type']]
        json_dict['input_type'] = [compounds_config[compounds]['preprocessing']['type_data']]
        json_dict['compounds'] = [compounds]

        with open(output_fn, 'w') as f:
            f.write(json.dumps(json_dict))

    # # test use PLS model written
    # from PLS_test import *
    # import json
    #
    # # Opening JSON file
    # f = open(os.path.join(sub_output_path,'plsr_estimator_params_VFA.json'))
    #
    # data_json = json.load(f)
    #
    # # Iterating through the json
    # data_json = {k: np.array(v) for k, v in data_json.items()}
    # # info['length'] = data_json['coef_'].size
    # model = PLS_analysis()
    # model.__dict__.update(data_json)
    # f.close()
    #
    # y_model_vrai = gscv.predict(X_test)
    # y_model_reconstitue = model.predict(X_test)
    # pareil = (y_model_vrai[:,0] == y_model_reconstitue).all()
    #
    # # read input X of SNAC code
    # X_reconstitue = pd.read_csv(input_path + '/file_test',index_col=0, header=0, sep=';', decimal='.')
    # file_name = merged_Res_analyselabo[merged_Res_analyselabo['0_1'] == '2019-11-20_05_SNAC061901_src.csv'].index[0]
    # X_vrai = pd.DataFrame(X.loc[file_name, :])
    #
    # plt.plot(X_vrai)
    # plt.plot(X_vrai.index, X_reconstitue.iloc[:, 0])

    logging.info('----------------- Run end ---------------------')

