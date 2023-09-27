import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import transforms, cm
from matplotlib.patches import Ellipse, Rectangle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import logging
from sklearn.linear_model import LinearRegression, QuantileRegressor
# from datasciencetools.metrics import *
from matplotlib import ticker
# from data_filter import *
import os
from pathlib import Path
# from datasciencetools.plot import *
import matplotlib.pyplot as plt
from scipy import stats
from tool import *
import numpy as np

# set paths
# folder_name = str('T21')
# input_path ="/home/silvio/Dropbox (BioEnTech)/CORPORATE/0 ORGANISATION DROPBOX CORPORATE/12-SNAC/2-Developpement/1-Developpement algo/4 - Analyse stat/"+folder_name
# output_path ="/home/silvio/Dropbox (BioEnTech)/CORPORATE/0 ORGANISATION DROPBOX CORPORATE/12-SNAC/2-Developpement/1-Developpement algo/4 - Analyse stat/"+folder_name+'/Article_graphs'

dpi_value = 80
fontsize = 18
fontsize_tick = 14
fontsize_legend =12
fontname='Calibri'

# color by matrix -----------------------
T1 = 'aqua'
T2 = 'deeppink'
T3 = 'lawngreen'
T4 = 'gold'

FOS_color = 'tomato'
TAC_color = 'yellowgreen'
IC_color = 'turquoise'
VFA_color = 'brown'
TAN_color = 'dodgerblue'

#------------------------


# color
bis_color = 'black'
accept_lim1= 'green'
accept_lim2= 'blue'
accept_lim3= 'red'
lin_regr_color = 'k'
mark1='-.'
mark2='-.'
mark3='-.'
mark4='-'
value_lim1= 0.2
value_lim2= 0.25
value_lim3= 0.30

def plot_Benchmark_param(data_dict, path, Res_SNAC_raw,linear_regression = None):

    dict_param_relation = {'VFA':'VFA',# param: ref ; this is important to create graphs from good data dictionary
                           'FOS':'VFA',
                           'corr_FOS': 'VFA',
                           'corr_Hach_FOS': 'VFA',
                           'pls_VFA':'VFA',
                           'sep_VFA1': 'VFA',
                           'sep_VFA2': 'VFA',
                           'TAN':'TAN',
                           'pls_TAN':'TAN',
                           'sep_TAN': 'TAN',
                           'TAC':'TAC',
                           'IC':'TAC'
                           }

    param_to_plot = {
    'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'uniform', 'lr': linear_regression},
    'TAN': {'unit': 'gN L' + get_super('-1'), 'color': TAN_color, 'test_color': 'uniform', 'lr': linear_regression},
    'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform', 'lr': linear_regression},
    'pls_TAN': {'unit': 'gN L' + get_super('-1'), 'X': 'TAN_ref', 'color': TAN_color, 'test_color': 'uniform', 'lr': linear_regression},
    'sep_VFA1': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform','lr': linear_regression},
    'sep_VFA2': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform','lr': linear_regression},
    'sep_TAN': {'unit': 'gN L' + get_super('-1'), 'X': 'TAN_ref', 'color': TAN_color, 'test_color': 'uniform','lr': linear_regression},
    'TAC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'uniform','lr': linear_regression},
    'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'FOS_ref', 'Y': 'FOS_geq', 'color': FOS_color,'test_color': 'uniform', 'title': 'FOS ref','lr': linear_regression},
    # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': FOS_color,'test_color': 'uniform', 'title': 'VFA ref','lr': linear_regression},
    'IC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'uniform','lr': linear_regression}
                    }
    # all graphs
    for param in param_to_plot:
        standard_plot_new(data_dict[dict_param_relation[param]], param_to_plot, param, path, 'All_conc_')


    ###### VFA zoom
    param_to_plot = {
        'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'uniform'},
        'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color, 'test_color': 'uniform','xy_lim':[2,2]},
        'sep_VFA1': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color, 'test_color': 'uniform'},
        'sep_VFA2': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color, 'test_color': 'uniform'},
                    }
    for param in param_to_plot:
        data_selection = data_dict[dict_param_relation[param]].copy(deep=True)
        data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]
        standard_plot_new(data_selection, param_to_plot, param, path, 'All_conc_zoom_')

    ###### profil exactitude
    param_to_plot = {
        'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'uniform'},
        'TAN': {'unit': 'gN L' + get_super('-1'), 'color': TAN_color, 'test_color': 'uniform'},
        'TAC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'uniform'},
        'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform'},
        'pls_TAN': {'unit': 'gN L' + get_super('-1'), 'X': 'TAN_ref', 'color': TAN_color,'test_color': 'uniform'},
        'sep_VFA1': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform'},
        'sep_VFA2': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform'},
        # 'corr_FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': FOS_color,'test_color': 'uniform', 'linear_correction': [1.33,1.01]},
        # 'corr_Hach_FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': FOS_color, 'test_color': 'uniform', 'linear_correction': [0.81, 1.16]},
        'sep_TAN': {'unit': 'gN L' + get_super('-1'), 'X': 'TAN_ref', 'color': TAN_color,'test_color': 'uniform'},
                     }
    for param in param_to_plot:
        data_selection = data_dict[dict_param_relation[param]]
        plot_exactitude_general(data_selection, param_to_plot, param, path, 'All_conc_AP_')

    ###### zoom profil exactitude
    param_to_plot = {'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'uniform'},
                     'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,
                                 'test_color': 'uniform'},
                     'sep_VFA1': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,
                                 'test_color': 'uniform'},
                     'sep_VFA2': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,
                                 'test_color': 'uniform'},
                        }

    for param in param_to_plot:
        data_selection = data_dict[dict_param_relation[param]].copy(deep=True)
        data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]
        plot_exactitude_general(data_selection, param_to_plot, param, path, 'All_conc_AP_zoom_')



    ##### FOS analysis based on VFA ref
    param_to_plot = {
        # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': FOS_color,
        #         'test_color': 'uniform', 'titre_supp': '_(VFA ref)'},
        'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': FOS_color,
                'test_color': 'uniform', 'titre_supp': '_(VFA ref)', 'linear_regression': 'yes'},
        # 'corr_FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'corr_FOS_geq', 'color': FOS_color,
        #         'test_color': 'uniform', 'titre_supp': '_(VFA ref)', 'linear_correction': [1.33,1.01]},
        # 'corr_Hach_FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'corr_Hach_FOS_geq', 'color': FOS_color,
        #         'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)', 'linear_correction': [0.81, 1.16]}
    }
    for param in param_to_plot:
        standard_plot_new(data_dict[dict_param_relation[param]], param_to_plot, param, path, 'All_conc_')


    ###### FOS analysis (Hach) based on VFA ref
    param_to_plot = {
        #'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': FOS_color,
         #       'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)'},
         # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': FOS_color,
         #         'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)', 'linear_regression': 'yes'},
         'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': FOS_color,
                 'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)', 'lr': linear_regression},
    }
    for param in param_to_plot:
        FOS_Hach_plot(data_dict[dict_param_relation[param]], param_to_plot, param, path, 'All_conc_')

    # #### FOS zoom
    param_to_plot = {
        # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': FOS_color,
        #         'test_color': 'uniform', 'titre_supp': '_(VFA ref)', 'xy_lim':[2,2]},
        'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': FOS_color,
                'test_color': 'uniform', 'titre_supp': '_(VFA ref)', 'linear_regression': 'yes', 'xy_lim':[2,2]},
        # 'corr_FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'corr_FOS_geq', 'color': FOS_color,
        #         'test_color': 'uniform', 'titre_supp': '_(VFA ref)', 'linear_correction': [1.33,1.01], 'xy_lim':[1.2,1.2]},
        # 'corr_Hach_FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'corr_Hach_FOS_geq','color': FOS_color,
        #                   'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)', 'linear_correction': [0.81, 1.16]}
    }

    for param in param_to_plot:
        data_selection = data_dict[dict_param_relation[param]].copy(deep=True)
        data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]
        standard_plot_new(data_selection, param_to_plot, param, path, 'All_conc_zoom_')

    param_to_plot = {
        # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': FOS_color,
        #         'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)', 'xy_lim':[2,2]},
        # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': FOS_color,
        #         'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)', 'linear_regression': 'yes', 'xy_lim':[2,2]},
        'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': FOS_color,
                'test_color': 'uniform', 'titre_supp': '_(Hach-VFA_ref)', 'lr': linear_regression, 'xy_lim': [2, 2]},
                    }
    for param in param_to_plot:
        data_selection = data_dict[dict_param_relation[param]].copy(deep=True)
        data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]
        FOS_Hach_plot(data_selection, param_to_plot, param, path, 'All_conc_zoom_')


def plot_Benchmark(data, path, Res_SNAC_raw):

    param_to_plot = {
                     'VFA':{'unit':'gAc_eq L'+get_super('-1'), 'color':VFA_color, 'test_color':'different'},
                     'TAN':{'unit':'gN L'+get_super('-1'), 'color':TAN_color, 'test_color':'different'},
                     'pls_VFA':{'unit':'gAc_eq L'+get_super('-1'),'X': 'VFA_ref', 'color':VFA_color, 'test_color':'different'},
                     'pls_TAN':{'unit':'gN L'+get_super('-1'),'X': 'TAN_ref', 'color':TAN_color, 'test_color':'different'},
                     'sep_VFA1':{'unit':'gAc_eq L'+get_super('-1'),'X': 'VFA_ref', 'color':VFA_color, 'test_color':'different'},
                     'sep_VFA2':{'unit':'gAc_eq L'+get_super('-1'),'X': 'VFA_ref', 'color':VFA_color, 'test_color':'different'},
                     'sep_TAN':{'unit':'gN L'+get_super('-1'),'X': 'TAN_ref', 'color':TAN_color, 'test_color':'different'},
                     'TAC':{'unit':'gCaCO'+get_sub('3')+'_eq L'+get_super('-1'), 'color':TAC_color, 'test_color':'different'},
                     'FOS':{'unit':'gAc_eq L'+get_super('-1'), 'X':'FOS_ref', 'Y':'FOS_geq', 'color':FOS_color, 'test_color':'different','title': 'FOS ref', 'linear_regression':'yes'},
                     'IC':{'unit':'gCaCO'+get_sub('3')+'_eq L'+get_super('-1'), 'color':TAC_color, 'test_color':'different'}
                     }

    ###### all graphs
    for param in param_to_plot:
        standard_plot_new(data, param_to_plot, param, path, 'T_all_')


    param_to_plot = {
                     'VFA':{'unit':'gAc_eq L'+get_super('-1'), 'color':VFA_color, 'test_color':'uniform'},
                     'TAN':{'unit':'gN L'+get_super('-1'), 'color':TAN_color, 'test_color':'uniform'},
                     'pls_VFA':{'unit':'gAc_eq L'+get_super('-1'),'X': 'VFA_ref', 'color':VFA_color, 'test_color':'uniform'},
                     'pls_TAN':{'unit':'gN L'+get_super('-1'),'X': 'TAN_ref', 'color':TAN_color, 'test_color':'uniform'},
                     'sep_VFA1': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform'},
                     'sep_VFA2': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'uniform'},
                     'sep_TAN': {'unit': 'gN L' + get_super('-1'), 'X': 'TAN_ref', 'color': TAN_color, 'test_color': 'uniform'},
                     'TAC':{'unit':'gCaCO'+get_sub('3')+'_eq L'+get_super('-1'), 'color':TAC_color, 'test_color':'uniform'},
                     'FOS':{'unit':'gAc_eq L'+get_super('-1'), 'X':'FOS_ref', 'Y':'FOS_geq', 'color':FOS_color, 'test_color':'uniform','title': 'VFA ref'},
                     'IC':{'unit':'gCaCO'+get_sub('3')+'_eq L'+get_super('-1'), 'color':TAC_color, 'test_color':'uniform'}
                     }

    ###### all graphs
    for param in param_to_plot:
        for Tx in data.index.get_level_values(0).drop_duplicates():
            name = Tx + '_'
            data_selection = data.loc[Tx, :]  # we select data
            standard_plot_new(data_selection, param_to_plot, param, path, name)


    ##### specific graph - crossed data #####

    ##### influence TAC on VFA and TAN
    selected_tests = ['T1_5', 'T2_1', 'T2_2', 'T2_3']
    data_selection = data.copy(deep=True)
    data_selection = data_selection.droplevel('Matrix').loc[selected_tests, :]
    # data_selection = data_selection.loc[pd.IndexSlice[:, selected_tests], :] # if I keep the multiindex
    param_to_plot = {'VFA':{'unit':'gAc_eq L'+get_super('-1'), 'color':VFA_color, 'libelled': data_selection.loc[:,'Ref_res_mean'].loc[:,'TAC_ref'], 'test_color':'uniform'},
                     'TAN':{'unit':'gN L'+get_super('-1'), 'color':TAN_color, 'libelled': data_selection.loc[:,'Ref_res_mean'].loc[:,'TAC_ref'],'test_color':'uniform'}
                     }
    for param in param_to_plot:
        standard_plot_new(data_selection, param_to_plot, param, path, 'TAC influence on ')

    # # influence TAN on VFA
    # selected_tests = ['T2_2', 'T3_5', 'T4_2']
    # data_selection = data.copy(deep=True)
    # data_selection = data_selection.droplevel('Matrix').loc[selected_tests, :]
    # # data_selection = data_selection.loc[pd.IndexSlice[:, selected_tests], :] # if I keep the multiindex
    # param_to_plot = {'VFA':{'unit':'gAc_eq L'+get_super('-1'), 'color':VFA_color, 'libelled': data_selection.loc[:,'Ref_res_mean'].loc[:,'TAN_ref']},
    #                  # 'TAN':{'unit':'gN L'+get_super('-1'), 'color':TAN_color, 'libelled': data_selection.loc[:,'Ref_res_mean'].loc[:,'TAC_ref']}
    #                  }
    # for i in param_to_plot:
    #     standard_plot(data_selection, param_to_plot, i, path, 'TAN influence on ')


    ###### VFA zoom
    data_selection = data.copy(deep=True)
    data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]
    param_to_plot = {'VFA':{'unit':'gAc_eq L'+get_super('-1'), 'color':VFA_color, 'test_color':'different'},
                     'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'different'},
                     'sep_VFA1': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'different'},
                     'sep_VFA2': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'different'},
                     }
    # graphs with all Tx
    for param in param_to_plot:
        standard_plot_new(data_selection, param_to_plot, param, path, 'T_all_zoom_')

    # graphs per Tx
    data_selection = data.copy(deep=True)
    # data_selection = data_selection.droplevel('Matrix')
    data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]
    param_to_plot = {'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'different'},
                     'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'different'}}

    for param in param_to_plot:
        for Tx in data_selection.index.get_level_values(0).drop_duplicates():
            name = Tx+'_zoom_'
            standard_plot_new(data_selection.loc[Tx,:], param_to_plot, param, path, name)


    ##### FOS analysis based on VFA ref
    param_to_plot= {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': VFA_color, 'test_color':'different','titre_supp': '_(VFA ref)'}}
    for i in param_to_plot:
        standard_plot_new(data, param_to_plot, i, path, 'T_all_')

    param_to_plot = {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': VFA_color,'test_color': 'uniform','titre_supp': '_(VFA ref)'}}
    for i in param_to_plot:
        for ii in data.index.get_level_values(0).drop_duplicates():
            name = ii + '_'
            data_plot = data.loc[ii, :]  # we select data
            standard_plot_new(data_plot, param_to_plot, i, path, name)

    plt.close('all')

    ###### FOS analysis (Hach) based on VFA ref
    param_to_plot= {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': VFA_color, 'test_color':'different','titre_supp': '_(Hach-VFA_ref)'}}
    for i in param_to_plot:
        FOS_Hach_plot(data, param_to_plot, i, path, 'T_all_') # need another function because
        # standard_plot_new(data, param_to_plot, i, path, 'T_all_')

    param_to_plot = {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': VFA_color,'test_color': 'uniform','titre_supp': '_Hach_(VFA_ref)'}}
    for i in param_to_plot:
        for ii in data.index.get_level_values(0).drop_duplicates():
            name = ii + '_'
            data_plot = data.loc[ii, :]  # we select data
            FOS_Hach_plot(data_plot, param_to_plot, i, path, name)
    plt.close('all')

    #### FOS zoom
    data_selection = data.copy(deep=True)
    data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]
        ##### FOS analysis based on VFA ref
    param_to_plot= {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': VFA_color, 'test_color':'different','titre_supp': '_(VFA ref)_zoom'}}
    for i in param_to_plot:
        standard_plot_new(data_selection, param_to_plot, i, path, 'T_all_')

    param_to_plot = {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_geq', 'color': VFA_color,'test_color': 'uniform','titre_supp': '_(VFA ref)_zoom'}}
    for i in param_to_plot:
        for Tx in data_selection.index.get_level_values(0).drop_duplicates():
            name = Tx + '_'
            data_plot = data_selection.loc[Tx, :]  # we select data
            standard_plot_new(data_plot, param_to_plot, i, path, name)

        ###### FOS analysis (Hach) based on VFA ref
    param_to_plot= {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': VFA_color, 'test_color':'different','titre_supp': '_Hach_(VFA_ref)_zoom'}}
    for param in param_to_plot:
        FOS_Hach_plot(data_selection, param_to_plot, param, path, 'T_all_')

    param_to_plot = {'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'Y': 'FOS_ref', 'color': VFA_color,'test_color': 'uniform','titre_supp': '_Hach_(VFA_ref)_zoom'}}
    for i in param_to_plot:
        for Tx in data_selection.index.get_level_values(0).drop_duplicates():
            name = Tx + '_'
            data_plot = data_selection.loc[Tx, :]  # we select data
            FOS_Hach_plot(data_plot, param_to_plot, i, path, name)
    plt.close('all')


    ###### TAC en fonction VFA
    data_selection = data.copy(deep=True)
    param_to_plot = {'IC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'libelled': data_selection.loc[:, 'Ref_res_mean'].loc[:, 'VFA_ref'], 'test_color': 'uniform'}}
    for param in param_to_plot:
        for Tx in data_selection.index.get_level_values(0).drop_duplicates():
            name = Tx + '_VFA influence on '
            data_plot = data_selection.loc[Tx, :]
            param_to_plot[param]['libelled'] = data_selection.loc[:, 'Ref_res_mean'].loc[Tx, 'VFA_ref']
            # we select data
            standard_plot_new(data_plot, param_to_plot, param, path, name)

    ###### TAC en fonction TAN
    data_selection = data.copy(deep=True)
    param_to_plot = {'IC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color,
                            'libelled': data_selection.loc[:, 'Ref_res_mean'].loc[:, 'TAN_ref'],
                            'test_color': 'uniform'}
                     }
    for param in param_to_plot:
        for Tx in data_selection.index.get_level_values(0).drop_duplicates():
            name = Tx + '_TAN influence on '
            data_plot = data_selection.loc[Tx, :]
            param_to_plot[param]['libelled'] = data_selection.loc[:, 'Ref_res_mean'].loc[Tx, 'TAN_ref']
            # we select data
            standard_plot_new(data_plot, param_to_plot, param, path, name)




    ###### variation pka en focntion conductivité intiiale
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
    ax.scatter(Res_SNAC_raw['202_pKa_acetatic/acetate'], Res_SNAC_raw['conductivity_initial'])
    save_name = 'pka_VFA = f(conductivity_i)'
    ax.set_title(save_name, fontsize=fontsize, fontname=fontname)
    ax.set_xlabel('pka_VFA', fontsize=fontsize, fontname=fontname)
    ax.set_ylabel('conductivity', fontsize=fontsize, fontname=fontname)
    ax.legend(loc='upper left')
    fig.savefig(path + '/image'+ '/' + save_name + '.png')
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
    ax.scatter(Res_SNAC_raw['202_pKa_NH4/NH3'], Res_SNAC_raw['conductivity_initial'])
    save_name = 'pka_TAN = f(conductivity_i)'
    ax.set_title(save_name, fontsize=fontsize, fontname=fontname)
    ax.set_xlabel('pka_TAN', fontsize=fontsize, fontname=fontname)
    ax.set_ylabel('conductivity', fontsize=fontsize, fontname=fontname)
    ax.legend(loc='upper left')
    fig.savefig(path + '/image'+ '/' + save_name + '.png')
    plt.close()


    ###### profil exactitude
    param_to_plot = {
        'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'different'},
        'TAN': {'unit': 'gN L' + get_super('-1'), 'color': TAN_color, 'test_color': 'different'},
         'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'different'},
         'pls_TAN': {'unit': 'gN L' + get_super('-1'), 'X': 'TAN_ref', 'color': TAN_color, 'test_color': 'different'},
        # 'TAC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'different'},
        # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'FOS_ref', 'Y': 'FOS_geq', 'color': FOS_color,
        #         'test_color': 'different'},
        # 'IC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'different'}
    }

    for param in param_to_plot:
        if 'SNAC_res_Rf' in data: # condition to be sure to have the data we need . Chose of Rf since variable in accuracy profile
            plot_exactitude_general(data, param_to_plot, param, path, 'T_all_AP_')
            for Tx in data.index.get_level_values(0).drop_duplicates():
                name = Tx + '_AP_'
                data_plot = data.loc[Tx, :]  # we select data
                plot_exactitude_general(data_plot, param_to_plot, param, path, name)

    ###### zoom profil exactitude
    param_to_plot = {
        'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'different'},
         'pls_VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'VFA_ref', 'color': VFA_color,'test_color': 'different'},
        # 'TAC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'different'},
        # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'FOS_ref', 'Y': 'FOS_geq', 'color': FOS_color,
        #         'test_color': 'different'},
        # 'IC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'different'}
    }

    data_selection = data.copy(deep=True)
    data_selection = data_selection[data_selection.loc[:, 'Ref_res_mean']['VFA_ref'] <= 1.0]

    if 'SNAC_res_Rf' in data: # condition to be sure to have the data we need . Chose of Rf since variable in accuracy profile
        for param in param_to_plot:
            plot_exactitude_general(data_selection, param_to_plot, param, path, 'T_all_AP_zoom_')
            # for Tx in data.index.levels[0]:
            #     name = Tx + '_AP_zoom'
            #     data_plot = data.loc[Tx, :]  # we select data
            #     plot_exactitude_general(data_plot, param_to_plot, param, path, name)


    # #### box plot

    # param_to_plot = {
    #     'VFA': {'unit': 'gAc_eq L' + get_super('-1'), 'color': VFA_color, 'test_color': 'different'},
    #     'TAN': {'unit': 'gN L' + get_super('-1'), 'color': TAN_color, 'test_color': 'different'},
    #     # 'TAC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'different'},
    #     # 'FOS': {'unit': 'gAc_eq L' + get_super('-1'), 'X': 'FOS_ref', 'Y': 'FOS_geq', 'color': FOS_color,
    #     #         'test_color': 'different'},
    #     # 'IC': {'unit': 'gCaCO'+get_sub('3')+'_eq L' + get_super('-1'), 'color': TAC_color, 'test_color': 'different'}
    # }
    #

    # for param in param_to_plot:
    #     #box_plot(data, Res_SNAC_twoindex, param_to_plot, param, path, 'T_all_PE_')
    #     for Tx in data.index.levels[0]:
    #         name = Tx + '_BP_'
    #         data_plot = data.loc[Tx, :]  # we select data
    #         box_plot(data_plot, Res_SNAC_twoindex, param_to_plot, param, path, name)

def standard_plot_new(data,dict_param,param,path,save_name):
    param_ref = param # we do that because the IC has no reference values and we use TAC for comparison
    if param == 'IC'or param == 'TAC':
        param_ref = 'TAC'
    Y = param + '_geq'
    X = param_ref + '_ref'
    if 'X' in dict_param[param]:
        X = dict_param[param]['X']
    if 'Y' in dict_param[param]:
        Y = dict_param[param]['Y']

    if Y in data.loc[:, 'SNAC_res_mean'].columns:
        if not ((data.loc[:, 'Ref_res_mean'][X].isna().all()) or (data.loc[:, 'SNAC_res_mean'][Y].isna().all())): # condition to avoid error with missing data
            if 'titre_supp' in dict_param[param]:
                title_save = save_name + param + dict_param[param]['titre_supp']
                title ='SNAC/Ref Comparison - ' + param
            else:
                title_save = save_name + param
                title ='SNAC/Ref Comparison - ' + param


            fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
            ref = data.loc[:, 'Ref_res_mean'][X]
            ref_std = data.loc[:, 'Ref_res_std'][X]
            mean = data.loc[:, 'SNAC_res_mean'][Y]
            std = data.loc[:, 'SNAC_res_std'][Y]
            median = data.loc[:, 'SNAC_res_median'][Y]

            max_bisector = max(ref.max(), mean.max())
            ax.plot([0, max_bisector], [0, max_bisector], '--', color=bis_color, label='bisector', lw=1)

            # ranges

            if (param in ['VFA', 'TAN', 'pls_VFA','pls_TAN', 'sep_VFA1', 'sep_VFA2','sep_TAN', 'corr_FOS']):
                value_lim = 0.2
                coeff = 0.2 * max_bisector  # 20 %
                ax.plot([1 - value_lim, max_bisector - coeff], [1, max_bisector], mark1, color=accept_lim1, label='interval 20%', lw=1)
                ax.plot([0, 1 - value_lim], [value_lim, 1], mark1, color=accept_lim1, lw=1)
                ax.plot([1, max_bisector], [1 - value_lim, max_bisector - coeff], mark1, color=accept_lim1, lw=1)
                ax.plot([value_lim, 1], [0, 1 - value_lim], mark1, color=accept_lim1, lw=1)

                value_lim = 0.25
                coeff = 0.25 * max_bisector  # 25 %
                ax.plot([1 - value_lim, max_bisector - coeff], [1, max_bisector], mark1, color=accept_lim2, label='interval 25%', lw=1)
                ax.plot([0, 1 - value_lim], [value_lim, 1], mark1, color=accept_lim2, lw=1)
                ax.plot([1, max_bisector], [1 - value_lim, max_bisector - coeff], mark1, color=accept_lim2, lw=1)
                ax.plot([value_lim, 1], [0, 1 - value_lim], mark1, color=accept_lim2, lw=1)

                value_lim = 0.30
                coeff = 0.30 * max_bisector  # 25 %
                ax.plot([1 - value_lim, max_bisector - coeff], [1, max_bisector], mark1, color=accept_lim3, label='interval 30%', lw=1)
                ax.plot([0, 1 - value_lim], [value_lim, 1], mark1, color=accept_lim3, lw=1)
                ax.plot([1, max_bisector], [1 - value_lim, max_bisector - coeff], mark1, color=accept_lim3, lw=1)
                ax.plot([value_lim, 1], [0, 1 - value_lim], mark1, color=accept_lim3, lw=1)
            else:
                coeff = 0.2 * max_bisector  # 20 %
                ax.plot([0,max_bisector - coeff], [0, max_bisector], mark1, color=accept_lim1, label='interval 20%', lw=1)
                ax.plot([0, max_bisector], [0, max_bisector-coeff], mark1, color=accept_lim1, lw=1)

                coeff = 0.25 * max_bisector  # 25 %
                ax.plot([0,max_bisector - coeff], [0, max_bisector], mark1, color=accept_lim2, label='interval 25%', lw=1)
                ax.plot([0, max_bisector], [0, max_bisector-coeff], mark1, color=accept_lim2, lw=1)

                coeff = 0.30 * max_bisector  # 30 %
                ax.plot([0,max_bisector - coeff], [0, max_bisector], mark1, color=accept_lim3, label='interval 30%', lw=1)
                ax.plot([0, max_bisector], [0, max_bisector-coeff], mark1, color=accept_lim3, lw=1)


            if (dict_param[param]['test_color'] == 'different') and (len(data.index.names)>1):
                count_median = 0
                count_std = 0
                for i in data.index.get_level_values(0).drop_duplicates():
                    ref_matrix = ref.loc[i,:]
                    mean_matrix =mean.loc[i,:]
                    std_matrix = std.loc[i,:]
                    median_matrix = median.loc[i, :]
                    ref_std_matrix = ref_std.loc[i,:]

                    # just to plot the legend of only one median
                    if count_median == 0:
                        ax.scatter(ref_matrix, median_matrix, marker='+', color='red', label=' median')
                        count_median = count_median + 1
                    else:
                        ax.scatter(ref_matrix, median_matrix, marker='+', color='red')

                    ax.scatter(ref_matrix, mean_matrix, color=globals()[i], label=i+' mean')

                    # just to plot the legend of only one std
                    if count_std == 0:
                        if not std_matrix.isna().all():
                            ax.errorbar(ref_matrix, mean_matrix, yerr=std_matrix, fmt=",", color='dimgrey', label='std', markersize=8, capsize=3, elinewidth=1)
                        if not ref_std.isna().all():  # if they are not all nan we plot it ( we have error only for FOS and TAC because made with the Hach fostac)
                            ax.errorbar(ref_matrix, mean_matrix, xerr=ref_std_matrix, fmt=",", color='dimgrey',label='std', markersize=8, capsize=3, elinewidth=1)
                        count_std = count_std + 1
                    else:
                        if not std_matrix.isna().all():
                            ax.errorbar(ref_matrix, mean_matrix, yerr=std_matrix, fmt=",", color='dimgrey',markersize=8, capsize=3, elinewidth=1)
                        if not ref_std.isna().all():  # if they are not all nan we plot it ( we have error only for FOS and TAC because made with the Hach fostac)
                            ax.errorbar(ref_matrix, mean_matrix, xerr=ref_std_matrix, fmt=",", color='dimgrey', markersize=8, capsize=3, elinewidth=1)

                    # plot the regression model
                    if 'linear_regression' in dict_param[param]:
                        reg, R2, coef, intercept = linear_regression(x=ref_matrix, y=mean_matrix)
                        X_regr = np.linspace(min(ref_matrix)*0.9, max(ref_matrix)*1.1, num=50)
                        X_regr = X_regr.reshape(X_regr.size, 1)
                        ax.plot(X_regr, reg.predict(X_regr), mark4, color=lin_regr_color, lw=2,
                                label='y=' + str(np.around(coef[0][0], decimals=2)) + 'x + ' + str(
                                    np.around(intercept[0], decimals=2)) + ', R2=' + str(np.around(R2, decimals=2))
                                )
                        ax.text(0.98, 0.02,
                                'lin. regr:\n$R^{2}$=%.2f\ny=%.2fx + %.2f' % (R2, coef[0][0], intercept[0]), ha='right',
                                va='bottom', transform=ax.transAxes, fontsize = fontsize_legend)

                    if 'lr' in dict_param[param]:
                        if dict_param[param]['lr'] == None:
                            print('no linear regression active')
                        elif dict_param[param]['lr'] == 'yes':
                            print('linear regression active')
                        elif dict_param[param]['lr'] == 'check':
                            reg, R2, a, b = linear_regression(x=ref_matrix, y=mean_matrix)
                            X_regr = np.linspace(min(ref_matrix) * 0.9, max(ref_matrix) * 1.1, num=50)
                            X_regr = X_regr.reshape(X_regr.size, 1)
                            ax.plot(X_regr, reg.predict(X_regr), mark4, color=lin_regr_color, lw=2,
                                    label='y=' + str(np.around(a, decimals=2)) + 'x + ' + str(
                                        np.around(b, decimals=2)) + ', R2=' + str(np.around(R2, decimals=2))
                                    )
                            print('linear regression active')
                        else: #we have a dictionary
                            index = param_ref + '-' + Y
                            a = dict_param['lr'].loc[index, 'a']
                            b = dict_param['lr'].loc[index, 'b']
                            R2 = dict_param['lr'].loc[index, 'R2']

                            X_regr = np.linspace(min(ref_matrix)*0.9, max(ref_matrix)*1.1, num=50)
                            X_regr = X_regr.reshape(X_regr.size, 1)
                            ax.plot(X_regr, a*X_regr+b, mark4, color=lin_regr_color, lw=2,
                                    label='y=' + str(np.around(a, decimals=2)) + 'x + ' + str(
                                        np.around(b, decimals=2)) + ', R2=' + str(np.around(R2, decimals=2))
                                    )
                        ax.text(0.98, 0.02,
                                'lin. regr:\n$R^{2}$=%.2f\ny=%.2fx + %.2f' % (R2, a, b), ha='right',
                                va='bottom', transform=ax.transAxes, fontsize = fontsize_legend)


                    if 'libelled' in dict_param[param]:
                        for xx in dict_param[param]['libelled'].index:
                            x = ref[xx]
                            y = mean[xx]
                            label = truncate_value(dict_param[param]['libelled'][xx],2)
                            ax.annotate(label,  # this is the text
                                        (x, y),  # these are the coordinates to position the label
                                        textcoords="offset points",  # how to position the text
                                        xytext=(0, 10),  # distance from text to points (x,y)
                                        ha='right')

            else:

                if 'linear_correction' in dict_param[param]:
                    # mean = (mean - dict_param[param]['linear_correction'][1])/dict_param[param]['linear_correction'][0]
                    # ax.scatter(ref, mean, color=dict_param[param]['color'],label='mean corrected')
                    ax.text(0.98, 0.02,
                            'lin. regr:\na=%.2f\nb=%.2f' % (dict_param[param]['linear_correction'][0],dict_param[param]['linear_correction'][1]),
                            ha='right', va='bottom', transform=ax.transAxes, fontsize = fontsize_legend)
                    # title_save = title_save + ' (corrected)'
                    # title = title + ' (corrected)'
                # else:
                ax.scatter(ref, mean, color=dict_param[param]['color'], label='mean')
                ax.scatter(ref, median, marker='+', color='red', label= 'median')

                if not std.isna().all():
                    ax.errorbar(ref, mean, yerr=std, fmt=",", color='dimgrey',label='std', markersize=8, capsize=3,elinewidth=1)
                if not ref_std.isna().all(): # if they are not all nan we plot it
                    ax.errorbar(ref, mean, xerr=ref_std, fmt=",", color='dimgrey',label='std', markersize=8, capsize=3, elinewidth=1)

                # plot the regression model
                if 'linear_regression' in dict_param[param]:
                    reg, R2, coef, intercept = linear_regression(x=ref, y=mean)
                    X_regr = np.linspace(min(ref)*0.9, max(ref)*1.1, num=50)
                    X_regr = X_regr.reshape(X_regr.size, 1)
                    ax.plot(X_regr, reg.predict(X_regr), mark4, color=lin_regr_color, lw=2,
                            label='lin. regr')
                    ax.text(0.98, 0.02,
                            'lin. regr:\n$R^{2}$=%.2f\ny=%.2fx + %.2f' % (R2,coef[0][0],intercept[0]),
                            ha='right', va='bottom', transform=ax.transAxes, fontsize = fontsize_legend)

                if 'lr' in dict_param[param]:
                    if not isinstance(dict_param[param]['lr'], pd.DataFrame):
                        if dict_param[param]['lr'] == None:
                            logging.info('no linear regression active for ' + param)
                        elif dict_param[param]['lr'] == 'yes':
                            reg, R2, a, b = linear_regression(x=ref, y=mean)
                            X_regr = np.linspace(min(ref) * 0.9, max(ref) * 1.1, num=50)
                            X_regr = X_regr.reshape(X_regr.size, 1)
                            ax.plot(X_regr, reg.predict(X_regr), mark4, color=lin_regr_color, lw=2,
                                    label='lin. regr')
                            ax.text(0.98, 0.02, 'lin. regr:\n$R^{2}$=%.2f\ny=%.2fx + %.2f' % (R2, a, b), ha='right',
                                    va='bottom', transform=ax.transAxes, fontsize=fontsize_legend)
                            logging.info('linear regression active ' + param)
                    else:  # we have a dictionary
                        index = X.replace('_ref', '') + '-' + Y
                        if index in dict_param[param]['lr'].index:
                            a = dict_param[param]['lr'].loc[index, 'a']
                            b = dict_param[param]['lr'].loc[index, 'b']
                            R2 = dict_param[param]['lr'].loc[index, 'R2']
                            X_regr = np.linspace(min(ref) * 0.9, max(ref) * 1.1, num=50)
                            X_regr = X_regr.reshape(X_regr.size, 1)
                            ax.plot(X_regr, a * X_regr + b, mark4, color=lin_regr_color, lw=2,
                                    label='lin. regr')
                            ax.text(0.98, 0.02, 'lin. regr:\n$R^{2}$='+R2+'\ny=%.2fx + %.2f' % (a, b), ha='right',
                                    va='bottom', transform=ax.transAxes, fontsize=fontsize_legend)
                        else:
                            logging.info('No corrective factors for ' + index)

                if 'libelled' in dict_param[param]:
                    for xx in dict_param[param]['libelled'].index:
                        x = ref[xx]
                        y = mean[xx]
                        label = truncate_value(dict_param[param]['libelled'][xx],2)
                        ax.annotate(label,  # this is the text
                                    (x, y),  # these are the coordinates to position the label
                                    textcoords="offset points",  # how to position the text
                                    xytext=(0, 10),  # distance from text to points (x,y)
                                    ha='right')

            ax.set_title(title, fontsize=fontsize, fontname=fontname)
            ax.set_ylim(ymin=-0.05)
            ax.set_xlim(xmin=-0.05)
            if 'xy_lim' in dict_param[param]:
                ax.set_ylim(ymax=dict_param[param]['xy_lim'][0])
                ax.set_xlim(xmax=dict_param[param]['xy_lim'][1])
                ax.yaxis.set_major_formatter("{x:.1f}")  # set tick format with only 1 decimal
            ax.set_xlabel('Reference (' + dict_param[param]['unit'] + ')', fontsize=fontsize, fontname=fontname)
            ax.set_ylabel('SNAC (' + dict_param[param]['unit'] + ')', fontsize=fontsize, fontname=fontname)
            ax.tick_params(axis='both', labelsize=fontsize_tick)
            ax.legend(loc='upper left',fontsize=fontsize_legend)
            if 'pls' not in title_save:
                fig.savefig(path+'/image' + '/'+title_save+ '.png')
            else:
                fig.savefig(path + '/image_pls' + '/' + title_save + '.png')

            plt.close()
    else:
        logging.info('Plot comparison profile impossible because '+Y + ' not in dataframe')
    return


def plot_exactitude_general(data, dict_param, param, path, save_name):
    logging.info(param)
    param_ref = param # we do that becaus the IC has no reference values and we use TAC for comparison
    # if param == 'IC'or param == 'TAC':
    #     param_ref = 'TAC_fostac'
    Y = param + '_geq'
    X = param_ref + '_ref'
    if 'X' in dict_param[param]:
        X = dict_param[param]['X']
    if 'Y' in dict_param[param]:
        Y = dict_param[param]['Y']

    if 'titre_supp' in dict_param[param]:
        title_save = save_name + param + dict_param[param]['titre_supp']
        title = 'Accuracy Profile - '+ param
    else:
        title_save = save_name + param
        title = 'Accuracy Profile - '+ param

    ref = data.loc[:, 'Ref_res_mean'][X]

    if Y in data.loc[:, 'SNAC_res_mean'].columns:
        # defines intervals
        tolerance_interval_up = data.loc[:, 'SNAC_res_Itol_up_rel'][Y] # il faut calculer pour tous les points
        tolerance_interval_down = data.loc[:, 'SNAC_res_Itol_down_rel'][Y]  # il faut calculer pour tous les points
        taux_recouvrement = data.loc[:, 'SNAC_res_Rf'][Y] #fraction retrouvée, exprimée en % par rapport à la valeur de référence du matériau d’essai, pour le niveau de concentration considéré
        acceptability_interval_x1, acceptability_interval_up1, acceptability_interval_down1 = accep_interval(abs_before_1 = 0.2, rel_after_1 = 20, limit_graphs=[0,1,max(ref)])
        acceptability_interval_x2, acceptability_interval_up2, acceptability_interval_down2 = accep_interval(abs_before_1=0.25,rel_after_1=25,limit_graphs=[0,1,max(ref)])
        acceptability_interval_x3, acceptability_interval_up3, acceptability_interval_down3 = accep_interval(abs_before_1=0.30,rel_after_1=30,limit_graphs=[0,1,max(ref)])
    
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
        # ax.axline((0, acceptability_interval_up), slope=0, ls=mark1, label='acceptability_interval_up', color='red')
        # ax.axline((0, acceptability_interval_down), slope=0, ls=mark1, label='acceptability_interval_down', color='red')
    
        ax.axline((0, 100), slope=0, ls='--',color = 'k', linewidth=1)
        ax.plot(acceptability_interval_x1, acceptability_interval_up1, ls=mark1, color=accept_lim1,linewidth=1)
        ax.plot(acceptability_interval_x1, acceptability_interval_down1, ls=mark1, label='interval 20%', color=accept_lim1,linewidth=1)
        ax.plot(acceptability_interval_x2, acceptability_interval_up2, ls=mark1, color=accept_lim2,linewidth=1)
        ax.plot(acceptability_interval_x2, acceptability_interval_down2, ls=mark1, label='interval 25%', color=accept_lim2,linewidth=1)
        ax.plot(acceptability_interval_x3, acceptability_interval_up3, ls=mark1, color=accept_lim3,linewidth=1)
        ax.plot(acceptability_interval_x3, acceptability_interval_down3, ls=mark1, label='interval 30%', color=accept_lim3,linewidth=1)
        #
        # ax.scatter(ref, tolerance_interval_up, marker='^', label='tolerance limit up', color='dimgrey')
        # ax.scatter(ref, tolerance_interval_down,marker='v', label='tolerance limit down', color='darkgrey')
        ax.errorbar(ref, taux_recouvrement, yerr=(tolerance_interval_up - tolerance_interval_down)/2, fmt=",",label='tol. limit (\u03B2=80%)', color='dimgrey', markersize=8, capsize=3, elinewidth=1)

        ax.scatter(ref, taux_recouvrement, label='recovery factor', color=dict_param[param]['color'])
        ax.set_ylim(0, 200)
    
        ax.set_xlabel('Reference ('+dict_param[param]['unit'] + ')', fontsize=fontsize, fontname=fontname)
        ax.set_ylabel('Relative incertitude (%)', fontsize=fontsize, fontname=fontname)
        ax.tick_params(axis='both', labelsize=fontsize_tick)
        ax.legend(loc='upper center',fontsize=fontsize_legend)
        ax.set_title(title, fontsize=fontsize, fontname=fontname)

        if 'linear_correction' in dict_param[param]:
            ax.text(0.98, 0.02,
                    'lin. regr:\na=%.2f\nb=%.2f' % (
                    dict_param[param]['linear_correction'][0], dict_param[param]['linear_correction'][1]),
                    ha='right', va='bottom', transform=ax.transAxes, fontsize=fontsize_legend)

        if 'pls' not in title_save:
            fig.savefig(path + '/image' + '/' + title_save + '.png')
        else:
            fig.savefig(path + '/image_pls' + '/' + title_save + '.png')
        plt.close('all')
    else:
        logging.info('Plot accuracy profile impossible because '+Y + ' not in dataframe')
    return

def FOS_Hach_plot(data,dict_param,param,path,save_name):
    param_ref = param # we do that because the IC has no reference values and we use TAC for comparison
    if param == 'IC'or param == 'TAC':
        param_ref = 'TAC_fostac'
    Y = param + '_geq'
    X = param_ref + '_ref'
    if 'X' in dict_param[param]:
        X = dict_param[param]['X']
    if 'Y' in dict_param[param]:
        Y = dict_param[param]['Y']

    # if 'ref' in Y:
    #     Y_select =
    if not ((data.loc[:, 'Ref_res_mean'][X].isna().all()) or (data.loc[:, 'Ref_res_mean'][Y].isna().all())):
        if 'titre_supp' in dict_param[param]:
            title_save = save_name + param + dict_param[param]['titre_supp']
            title = 'Hach/Ref Comparison - ' + param
        else:
            title_save = save_name + param
            title = 'Hach/Ref Comparison - ' + param

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
        ref = data.loc[:, 'Ref_res_mean'][X]
        ref_std = data.loc[:, 'Ref_res_std'][X]
        mean = data.loc[:, 'Ref_res_mean'][Y]
        std = data.loc[:, 'Ref_res_std'][Y]
        median = data.loc[:, 'Ref_res_median'][Y]

        max_bisector = max(ref.max(), mean.max())
        ax.plot([0, max_bisector], [0, max_bisector], '--', color=bis_color, label='bisector', lw=1)

        # ranges
        coeff = 0.2 * max_bisector  # 20 %
        ax.plot([0,max_bisector-coeff], [0, max_bisector], mark1, color=accept_lim1, label='interval 20%', lw=1)
        ax.plot([0, max_bisector], [0, max_bisector-coeff], mark1, color=accept_lim1, lw=1)

        coeff = 0.25 * max_bisector  # 25 %
        ax.plot([0,max_bisector-coeff], [0, max_bisector], mark1, color=accept_lim2,label='interval 25%', lw=1)
        ax.plot([0, max_bisector], [0, max_bisector-coeff], mark1, color=accept_lim2, lw=1)

        coeff = 0.30 * max_bisector  # 25 %
        ax.plot([0,max_bisector-coeff], [0, max_bisector], mark1, color=accept_lim3,label='interval 30%', lw=1)
        ax.plot([0, max_bisector], [0, max_bisector-coeff], mark1, color=accept_lim3, lw=1)


        if (dict_param[param]['test_color'] == 'different') and (len(data.index.names) > 1):
            count_median = 0
            count_std = 0
            for i in data.index.get_level_values(0).drop_duplicates():
                ref_matrix = ref.loc[i,:]
                mean_matrix = mean.loc[i,:]
                std_matrix = std.loc[i,:]
                median_matrix = median.loc[i, :]
                ref_std_matrix = ref_std.loc[i,:]
                # plot median
                if count_median == 0:
                    ax.scatter(ref_matrix, median_matrix, marker='+', color='red', label=' median')
                    count_median = count_median +1
                else:
                    ax.scatter(ref_matrix, median_matrix, marker='+', color='red')

                # std
                if count_std == 0:
                    if not std_matrix.isna().all():
                        ax.errorbar(ref_matrix, mean_matrix, yerr=std_matrix, fmt=",", color='dimgrey', label='std',
                                    markersize=8, capsize=3, elinewidth=1)
                    if not ref_std.isna().all():  # if they are not all nan we plot it ( we have error only for FOS and TAC because made with the Hach fostac)
                        ax.errorbar(ref_matrix, mean_matrix, xerr=ref_std_matrix, fmt=",", color='dimgrey', label='std',
                                    markersize=8, capsize=3, elinewidth=1)
                    count_std = count_std + 1
                else:
                    if not std_matrix.isna().all():
                        ax.errorbar(ref_matrix, mean_matrix, yerr=std_matrix, fmt=",", color='dimgrey', markersize=8,capsize=3, elinewidth=1)
                    if not ref_std.isna().all():  # if they are not all nan we plot it ( we have error only for FOS and TAC because made with the Hach fostac)
                        ax.errorbar(ref_matrix, mean_matrix, xerr=ref_std_matrix, fmt=",", color='dimgrey',markersize=8, capsize=3, elinewidth=1)
                # data
                ax.scatter(ref_matrix, mean_matrix, color=globals()[i], label=i + ' mean')
        else:
            ax.scatter(ref, mean, color=dict_param[param]['color'], label='mean')
            ax.scatter(ref, median, marker='+', color='red', label=' median')
            if not std.isna().all():
                ax.errorbar(ref, mean, yerr=std, fmt=",", color='dimgrey',label='std', markersize=8, capsize=3,elinewidth=1)
            if not ref_std.isna().all(): # if they are not all nan we plot it
                ax.errorbar(ref, mean, xerr=ref_std, fmt=",", color='dimgrey',label='std', markersize=8, capsize=3, elinewidth=1)

        if 'linear_regression' in dict_param[param]:
            reg, R2, coef, intercept = linear_regression(x=ref, y=mean)
            X_regr = np.linspace(min(ref) * 0.9, max(ref) * 1.1, num=50)
            X_regr = X_regr.reshape(X_regr.size, 1)
            ax.plot(X_regr, reg.predict(X_regr), mark4, color=lin_regr_color, lw=2,
                    label='lin. regr')
            ax.text(0.98, 0.02,
                    'lin. regr:\n$R^{2}$=%.2f\ny=%.2fx + %.2f' % (R2, coef[0][0], intercept[0]), ha='right', va='bottom',
                    transform=ax.transAxes, fontsize = fontsize_legend)

        if 'lr' in dict_param[param]:
            if not isinstance(dict_param[param]['lr'], pd.DataFrame):
                if dict_param[param]['lr'] == None:
                    logging.info('no linear regression active for ' + param)
                elif dict_param[param]['lr'] == 'yes':
                    reg, R2, a, b = linear_regression(x=ref, y=mean)
                    X_regr = np.linspace(min(ref) * 0.9, max(ref) * 1.1, num=50)
                    X_regr = X_regr.reshape(X_regr.size, 1)
                    ax.plot(X_regr, reg.predict(X_regr), mark4, color=lin_regr_color, lw=2,
                            label='lin. regr')
                    ax.text(0.98, 0.02, 'lin. regr:\n$R^{2}$=%.2f\ny=%.2fx + %.2f' % (R2, a, b), ha='right',
                            va='bottom', transform=ax.transAxes, fontsize=fontsize_legend)
                    logging.info('linear regression active ' + param)
            else:  # we have a dictionary
                index = X.replace('_ref', '') + '-' + Y
                if index in dict_param[param]['lr'].index:
                    a = dict_param[param]['lr'].loc[index, 'a']
                    b = dict_param[param]['lr'].loc[index, 'b']
                    R2 = dict_param[param]['lr'].loc[index, 'R2']
                    X_regr = np.linspace(min(ref) * 0.9, max(ref) * 1.1, num=50)
                    X_regr = X_regr.reshape(X_regr.size, 1)
                    ax.plot(X_regr, a * X_regr + b, mark4, color=lin_regr_color, lw=2,
                            label='lin. regr')
                    ax.text(0.98, 0.02, 'lin. regr:\n$R^{2}$=' + R2 + '\ny=%.2fx + %.2f' % (a, b), ha='right',
                            va='bottom', transform=ax.transAxes, fontsize=fontsize_legend)
                else:
                    logging.info('No corrective factors for ' + index)

        ax.set_title(title, fontsize=fontsize, fontname=fontname)
        ax.set_ylim(ymin=-0.05)
        ax.set_xlim(xmin=-0.05)
        if 'xy_lim' in dict_param[param]:
            ax.set_ylim(ymax=dict_param[param]['xy_lim'][0])
            ax.set_xlim(xmax=dict_param[param]['xy_lim'][1])
            ax.yaxis.set_major_formatter("{x:.1f}")  # set tick format with only 1 decimal
        ax.set_xlabel('Reference (' + dict_param[param]['unit'] + ')', fontsize=fontsize, fontname=fontname)
        ax.set_ylabel('Hach (' + dict_param[param]['unit'] + ')', fontsize=fontsize, fontname=fontname)
        ax.tick_params(axis='both', labelsize=fontsize_tick)
        ax.legend(loc='upper left', fontsize=fontsize_legend)
        if 'pls' not in title_save:
            fig.savefig(path + '/image' + '/' + title_save + '.png')
        else:
            fig.savefig(path + '/image_pls' + '/' + title_save + '.png')
        plt.close()
    return


def box_plot(data_x,data_y, dict_param, param, path, save_name):
    param_ref = param  # we do that becaus the IC has no reference values and we use TAC for comparison
    if param == 'IC' or param == 'TAC':
        param_ref = 'TAC_fostac'
    Y = param + '_geq'
    X = param_ref + '_ref'
    if 'X' in dict_param[param]:
        X = dict_param[param]['X']
    if 'Y' in dict_param[param]:
        Y = dict_param[param]['Y']

    if 'titre_supp' in dict_param[param]:
        save_name = save_name + dict_param[param]['titre_supp']

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
    ref = data_x.loc[:, 'Ref_res_mean'][X]
    # ref_std = data.loc[:, 'Ref_res_std'][X]
    mean = data_x.loc[:, 'SNAC_res_mean'][Y]
    # std = data.loc[:, 'SNAC_res_std'][Y]

    data_y_BP = list()
    for i in data_y.index.levels[0]:
        if 'T1'in i:
            data_y_BP.append(data_y.loc[i,Y])
### df plot
    from copy import deepcopy
    data_y_BP_df = deepcopy(data_y)
    for i in data_y.index.levels[0]:
        if 'T1' not in i:
            data_y_BP_df.drop(index=i, inplace=True)

    parameter = 'VFA_geq'
    df_param = data_y_BP_df[parameter]

    df_plot = pd.DataFrame(columns=['T1_1','T1_2','T1_3'],index=[1,2,3,4,5])
    df_plot = pd.DataFrame()
    for i in ['T1_1', 'T1_2', 'T1_3','T1_4']:
        df_plot = pd.concat([df_plot, df_param.loc[i].reset_index(drop=True)], axis=1, ignore_index=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
    df_plot.boxplot()
    #data_y_BP_df.boxplot(column=['VFA_geq'], )

############### fin df box plot #############

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
    max_bisector = max(ref.max(), mean.max())
    ax.plot([0, max_bisector], [0, max_bisector], '--', color=bis_color, label='bisector', lw=1)

    ax.boxplot(data_y_BP, positions = truncate_value(ref,3), widths = 0.1)

    ax.set_title(save_name + param, fontsize=fontsize, fontname=fontname)
    ax.set_xlabel('ref (' + dict_param[param]['unit'] + ')', fontsize=fontsize, fontname=fontname)
    ax.set_ylabel('SNAC(' + dict_param[param]['unit'] + ')', fontsize=fontsize, fontname=fontname)
    ax.legend(loc='upper left',fontsize=fontsize)

    if 'libelled' in dict_param[param]:
        for xx in dict_param[param]['libelled'].index:
            x = ref[xx]
            y = mean[xx]
            label = truncate_value(dict_param[param]['libelled'][xx],2)
            ax.annotate(label,  # this is the text
                         (x, y),  # these are the coordinates to position the label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='right')

    fig.savefig(path + '/' + save_name + param + '.png')
    plt.close()
    return







