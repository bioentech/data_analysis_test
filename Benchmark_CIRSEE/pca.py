import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# from datasciencetools.eda import *
# from datasciencetools.plot import *
# from datasciencetools.metrics import *
# from datasciencetools.modeling import *
import pandas as pd
from plot_article import *
from tool import *
import re
import logging
# from data_filter import *

logging.basicConfig(
            level=logging.INFO,
            #format="[%(asctime)-15s][%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
)

# set paths
project_path = os.path.dirname(os.path.abspath(__file__))
input_path = project_path + '\inputs'
output_path = project_path + '\outputs'

# create folders
for p in [input_path, output_path]:
       if not os.path.exists(p):
              os.makedirs(p)

# chose file input to work on
file_SNAC = '/Res_normal.csv'
# file_SNAC = '/Res_sep_opt_20230824.csv'
# file_SNAC = '/BRL_20230831_sep_opt_VFA12.csv'
# file_SNAC = '/Res_noreject.csv'

# chose the sensitivity between different series
value_round = 1

logging.info(file_SNAC)
output_folder_path = output_path + file_SNAC.replace('.csv','')

# read data
Res_SNAC_raw = pd.read_csv(input_path + file_SNAC, index_col=0, header=0, sep=';', decimal='.')
Res_ref_raw = pd.read_csv(input_path + '/Data_reference.csv', index_col=0, header=0, sep=';', decimal=',')
Res_ref_raw['TAC_fostac_ref'] = Res_ref_raw['TAC_fostac_ref'] / 1000
Res_ref_raw.drop(columns='TAC_ref', inplace=True) # this columns was simply a reference to follow, but the true value is the one measured
Res_ref_raw.rename(columns={'TAC_fostac_ref': 'TAC_ref'}, inplace=True)
Res_ref_raw['FOS_ref'] = Res_ref_raw['FOS_ref']/ 1000
Res_ref = round_value_pd(Res_ref_raw,decimals=3)

Res_SNAC = round_value_pd(Res_SNAC_raw,decimals=3)

list_name = {'506_VFA_geq_PLS': 'pls_VFA_geq',
            '506_TAN_geq_PLS': 'pls_TAN_geq',
            '505_VFA_opt_sep1': 'sep_VFA1_geq',
            '505_VFA_opt_sep2': 'sep_VFA2_geq',
            '505_TAN_opt_sep': 'sep_TAN_geq'}
for i in list_name:
    if i in Res_SNAC.columns:
        Res_SNAC.rename(columns={i:list_name[i]}, inplace=True)

All_res_normal_raw = pd.concat([Res_ref, Res_SNAC], axis=1, keys=['reference data', 'SNAC data']) # erase "ok" in data reference

# cleaning data
logging.info('############### DATA CLEANING ##############')
drop_analysis = {'T3_2_1',# salt did not dissolved properly
                 'T3_2_2',# salt did not dissolved properly
                 'T3_2_3',# salt did not dissolved properly
                 'T3_7_1',# salt did not dissolved properly
                 'T3_7_2',# salt did not dissolved properly
                 'T3_7_3',# salt did not dissolved properly
                 'T3_8_1',# because we changed concentration of T3_8 afterwards
                 'T3_8_2',# because we changed concentration of T3_8 afterwards
                 'T3_8_3',# salt did not dissolved properly
                 'T3_8_4',# salt did not dissolved properly
                 'T3_8_5',# no results because volume too important
                 'T3_8_6',  # salt did not dissolved properly
                 'T3_8_7',  # salt did not dissolved properly
                 'T4_5_1',  # pas de fichier src
                 'T4_5_2',  # pas de fichier src
                 'T4_5_3'}  # les autres n'étant pas arrivé au serveur, je ne le considère pas
for res in ['Res_SNAC', 'Res_ref']: # we erase only in SNAC res because if not it could erase the _1 where the reference values are saved
    for test in drop_analysis:
        data = globals()[res]
        if test in data.index:
            data.drop(test, inplace=True)
            logging.info(test + ' erased from ' + res)


# check if res have the same elements
if Res_ref.equals(Res_SNAC.index):
    logging.info("Res_ref and Res_SNAC_raw have the same elements")
else: # the difference can come from items order
    logging.info("Res_ref and Res_SNAC have NOT the same elements")
    logging.info('Res_SNAC misses : ')
    logging.info(Res_ref.index.difference(Res_SNAC.index))
    logging.info('Res_ref misses : ')
    logging.info(Res_SNAC.index.difference(Res_ref.index))


#   Metadata table
All_res_normal_metainfo = pd.DataFrame(
    index=['Total analysis','Total conditions', 'pH_acid_accepted', 'pH_basic_accepted','cond_basic_accepted',
           'result provided', 'result not provided','erased analysis'], columns=['All','T1','T2','T3','T4', 'VFA', 'TAN','IC', 'TAC', 'FOS','pls_VFA','pls_TAN'])
All_res_normal_metainfo['All']['erased analysis'] = str(drop_analysis.__len__()) + ' : ' + str(drop_analysis)
All_res_normal_metainfo['All']['Total analysis'] = Res_SNAC.shape[0] # we need to keep all the analysis in the data reference
All_res_normal_metainfo['All']['pH_acid_accepted'] = Res_SNAC[Res_SNAC['101_0']==True].shape[0]
All_res_normal_metainfo['All']['pH_basic_accepted'] = Res_SNAC[Res_SNAC['102_0']==True].shape[0]
All_res_normal_metainfo['All']['cond_basic_accepted'] = Res_SNAC[Res_SNAC['103_0']==True].shape[0]

# I need to trier with for loop since I cannot do it with index
for i in ['T1','T2','T3','T4']:
    count = 0
    count_pH_acid = 0
    count_pH_basic = 0
    count_cond_basic = 0
    for j in Res_SNAC.index.values:
        if (i in j):
            count = count+1
        if (i in j) and (Res_SNAC['101_0'][j]==True):
            count_pH_acid = count_pH_acid + 1
        if (i in j) and (Res_SNAC['102_0'][j]==True):
            count_pH_basic = count_pH_basic + 1
        if (i in j) and (Res_SNAC['103_0'][j]==True):
            count_cond_basic = count_cond_basic + 1
    if count!=0: # condition to avoid miscaucultaion whane some matrices are not analysed
        All_res_normal_metainfo[i]['Total analysis'] = count
        All_res_normal_metainfo[i]['pH_acid_accepted'] = format(count_pH_acid/count, ".0%")
        All_res_normal_metainfo[i]['pH_basic_accepted'] = format(count_pH_basic/count, ".0%")
        All_res_normal_metainfo[i]['cond_basic_accepted'] = format(count_cond_basic/count, ".0%")


for i in ['VFA', 'TAN','IC', 'TAC', 'FOS','pls_VFA','pls_TAN']:
    All_res_normal_metainfo[i]['result not provided'] = len(Res_SNAC[Res_SNAC[i+'_geq'].isna()].index.tolist())
    All_res_normal_metainfo[i]['result provided'] = Res_SNAC[i+'_geq'].size - All_res_normal_metainfo[i]['result not provided']

# Select only interesting data
interesting_columns=['VFA_geq','TAN_geq','IC_geq','FOS_geq','TAC_geq',
                    'pH_initial','conductivity_initial','pls_VFA_geq',
                    'pls_TAN_geq','sep_VFA1_geq','sep_VFA2_geq','sep_TAN_geq']
for i in Res_SNAC.columns:
    if i not in interesting_columns:
        Res_SNAC = Res_SNAC.drop(i, axis=1)


# Multindex creation based on SNAC_res names
idx2 = pd.DataFrame(Res_SNAC.index.copy(deep=True))
idx1 = pd.DataFrame(Res_SNAC.index.copy(deep=True))

i=0
for test in idx2['index']:
    count = 0
    for m in re.finditer('_', test): # must use this for loop to slice well the names based on the fact that we always have 2 "_"
        count = count + 1
        if count == 2:
            end = m.start()
    idx1['index'][i] = test[0:end]
    i = i + 1
idx = pd.concat([idx1, idx2], axis=1)
index = pd.MultiIndex.from_frame(idx, names=["test", "number"])
Res_SNAC_twoindex = pd.DataFrame(Res_SNAC.values, columns=Res_SNAC.columns, index=index)

# providing multindex to Res_ref
idx2_ref = pd.DataFrame(Res_ref.index)
idx1_ref = pd.DataFrame(Res_ref.index)
i=0
for test in idx2_ref['nom']:
    count = 0
    for m in re.finditer('_',test):  # must use this for loop to slice well the names based on the fact that we always have 2 "_"
        count = count + 1
        if count == 2:
            end = m.start()
    idx1_ref['nom'][i] = test[0:end]
    i = i + 1
idx_ref = pd.concat([idx1_ref, idx2_ref], axis=1)
index_ref = pd.MultiIndex.from_frame(idx_ref, names=["Serie", "Repetition"])
Res_ref_twoindex = pd.DataFrame(Res_ref.values, columns=Res_ref.columns, index=index_ref)

# put Res_ref et res_SNAC together to ensure that they use the same samples
All_res_twoindex = pd.concat([Res_ref_twoindex,Res_SNAC_twoindex],
                             axis=1,join='inner', verify_integrity = True,keys=['Res_ref','Res_SNAC'])

All_res_twoindex.index.set_names(["Serie", "Repetition"], level=[0,1], inplace=True)

# add corrected FOS
All_res_twoindex['Res_SNAC','corr_FOS_geq'] = (All_res_twoindex['Res_SNAC']['FOS_geq'] - 1.01)/1.33 # 1.01 et 1.33 sont issus de la regression linéaire
All_res_twoindex['Res_SNAC']['corr_FOS_geq'][All_res_twoindex['Res_SNAC']['corr_FOS_geq']<0] = 0 # si c'est negatif après correction il faut mettre zzro car pas de sens sinon

All_res_twoindex['Res_SNAC','corr_Hach_FOS_geq'] = (All_res_twoindex['Res_ref']['FOS_ref'] - 1.16)/0.81 # 1.01 et 1.33 sont issus de la regression linéaire
All_res_twoindex['Res_SNAC']['corr_Hach_FOS_geq'][All_res_twoindex['Res_SNAC']['corr_Hach_FOS_geq']<0] = 0 # si c'est negatif après correction il faut mettre zzro car pas de sens sinon



    # add several index for inter intra seria
for iii in All_res_twoindex.loc[:, 'Res_ref']:
    if iii not in ['TAC_ref', 'FOS_ref']: # I compute these later since they are alread the reference
        All_res_twoindex.insert(0, iii + '_Serie', round_value_serie(All_res_twoindex.loc[:, 'Res_ref'][iii], decimals=value_round))
        All_res_twoindex.set_index(All_res_twoindex[iii + '_Serie'], drop=True,append=True, inplace=True)
        All_res_twoindex.drop(iii + '_Serie', axis =1, inplace=True)


# computing mean and standard deviations INTRA serie
Res_ref_std = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
Res_ref_mean = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
Res_ref_median = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
Res_ref_SCE = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
Res_ref_nb = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
Res_ref_CV = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())

for i in All_res_twoindex.index.levels[0]:
    for ii in All_res_twoindex['Res_ref'].columns:
        Res_ref_std.loc[i,ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].std()
        Res_ref_mean.loc[i,ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].mean()
        Res_ref_median.loc[i,ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].median()
        Res_ref_SCE.loc[i,ii] = ((All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii] - Res_ref_mean.loc[i, ii])**2).sum()
        Res_ref_nb.loc[i,ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].count()
        Res_ref_CV.loc[i,ii] = Res_ref_std.loc[i,ii]/Res_ref_mean.loc[i,ii] * 100 # coefficient de variation
Res_ref_std.loc[:,'VFA_ref'] = np.nan # VFA, TAN n'ont pas de std car c'est une seule valeur repetée
Res_ref_std.loc[:,'TAN_ref'] = np.nan

# computing stat variables in Res_SNAC
Res_SNAC_std = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
Res_SNAC_mean = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
Res_SNAC_median = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
Res_SNAC_SCE = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
Res_SNAC_nb = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
Res_SNAC_CV = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
for i in idx1.drop_duplicates()['index']:
    for ii in All_res_twoindex['Res_SNAC'].columns:
        Res_SNAC_std.loc[i,ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].std()
        Res_SNAC_mean.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].mean()
        Res_SNAC_median.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].median()
        Res_SNAC_SCE.loc[i, ii] = ((All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii] - Res_SNAC_mean.loc[i, ii])**2).sum()
        Res_SNAC_nb.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].count()
        Res_SNAC_CV.loc[i, ii] = Res_SNAC_std.loc[i,ii]/Res_SNAC_mean.loc[i, ii] * 100 # coefficient de variation

# # Creating results table

# All_res_normal = pd.concat([Res_ref_mean,Res_SNAC_mean,Res_SNAC_std,], axis=1, verify_integrity = True, keys=['Ref_res','SNAC_res_mean','SNAC_res_std'])
All_res_INTRAserie = pd.concat([Res_ref_mean,
                            Res_ref_std,
                            Res_ref_median,
                            Res_ref_SCE,
                            Res_SNAC_mean,
                            Res_SNAC_std,
                            Res_SNAC_median,
                            Res_SNAC_SCE,
                            Res_SNAC_nb], axis=1, verify_integrity = True,
                           keys=['Ref_res_mean',
                                 'Ref_res_std',
                                 'Ref_res_median',
                                 'Ref_res_SCE',
                                 'SNAC_res_mean',
                                 'SNAC_res_std',
                                 'SNAC_res_median',
                                 'SNAC_res_SCE',
                                 'SNAC_res_nb'])

#   providing multindex to Res_ref
idx2_all = pd.DataFrame(All_res_INTRAserie.index.copy(deep=True))
idx1_all = pd.DataFrame(All_res_INTRAserie.index.copy(deep=True))
i=0
for test in idx2_all[0]:
    end = test.index('_')
    idx1_all[0][i] = test[0:end]
    i = i + 1
idx_all = pd.concat([idx1_all, idx2_all], axis=1)
idx_all_ref = pd.MultiIndex.from_frame(idx_all, names=["Matrix", "Serie"])
All_res_INTRAserie.set_index(idx_all_ref, inplace=True)


# changing the index for FOS and TAC since they do not really have a reference I must use the mean of each condition
for ref in ['FOS_ref', 'TAC_ref']:
    pdd = pd.DataFrame(columns=['value'], index=All_res_twoindex.index)
    for i in All_res_twoindex.index:
        pdd['value'][i] = round(All_res_INTRAserie['Ref_res_mean', ref].xs(i[0], level='Serie').values[0], 2) # rounding by 3 means that any data is agregated
    All_res_twoindex[ref+'_Serie'] = pdd
    All_res_twoindex.set_index(All_res_twoindex[ref+'_Serie'], drop=True, append=True, inplace=True)
    All_res_twoindex.drop(ref+'_Serie', axis=1, inplace=True)


# Analyse Profil d'exactitude
""" source: 2010_Cahier_des_techniques dans WP05"""
# for param in ['VFA','TAN']:
    # All_res_INTRAserie = compute_profile_exactitude_simplified(All_res_INTRAserie, param)


# creating dataframe with data INTER serie
All_res_ALLserie = deepcopy(All_res_twoindex)
All_res_ALLserie_dict ={}
All_res_ALLserie_dict_raw ={}
All_res_INTERserie_dict ={}
All_res_INTERserie_dict_raw ={}
dict_param_relation = {'VFA': ['VFA_geq', 'pls_VFA_geq', 'FOS_geq','corr_FOS_geq','corr_Hach_FOS_geq','sep_VFA1_geq','sep_VFA2_geq'],  # ref : [all param with this ref]
                       'TAN': ['TAN_geq', 'pls_TAN_geq','sep_TAN_geq'],
                       'TAC': ['TAC_geq', 'IC_geq'],
                       'FOS': ['FOS_geq'],
                       }
for i in All_res_ALLserie.xs('Res_ref',axis=1).columns:
    param = i.replace('_ref','')
    Res_param = deepcopy(All_res_ALLserie)
    Res_param = erase_unusefull_info(Res_param, param) # erase not used info in specific pd param
    for ii in Res_param.index.names:
        if ii not in['Serie', 'Repetition',i+'_Serie']:
            Res_param.reset_index(level=ii, drop=True, inplace= True)
    Res_param = Res_param.swaplevel(i=-3, j=-1)
    Res_param = Res_param.swaplevel(i=-2, j=-1)

    All_res_ALLserie_dict_raw[param] = deepcopy(All_res_ALLserie)
    # All_res_ALLserie_dict_raw[param] = erase_unusefull_info(All_res_ALLserie_dict_raw[param], param)
    All_res_ALLserie_dict_raw[param].sort_index(level=-1, ascending=True, inplace=True,na_position='last')

    All_res_ALLserie_dict[param] = pd.DataFrame()
    All_res_ALLserie_dict[param] = compute_stat(Res_param)
    #All_res_ALLserie_dict[param] = compute_profile_exactitude_simplified(All_res_ALLserie_dict[param], param)
    All_res_ALLserie_dict[param] = compute_profile_exactitude_global(data_global = All_res_twoindex, data_intra = All_res_INTRAserie,
                                                                     data_all = All_res_ALLserie_dict[param], ref = param, eq_des = None, serie='no')
    All_res_ALLserie_dict[param].sort_index(level=-1, ascending=True, inplace=True, na_position='last')

    All_res_INTERserie_dict[param] = compute_profile_exactitude_global(data_global = All_res_twoindex, data_intra = All_res_INTRAserie,
                                                                       data_all = All_res_ALLserie_dict[param], ref = param, eq_des = None, serie='yes')

# creating/emptying output folder
import shutil
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)
os.makedirs(output_folder_path)
# I create different folder for all the types of data
intra_path = os.path.join(output_folder_path, 'intra')
inter_path = os.path.join(output_folder_path, 'inter')
all_path = os.path.join(output_folder_path, 'all')
for i in ['intra','inter','all']:
    os.makedirs(os.path.join(output_folder_path,i,'image'))
    os.makedirs(os.path.join(output_folder_path,i,'image_pls'))

    
# SAVING IN EXCEL
    # put something else in the metainfo
All_res_normal_metainfo['All']['Total conditions'] = All_res_INTRAserie.shape[0] # we need to keep all the analysis in the data reference
# I need to trier with for loop since I cannot do it with index
for i in ['T1','T2','T3','T4']:
    All_res_normal_metainfo[i]['Total conditions'] = All_res_INTRAserie.loc[i,:].shape[0]
    logging.info('creating meta info from All_res_normal. not valid for other files input')

    # Saving results table
All_res_normal = truncate_value_pd(All_res_INTRAserie,decimals=3)
All_res_normal_raw = truncate_value_pd(All_res_normal_raw,decimals=3)

    # save all data with specific analysis
writer = pd.ExcelWriter(output_folder_path + file_SNAC.replace('.csv','') +' intra_results.xlsx')
All_res_normal_raw.to_excel(writer, sheet_name = 'res_raw',startcol=1, startrow=1, freeze_panes=(3,2))
All_res_normal.to_excel(writer, sheet_name = 'res_analysed',startcol=1, startrow=1, freeze_panes=(3,2))
All_res_normal_metainfo.to_excel(writer, sheet_name = 'metainfo',startcol=1, startrow=1, freeze_panes=(2,2))
writer.save()
writer.close()

    # save all data with global analysis
writer = pd.ExcelWriter(output_folder_path + file_SNAC.replace('.csv','') +' all_results.xlsx')
for i in All_res_ALLserie_dict:
    round_value_pd(All_res_ALLserie_dict[i], decimals=2).to_excel(writer, sheet_name = str(i),startcol=1, startrow=1, freeze_panes=(3,2))
writer.save()
writer.close()

    # save all data with inter serie analysis (complete analysis)
writer = pd.ExcelWriter(output_folder_path + file_SNAC.replace('.csv','') +' inter_results.xlsx')
for i in All_res_INTERserie_dict:
    round_value_pd(All_res_INTERserie_dict[i], decimals=2).to_excel(writer, sheet_name = str(i),startcol=1, startrow=1, freeze_panes=(3,3))

        # I put in a table the information of inter data to put  in the livrable.
All_res_INTERserie_dict_metainfo = deepcopy(All_res_INTERserie_dict)
ref_imp = {'VFA':['pls_VFA_geq'], 'TAN':['TAN_geq'], 'TAC':['TAC_geq']}
parameter_imp = {'Ref_res_mean':1,'SNAC_res_std':3,'SNAC_res_mean':1,'SNAC_res_k':1,'SNAC_res_U':1,'SNAC_res_U_rel':0} # :'param' :  number --> decimel to round value
#:format(count_pH_acid/count, ".0%" All_res_INTERserie_dict_metainfo[ref][param].style.format("{:.0%}")
for ref in All_res_INTERserie_dict:
    if ref in ref_imp:
        for param in All_res_INTERserie_dict[ref].columns.levels[0].values:
            if param not in parameter_imp:
                del All_res_INTERserie_dict_metainfo[ref][param]
            else:#todo : it does not work for now
                All_res_INTERserie_dict_metainfo[ref][param]=round_value_pd(All_res_INTERserie_dict_metainfo[ref][param], decimals=parameter_imp[param])
                for estimator in All_res_INTERserie_dict_metainfo[ref][param]:
                    if estimator not in ref_imp[ref]:
                        del All_res_INTERserie_dict_metainfo[ref][param][estimator]
    else:
        del All_res_INTERserie_dict_metainfo[ref]

# parameter_imp = {'SNAC_res_U_aggr':1,'SNAC_res_U_aggr_rel':1}
# agrr = [[0,2],[2,5],[5,10],[10,20],[20,30]]
# b=deepcopy(All_res_INTERserie_dict_metainfo)
# c = b
# for param in parameter_imp :
#     for estimator in ['IC_geq', 'TAC_geq']:
#         new = pd.DataFrame(columns=[param],index=c['TAC'].index)
#         for i in new.index:
#             if not np.isnan(c['TAC']['SNAC_res_U'][estimator].loc[i]):
#                 for ii in agrr: # select the right interval
#                     if (c['TAC']['SNAC_res_U'][estimator].loc[i] > ii[0]) and ( c['TAC']['SNAC_res_U'][estimator].loc[i] <= ii[1]):
#                         sel = ii
#                 new.loc[i][param] = c['TAC']['SNAC_res_U'][estimator].loc[(c['TAC']['SNAC_res_U'][estimator] > sel[0]) & (c['TAC']['SNAC_res_U'][estimator] <= sel[1])].median()
#         c['TAC'][param,estimator] = new

parameter_imp = {'SNAC_res_std_aggr':1,'SNAC_res_std_aggr_rel':1}
agrr = [[0,2],[2,10],[10,30]]
# b=deepcopy(All_res_INTERserie_dict_metainfo)
c = All_res_INTERserie_dict_metainfo
for estimator in ['IC_geq', 'TAC_geq']:
    new = pd.DataFrame(columns=['SNAC_res_std_aggr'],index=c['TAC'].index)
    new_rel = pd.DataFrame(columns=['SNAC_res_std_aggr_rel'], index=c['TAC'].index)
    for i in new.index:
        if not np.isnan(i):
            for ii in agrr: # select the right interval
                if (i > ii[0]) and (i <= ii[1]):
                    sel = ii
                    break
            print('value: '+str(i))
            print('agre: '+str(sel))
            selected_data = c['TAC'].loc[(c['TAC']['SNAC_res_std'][estimator].index > sel[0]) & (c['TAC']['SNAC_res_std'][estimator].index <= sel[1])]
            observed_mean = selected_data['SNAC_res_mean'][estimator].mean()
            new.loc[i]['SNAC_res_std_aggr'] = selected_data['SNAC_res_std'][estimator].median() #obseved std median of selected data
            new_rel.loc[i]['SNAC_res_std_aggr_rel'] = format(new.loc[i]['SNAC_res_std_aggr']/ observed_mean, ".1%")
    c['TAC']['SNAC_res_std_aggr',estimator] = new
    c['TAC']['SNAC_res_std_aggr_rel', estimator] = new_rel


for i in All_res_INTERserie_dict_metainfo:
    All_res_INTERserie_dict_metainfo[i].to_excel(writer, sheet_name = i+' metainfo',startcol=1, startrow=1, freeze_panes=(3,3))

for i in All_res_INTERserie_dict_raw:
    round_value_pd(All_res_INTERserie_dict_raw[i], decimals=2).to_excel(writer, sheet_name=str(i), startcol=1, startrow=1, freeze_panes=(3, 3))

writer.save()
writer.close()

logging.info('Files saved correctly')

##### PLOTTING ######

# plot_Benchmark_param(All_res_ALLserie_dict, all_path, truncate_value_pd(Res_SNAC_raw,decimals=1), Res_SNAC_twoindex)
plot_Benchmark(All_res_INTRAserie, intra_path, truncate_value_pd(Res_SNAC_raw,decimals=1), Res_SNAC_twoindex)
plot_Benchmark_param(All_res_INTERserie_dict, inter_path, truncate_value_pd(Res_SNAC_raw,decimals=1), Res_SNAC_twoindex)

logging.info('Analysis ended correctly')



