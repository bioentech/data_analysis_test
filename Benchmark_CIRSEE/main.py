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

name_to_change = {'506_VFA_geq_PLS': 'pls_VFA_geq',
            '506_TAN_geq_PLS': 'pls_TAN_geq',
            '505_VFA_opt_sep1': 'sep_VFA1_geq',
            '505_VFA_opt_sep2': 'sep_VFA2_geq',
            '505_TAN_opt_sep': 'sep_TAN_geq',
             # '0_1': 'file', '0_2': 'SNAC_number'
                  }
for i in name_to_change:
    if i in Res_SNAC.columns:
        Res_SNAC.rename(columns={i:name_to_change[i]}, inplace=True)

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


#All_res_twoindex = two_index_creation()

# Select only interesting data
interesting_columns=['VFA_geq','TAN_geq','IC_geq','FOS_geq','TAC_geq',
                    'pH_initial','conductivity_initial','pls_VFA_geq',
                     ]
    # add automatically the columns whose name I changed (under the hypothesis that I need them)
for i in name_to_change:
    if name_to_change[i] not in interesting_columns:
        interesting_columns.append(name_to_change[i])

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
# All_res_twoindex['Res_SNAC','corr_FOS_geq'] = (All_res_twoindex['Res_SNAC']['FOS_geq'] - 1.01)/1.33 # 1.01 et 1.33 sont issus de la regression linéaire
# All_res_twoindex['Res_SNAC']['corr_FOS_geq'][All_res_twoindex['Res_SNAC']['corr_FOS_geq']<0] = 0 # si c'est negatif après correction il faut mettre zzro car pas de sens sinon
#
# All_res_twoindex['Res_SNAC','corr_Hach_FOS_geq'] = (All_res_twoindex['Res_ref']['FOS_ref'] - 1.16)/0.81 # 1.01 et 1.33 sont issus de la regression linéaire
# All_res_twoindex['Res_SNAC']['corr_Hach_FOS_geq'][All_res_twoindex['Res_SNAC']['corr_Hach_FOS_geq']<0] = 0 # si c'est negatif après correction il faut mettre zzro car pas de sens sinon


#create multindex
All_res_multiindex = multiindex_analysis_simple(All_res_twoindex,value_round)

# computing mean and standard deviations INTRA serie
All_res_INTRAserie = INTRA_analysis (All_res_multiindex,idx1_ref,idx1)

# create multindex
All_res_multiindex = multiindex_analysis_specific(All_res_multiindex,All_res_INTRAserie,value_round)
#------

# Analyse Profil d'exactitude
""" source: 2010_Cahier_des_techniques dans WP05"""
# for param in ['VFA','TAN']:
    # All_res_INTRAserie = compute_profile_exactitude_simplified(All_res_INTRAserie, param)


# creating dataframe with data INTER serie
dict_param_relation = {'VFA': ['VFA_geq', 'pls_VFA_geq', 'FOS_geq','sep_VFA1_geq','sep_VFA2_geq'],  # ref : [all param with this ref]
                       'TAN': ['TAN_geq', 'pls_TAN_geq','sep_TAN_geq'],
                       'TAC': ['TAC_geq', 'IC_geq'],
                       'FOS': ['FOS_geq'],
                       }
All_res_ALLserie_dict, All_res_ALLserie_dict_raw, All_res_INTERserie_dict, All_res_INTERserie_dict_raw = INTER_analysis(All_res_multiindex,All_res_INTRAserie, dict_param_relation)

# metainfo
All_res_INTERserie_dict_metainfo = metainfo(All_res_INTERserie_dict)

#### repeat all code for corrected values #################
# correction estimators with linear regression
dict_param_relation_Corr = {}
corr_factor = pd.DataFrame(columns=['Param','Estimator','a','b','R2'])
for param in dict_param_relation:
    if param in All_res_INTERserie_dict:
        dict_param_relation_Corr[param]=[]
        ref = param+'_ref'
        X = All_res_INTERserie_dict[param]['Ref_res_mean',ref]
        for estimator in dict_param_relation[param]:
            if estimator in All_res_INTERserie_dict[param]['SNAC_res_mean'].columns:
                Y = All_res_INTERserie_dict[param]['SNAC_res_mean',estimator]
                reg, R2, coef, intercept = linear_regression(x=X, y=Y)
                index = param+'-'+estimator
                corr_factor.loc[index,'Param'] = param
                corr_factor.loc[index,'Estimator'] = estimator
                corr_factor.loc[index,'a'] = round(coef[0][0],2)
                corr_factor.loc[index,'b'] = round(intercept[0], 2)
                corr_factor.loc[index,'R2'] = format(R2, ".0%")
                dict_param_relation_Corr[param].append('corr_'+estimator)
#
# correction estimators with linear regression
All_res_twoindex_Corr=deepcopy(All_res_twoindex) # it is a multilevel dataframe
for param in dict_param_relation:
    for estimator in dict_param_relation[param]:
        if estimator in All_res_twoindex_Corr['Res_SNAC']:
            index = param + '-' + estimator
            a = corr_factor.loc[index, 'a']
            b = corr_factor.loc[index, 'b']
            All_res_twoindex_Corr.drop(columns=estimator, level=1, inplace=True)
            All_res_twoindex_Corr['Res_SNAC', estimator] = (All_res_twoindex['Res_SNAC'][estimator] - b) / a
            All_res_twoindex_Corr['Res_SNAC'][estimator][All_res_twoindex['Res_SNAC'][estimator] < 0] = 0 # si c'est negatif après correction il faut mettre zzro car pas de sens sinon

#create multindex
All_res_multiindex_Corr = multiindex_analysis_simple(All_res_twoindex_Corr,value_round)

# computing mean and standard deviations INTRA serie
All_res_INTRAserie_Corr = INTRA_analysis (All_res_twoindex_Corr,idx1_ref,idx1)

# create multindex
All_res_multiindex_Corr = multiindex_analysis_specific(All_res_multiindex_Corr,All_res_INTRAserie_Corr,value_round)

# Accuracy profile analysis
All_res_ALLserie_dict_Corr, All_res_ALLserie_dict_raw_Corr, All_res_INTERserie_dict_Corr, All_res_INTERserie_dict_raw_Corr = INTER_analysis(All_res_multiindex_Corr,All_res_INTRAserie_Corr, dict_param_relation_Corr)

# metainfo
All_res_INTERserie_dict_metainfo_Corr = metainfo(All_res_INTERserie_dict_Corr)


# creating/emptying output folder
import shutil
if os.path.exists(output_folder_path):
    shutil.rmtree(output_folder_path)
os.makedirs(output_folder_path)

raw_output_folder_path = os.path.join(output_folder_path,'raw')
corr_output_folder_path = os.path.join(output_folder_path,'corr')

# I create different folder for all the types of data
raw_intra_path = os.path.join(raw_output_folder_path, 'intra')
raw_inter_path = os.path.join(raw_output_folder_path, 'inter')
raw_all_path = os.path.join(raw_output_folder_path, 'all')
for i in ['intra','inter','all']:
    os.makedirs(os.path.join(raw_output_folder_path, i,'image'))
    os.makedirs(os.path.join(raw_output_folder_path,i,'image_pls'))

corr_intra_path = os.path.join(corr_output_folder_path, 'intra')
corr_inter_path = os.path.join(corr_output_folder_path, 'inter')
corr_all_path = os.path.join(corr_output_folder_path, 'all')
for i in ['intra','inter','all']:
    os.makedirs(os.path.join(corr_output_folder_path,i,'image'))
    os.makedirs(os.path.join(corr_output_folder_path,i,'image_pls'))


# SAVING IN EXCEL
save_Excel(All_res_normal_metainfo,All_res_INTRAserie,All_res_normal_raw,All_res_ALLserie_dict,
               All_res_INTERserie_dict, All_res_INTERserie_dict_metainfo, All_res_INTERserie_dict_raw,
               raw_output_folder_path,file_SNAC)

# SAVING IN EXCEL
save_Excel(All_res_normal_metainfo, All_res_INTRAserie_Corr,All_res_normal_raw, All_res_ALLserie_dict_Corr,
               All_res_INTERserie_dict_Corr, All_res_INTERserie_dict_metainfo_Corr, All_res_INTERserie_dict_raw_Corr,
               corr_output_folder_path,file_SNAC)

# save corrective factor
writer = pd.ExcelWriter(corr_output_folder_path + file_SNAC.replace('.csv', '') + ' corrective factors.xlsx')
round_value_pd(corr_factor, decimals=2).to_excel(writer, startcol=1, startrow=1,freeze_panes=(3, 2))
writer.save()
writer.close()

logging.info('Saving data ended correctly')


##### PLOTTING ######

# plot_Benchmark_param(All_res_ALLserie_dict, all_path, truncate_value_pd(Res_SNAC_raw,decimals=1), Res_SNAC_twoindex)
# plot_Benchmark(All_res_INTRAserie, raw_intra_path, truncate_value_pd(Res_SNAC_raw,decimals=1))
plot_Benchmark_param(All_res_INTERserie_dict, raw_inter_path, truncate_value_pd(Res_SNAC_raw,decimals=1),linear_regression = corr_factor)
logging.info('Plotting raw data ended correctly')

# plot_Benchmark(All_res_INTRAserie_Corr, corr_intra_path, truncate_value_pd(Res_SNAC_raw,decimals=1))
plot_Benchmark_param(All_res_INTERserie_dict_Corr, corr_inter_path, truncate_value_pd(Res_SNAC_raw,decimals=1),linear_regression = 'yes' )
logging.info('Plotting corrected data ended correctly')


logging.info('Analysis ended correctly')



