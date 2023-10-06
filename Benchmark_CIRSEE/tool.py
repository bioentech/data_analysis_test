import logging

import numpy as np
from scipy import stats
import pandas as pd
from copy import deepcopy
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


def two_index_creation():
    return
def INTRA_analysis(All_res_twoindex,idx1_ref,idx1):
    # computing mean and standard deviations INTRA serie
    Res_ref_std = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
    Res_ref_mean = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
    Res_ref_median = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
    Res_ref_SCE = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
    Res_ref_nb = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())
    Res_ref_CV = pd.DataFrame(columns=All_res_twoindex['Res_ref'].columns, index=idx1_ref['nom'].drop_duplicates())

    for i in All_res_twoindex.index.levels[0]:
        for ii in All_res_twoindex['Res_ref'].columns:
            Res_ref_std.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].std()
            Res_ref_mean.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].mean()
            Res_ref_median.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].median()
            Res_ref_SCE.loc[i, ii] = ((All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii] - Res_ref_mean.loc[i, ii]) ** 2).sum()
            Res_ref_nb.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_ref', axis=1)[ii].count()
            Res_ref_CV.loc[i, ii] = Res_ref_std.loc[i, ii] / Res_ref_mean.loc[i, ii] * 100  # coefficient de variation
    Res_ref_std.loc[:, 'VFA_ref'] = np.nan  # VFA, TAN n'ont pas de std car c'est une seule valeur repetée
    Res_ref_std.loc[:, 'TAN_ref'] = np.nan

    # computing stat variables in Res_SNAC
    Res_SNAC_std = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
    Res_SNAC_mean = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
    Res_SNAC_median = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
    Res_SNAC_SCE = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
    Res_SNAC_nb = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
    Res_SNAC_CV = pd.DataFrame(columns=All_res_twoindex['Res_SNAC'].columns, index=idx1['index'].drop_duplicates())
    for i in idx1.drop_duplicates()['index']:
        for ii in All_res_twoindex['Res_SNAC'].columns:
            Res_SNAC_std.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].std()
            Res_SNAC_mean.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].mean()
            Res_SNAC_median.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].median()
            Res_SNAC_SCE.loc[i, ii] = ((All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii] -
                                        Res_SNAC_mean.loc[i, ii]) ** 2).sum()
            Res_SNAC_nb.loc[i, ii] = All_res_twoindex.xs(i, level='Serie').xs('Res_SNAC', axis=1)[ii].count()
            Res_SNAC_CV.loc[i, ii] = Res_SNAC_std.loc[i, ii] / Res_SNAC_mean.loc[
                i, ii] * 100  # coefficient de variation

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
                                    Res_SNAC_nb], axis=1, verify_integrity=True,
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
    i = 0
    for test in idx2_all[0]:
        end = test.index('_')
        idx1_all[0][i] = test[0:end]
        i = i + 1
    idx_all = pd.concat([idx1_all, idx2_all], axis=1)
    idx_all_ref = pd.MultiIndex.from_frame(idx_all, names=["Matrix", "Serie"])
    All_res_INTRAserie.set_index(idx_all_ref, inplace=True)

    return All_res_INTRAserie

def multiindex_analysis_simple(All_res_twoindex,value_round):
    # add several index for inter intra seria
    All_res_multiindex = deepcopy(All_res_twoindex)
    for iii in All_res_multiindex.loc[:, 'Res_ref']:
        if iii not in ['TAC_ref', 'FOS_ref']:  # I compute these later since they are already the reference
            All_res_multiindex.insert(0, iii + '_Serie', round_value_serie(All_res_multiindex.loc[:, 'Res_ref'][iii],
                                                                           decimals=value_round))
            All_res_multiindex.set_index(All_res_multiindex[iii + '_Serie'], drop=True, append=True, inplace=True)
            All_res_multiindex.drop(iii + '_Serie', axis=1, inplace=True)
    return All_res_multiindex

def multiindex_analysis_specific(All_res_multiindex, All_res_INTRAserie, value_round):
    # changing the index for FOS and TAC : FOS since they do not really have a reference I must use the mean of each condition
    for ref in ['FOS_ref']:
        new = pd.DataFrame(columns=['value'], index=All_res_multiindex.index)
        for i in All_res_multiindex.index:
            new['value'][i] = round(All_res_INTRAserie['Ref_res_mean', ref].xs(i[0], level='Serie').values[0],
                                    2)  # rounding by 3 means that no data is agregated
        All_res_multiindex[ref + '_Serie'] = new
        All_res_multiindex.set_index(All_res_multiindex[ref + '_Serie'], drop=True, append=True, inplace=True)
        All_res_multiindex.drop(ref + '_Serie', axis=1, inplace=True)

    # Agregate more TAC ref for AP analysis. I agregate TAC using a reage (step)
    for ref in ['TAC_ref']:
        new = pd.DataFrame(columns=['value'], index=All_res_multiindex.index)
        step = 0.3 # g CaCO3/L
        agr = np.arange(0 - step, 31, step, dtype=float)  # I use a bigger interval to not treat the limit case
        for i in All_res_multiindex.index:
            value = round(All_res_INTRAserie['Ref_res_mean', ref].xs(i[0], level='Serie').values[0], 2)
            for ii in range(0, agr.size):
                # print(ii) # select the right interval
                if (value >= agr[ii] - step / 2) and (
                        value < agr[ii] + step / 2):  # since it is filterd it goes up and only one condition is enough
                    new['value'][i] = round(agr[ii], 2)
                    break
        All_res_multiindex[ref + '_Serie'] = new
        All_res_multiindex.set_index(All_res_multiindex[ref + '_Serie'], drop=True, append=True, inplace=True)
        All_res_multiindex.drop(ref + '_Serie', axis=1, inplace=True)
    return All_res_multiindex


def INTER_analysis(All_res_multiindex,All_res_INTRAserie, dict_param_relation):
    All_res_ALLserie = deepcopy(All_res_multiindex)
    All_res_ALLserie_dict = {}
    All_res_ALLserie_dict_raw = {}
    All_res_INTERserie_dict = {}
    All_res_INTERserie_dict_raw = {}

    for i in All_res_ALLserie.xs('Res_ref', axis=1).columns:
        ref = i.replace('_ref', '')
        Res_ref = deepcopy(All_res_ALLserie)
        Res_ref = erase_unusefull_info(Res_ref, ref)  # erase not used info in specific pd ref
        for ii in Res_ref.index.names:
            if ii not in ['Serie', 'Repetition', i + '_Serie']:
                Res_ref.reset_index(level=ii, drop=True, inplace=True)
        Res_ref = Res_ref.swaplevel(i=-3, j=-1)
        Res_ref = Res_ref.swaplevel(i=-2, j=-1)

        All_res_ALLserie_dict_raw[ref] = deepcopy(All_res_ALLserie)
        # All_res_ALLserie_dict_raw[ref] = erase_unusefull_info(All_res_ALLserie_dict_raw[ref], ref)
        All_res_ALLserie_dict_raw[ref].sort_index(level=-1, ascending=True, inplace=True, na_position='last')

        All_res_ALLserie_dict[ref] = pd.DataFrame()
        All_res_ALLserie_dict[ref] = compute_stat(Res_ref)
        All_res_ALLserie_dict[ref] = compute_profile_exactitude_global(dict_param_relation,
                                                                       data_global=All_res_multiindex,
                                                                       data_intra=All_res_INTRAserie,
                                                                       data_all=All_res_ALLserie_dict[ref], ref=ref,
                                                                       eq_des=None, serie='no')
        All_res_ALLserie_dict[ref].sort_index(level=-1, ascending=True, inplace=True, na_position='last')

        All_res_INTERserie_dict[ref] = compute_profile_exactitude_global(dict_param_relation,
                                                                         data_global=All_res_multiindex,
                                                                         data_intra=All_res_INTRAserie,
                                                                         data_all=All_res_ALLserie_dict[ref], ref=ref,
                                                                         eq_des=None, serie='yes')

    return All_res_ALLserie_dict, All_res_ALLserie_dict_raw, All_res_INTERserie_dict, All_res_INTERserie_dict_raw
def linear_regression(x=None, y=None):
    """
    Returns all the coeff of the linear regression.
    """
    X = deepcopy(x)
    Y = deepcopy(y)
    if (len(X) >1) and (len(Y) >1):
        # erase nan
        X_nan = X[X.isna()].index.tolist()
        X.drop(X_nan, inplace=True)
        Y.drop(X_nan, inplace=True)
        Y_nan = Y[Y.isna()].index.tolist()
        Y.drop(Y_nan, inplace=True)
        X.drop(Y_nan, inplace=True)

        X = X.values
        Y = Y.values
        X_length = X.size
        Y_length = Y.size
        X = X.reshape(X_length, 1)
        Y = Y.reshape(Y_length, 1)
        regr = linear_model.LinearRegression()
        regr.fit(X,Y)
        R2 = regr.score(X,Y)
        coeff = regr.coef_ # this gives all the coeff if several
        intercept = regr.intercept_
    else:
        logging.info('No data for linear regression')

    return regr, R2,coeff, intercept

def truncate_value(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return np.trunc(number)

    factor = 10.0 ** decimals
    return np.trunc(number * factor) / factor

def truncate_value_pd(data, decimals=1):
    """
    Returns dataframe with all truncated values
    """
    data_new = pd.DataFrame(index=data.index, columns= data.columns)
    for i in data.index:
        for ii in data.columns:
            if isinstance(data[ii][i], (int, float, np.integer)): # np.integer because int64 are not recognized as integer
                data_new[ii][i] = truncate_value(data[ii][i], decimals=decimals)

    return data_new

def round_value_pd(data, decimals=1):
    """
    Returns dataframe with all rounded values
    """
    data_new = pd.DataFrame(index=data.index, columns= data.columns)
    for i in data.index:
        for ii in data.columns:
            if isinstance(data[ii][i], (int, float, np.integer)): # np.integer because int64 are not recognized as integer
                data_new[ii][i] = np.around(data[ii][i], decimals=decimals)
            else:
                data_new[ii][i] = data[ii][i]

    return data_new

def round_value_serie(data, decimals=1):
    """
    Returns dataframe with all truncated values
    """
    data_new=deepcopy(data)
    for i in data.index:
        if isinstance(data[i], (int, float, np.integer)): # np.integer because int64 are not recognized as integer
            data_new[i] = np.around(data[i], decimals=decimals)

    return data_new

def calc_student(degree,beta=0.8):
    """ quantile of Student function based on:
    Beta: % probability to be inside the interval
    degree : freedon degrees
    https://fr.wikipedia.org/wiki/Loi_de_Student
    """
    if isinstance(degree, float): # if a float
        K_tol = abs(stats.t.isf((1+beta)/2, degree))
    else: # if a dataframe
        K_tol = pd.Series(np.nan, index=degree.index)
        for i in degree.index:
            K_tol[i] = abs(stats.t.isf((1+beta)/2, degree[i])) # the (1+beta)/2 is needed to use the double tail function beta = 80% --> error of 10% on one side

    # t_student = pd.DataFrame(index='degree', columns='beta')
    # for beta in range(0, 10, 1):
    #     for degree in range(1, 10, 1):
    #         value = -stats.t.isf(beta / 10, degree)
    #         logging.info('beta= ' + str(beta/10) + ', degree= ' + str(degree) + ', value= ' + str(value))
    #         t_student.iat[str(degree), str(beta)] = 1
    #         # t_student.loc['beta'][=-stats.t.isf(beta/10, degree)
    return K_tol

def parser(s):
    logging.info(s)
    folder_name, y_name, phrange, method, X = s.split('_')[:5]
    if len(s.split('_'))>5:
        other = '_'.join(s.split('_')[5:])
    else:
        other=''
    return tuple([y_name, phrange, method, X, other])

def add_metrics(ax, obs_data, pred_data, position = 0.75):
    # if position == None:
    #     position = 0.75
    metrics = dict()
    metrics['rmse'] = np.sqrt(mean_squared_error(obs_data, pred_data))
    # metrics['mae'] = mean_absolute_error(obs_data, pred_data)
    metrics['mad'] = median_absolute_error(obs_data, pred_data)
    # metrics['rmae'] = mean_absolute_percentage_error(obs_data, pred_data)
    metrics['R2'] = r2_score(obs_data, pred_data)
    ax.text(0.05, position,
             '$R^{2}$=%.2f\nRMSE=%.2f\nMAD=%.2f\n' % (
                 metrics['R2'], metrics['rmse'], metrics['mad']),
             ha='left', va='top', transform=ax.transAxes)
    return

def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

# logging.info(get_super('GeeksforGeeks'))
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


def create_All_pd(data_ref, data_SNAC):
    data_ref_std,data_ref_mean,data_ref_median,data_ref_nb,data_ref_CV = compute_stat(data_ref, idx, two_idx)
    data_SNAC_std,data_SNAC_mean,data_SNAC_median,data_SNAC_nb,data_SNAC_CV = compute_stat(data_SNAC, idx, two_idx)

    data_combined = pd.concat([data_ref_mean,
                                data_ref_std,
                                data_ref_median,
                                data_SNAC_mean,
                                data_SNAC_std,
                                data_SNAC_median,
                                data_ref_nb,
                                data_SNAC_nb], axis=1, verify_integrity=True,
                               keys=['Ref_res_mean',
                                     'Ref_res_std',
                                     'Ref_res_median',
                                     'SNAC_res_mean',
                                     'SNAC_res_std',
                                     'SNAC_res_median',
                                     'Ref_res_nb',
                                     'SNAC_res_nb'])

    return data_combined

def compute_stat(data):
    """ compute all basic stat on dataframe"""
    idx = data.index
    idx_name1 = data.index.names[0]
    idx_name2 = data.index.names[1]
    column_select = data.columns
    # computing mean and standard deviations
    df_std = pd.DataFrame(columns=column_select, index=idx.droplevel(level=[1,2]).drop_duplicates())
    df_mean = pd.DataFrame(columns=column_select, index=idx.droplevel(level=[1,2]).drop_duplicates())
    df_median = pd.DataFrame(columns=column_select, index=idx.droplevel(level=[1,2]).drop_duplicates())
    df_SCE = pd.DataFrame(columns=column_select, index=idx.droplevel(level=[1,2]).drop_duplicates())
    df_nb = pd.DataFrame(columns=column_select, index=idx.droplevel(level=[1,2]).drop_duplicates())
    df_CV = pd.DataFrame(columns=column_select, index=idx.droplevel(level=[1,2]).drop_duplicates())

    for i in idx.levels[0]:
        for ii in column_select:
            df_std.loc[i,ii] = data.xs(i, level=idx_name1)[ii].std()
            df_mean.loc[i,ii] = data.xs(i, level=idx_name1)[ii].mean()
            df_median.loc[i,ii] = data.xs(i, level=idx_name1)[ii].median()
            df_SCE.loc[i,ii] = ((data.xs(i, level=idx_name1)[ii] - df_mean.loc[i, ii])**2).sum()
            df_nb.loc[i,ii] = data.xs(i, level=idx_name1)[ii].count()
            df_CV.loc[i,ii] = df_std.loc[i,ii]/df_mean.loc[i,ii] * 100 # coefficient de variation
    df_std.loc[:,'Res_ref'].loc[:,'VFA_ref'] = np.nan  # VFA, TAN n'ont pas de std car c'est une seule valeur repetée
    df_std.loc[:,'Res_ref'].loc[:, 'TAN_ref'] = np.nan

    data_combined = pd.concat([df_mean.loc[:,'Res_ref'],
                                df_std.loc[:,'Res_ref'],
                                df_median.loc[:,'Res_ref'],
                                df_SCE.loc[:,'Res_ref'],
                                df_nb.loc[:, 'Res_ref'],
                                df_mean.loc[:,'Res_SNAC'],
                                df_std.loc[:,'Res_SNAC'],
                                df_median.loc[:,'Res_SNAC'],
                                df_SCE.loc[:,'Res_SNAC'],
                                df_nb.loc[:,'Res_SNAC'],
                               ], axis=1, verify_integrity=True,
                               keys=['Ref_res_mean',
                                     'Ref_res_std',
                                     'Ref_res_median',
                                     'Ref_res_SCE',
                                     'Ref_res_nb',
                                     'SNAC_res_mean',
                                     'SNAC_res_std',
                                     'SNAC_res_median',
                                     'SNAC_res_SCE',
                                     'SNAC_res_nb'])

    data_combined = truncate_value_pd(data_combined, decimals=3)

    return data_combined

def compute_profile_exactitude_simplified(data_old, param):
    """ source: 2010_Cahier_des_techniques dans WP05"""

    data = deepcopy(data_old)
    dict_param_relation = {'VFA':['VFA_geq','pls_VFA_geq','FOS_geq'], # ref : [all param with this ref]
                           'TAN':['TAN_geq','pls_TAN_geq'],
                           'TAC':['TAC_geq','IC_geq'],
                           'FOS':['FOS_geq'],
                           }
    profil_param = ['std_kr','std_kb', 'std_FI', 'biais_abs', 'biais_rel', 'Tr', 'std_IT', 'degree', 'k_tol', 'Itol_up_rel',
                    'Itol_down_rel']

    for j in dict_param_relation[param]:
        for i in profil_param:
            data['SNAC_res_' + i, j] = np.nan

        data['SNAC_res_std_kr', j] = data['SNAC_res_std',j] # ecart type repetabilié
        data['SNAC_res_std_kb',j] = 0 # ecart type interSerie #
        data['SNAC_res_std_FI',j] = (data['SNAC_res_std_kr',j]**2 + data['SNAC_res_std_kb',j]**2)**(0.5)# fidelité intermediaire. On pourrait rajouter d'autres erreurs aussi
        data['SNAC_res_biais_abs',j] = data['SNAC_res_mean',j] - data['Ref_res_mean',param+'_ref']
        data['SNAC_res_biais_rel',j] = data['SNAC_res_biais_abs',j] / data['Ref_res_mean',param+'_ref'] * 100
        data['SNAC_res_Tr',j] = data['SNAC_res_mean',j]/data['Ref_res_mean',param+'_ref'] * 100 #taux_recouvrement moyen

        R = data['SNAC_res_std_kb',j]**2 / data['SNAC_res_std_kr',j]**2
        J = 1 # number of replicates # todo: if any replicates put as a variable (FAUX: this should be the replicates)
        B = ((R+1)/(J*R+1))**0.5
        I = data['SNAC_res_nb',j] # number of test per condition
        data['SNAC_res_std_IT',j] = data['SNAC_res_std_FI',j]*((1+1/(I*J*B**2))**(0.5)) # ecart-type de tolérance

        data['SNAC_res_degree',j] = ((R +1)**2)/(((R +1/J)**2)/(I-1) + (1-1/J)/(I*J))
        data['SNAC_res_k_tol',j] = calc_student(data['SNAC_res_degree',j], beta=0.8) # todo: hypothèse de propabilité pour Student
        data['SNAC_res_Itol_up_rel',j] = (data['SNAC_res_mean',j] + data['SNAC_res_k_tol',j]*data['SNAC_res_std_IT',j])/data['Ref_res_mean',param+'_ref']*100 # interval de tolerance
        data['SNAC_res_Itol_down_rel',j] = (data['SNAC_res_mean',j] - data['SNAC_res_k_tol',j]*data['SNAC_res_std_IT',j])/data['Ref_res_mean',param+'_ref']*100 # interval de tolerance

    return data

def compute_profile_exactitude_global(dict_param_relation, data_global = None, data_intra = None, data_all = None, ref = None, eq_des = None, serie='yes'):
    """ source: 2010_Cahier_des_techniques dans WP05"""
    logging.info('ref= ' + ref)
    data_global_new = deepcopy(data_global)
    #data_intra_new = pd.DataFrame(index=data_intra.index, columns=data_intra.columns)
    data_intra_new = deepcopy(data_intra)

    data_inter_new = deepcopy(data_all)

    # dict_param_relation = {'VFA':['VFA_geq','pls_VFA_geq','FOS_geq','corr_FOS_geq','corr_Hach_FOS_geq'], # ref : [all param with this ref]
    #                        'TAN':['TAN_geq','pls_TAN_geq'],
    #                        'TAC':['TAC_geq','IC_geq'],
    #                        'FOS':['FOS_geq'],
    #                        }

    profil_param = ['eq_des','J','I','SCEr','SCEb','SCE_FI','N','n2','N_','var_r','var_b', 'var_FI', 'R','std_kr','std_kb', 'std_FI', 'std_IT', 'degree', 'k_tol', 'Rf', 'Itol_up_rel',
                    'Itol_down_rel', 'biais_abs', 'biais_rel', 'Uc','k', 'U', 'U_rel','error_real']
    # erase ancient profil param pd if any
    for i in data_inter_new.columns.levels[0].values:
        for ii in profil_param:
            if 'SNAC_res_'+ii == i:
                del data_inter_new[i]
    # create new profil param dict
    for i in profil_param:
        for ii in data_inter_new['SNAC_res_mean'].columns:
            data_inter_new['SNAC_res_'+i,ii] = np.nan


    for param in data_inter_new.xs('SNAC_res_nb', axis=1).columns:
        logging.info('param= '+param)
        # check all conditions for

        if len(data_inter_new['SNAC_res_mean', param]) <=3:
            logging.info("The data of " + param+ " have not enough level of concentration for accuracy profile")
            break

        if (data_inter_new['SNAC_res_nb', param] < 2).all():
            logging.info("At least one serie has not enough repetitions (2). We whould erase this serie")

        for i in data_inter_new.index: # i is the level or concentration
            # initialize to np.nan all variables
            #initialize_param()
            eq_des = np.nan
            J = np.nan
            I = np.nan
            SCEr = np.nan
            SCEb = np.nan
            SCE_FI = np.nan
            N = np.nan
            n2 = np.nan
            N_ = np.nan
            var_r = np.nan
            var_b = np.nan
            var_FI = np.nan
            R = np.nan
            std_kr = np.nan
            std_kb = np.nan
            std_FI = np.nan
            std_IT = np.nan
            degree = np.nan
            k_tol = np.nan
            Rf = np.nan
            Itol_up_rel = np.nan
            Itol_down_rel = np.nan
            biais_abs = np.nan
            biais_rel = np.nan
            Uc = np.nan # composed incertitude
            k = np.nan # facteur d'elargissement
            U = np.nan # incertitude (elergie)
            U_rel = np.nan # relative uncertainty
            error_real = np.nan # erreur à afficher au client

            try:
                SNAC_mean = data_inter_new['SNAC_res_mean',param][i]
                Res_mean = data_inter_new['Ref_res_mean', ref+'_ref'][i]
                SCE_FI = data_inter_new['SNAC_res_SCE', param][i]

                data_global_neww = deepcopy(data_global)
                data_global_neww.reset_index(level=['Serie', ref+ '_ref_Serie'], inplace=True)
                Tx_list = list(data_global_neww[data_global_neww[ref+ '_ref_Serie'] == i]['Serie'].drop_duplicates())

                # check if data is equilibrated or not on a level
                nb = []
                for Tx in Tx_list:
                    nb.append(data_intra_new.xs(Tx,level='Serie')['SNAC_res_nb',param].values)
                dup = [x for aa, x in enumerate(nb) if x not in nb[:aa]]
                if len(dup)>1:
                    eq_des = 'des'
                    logging.info("The data of "+str(param)+' level '+str(i)+ " are NOT equilibrated for accuracy profile")
                else:
                    eq_des = 'eq'
                    logging.info("The data of "+str(param)+' level '+str(i)+" are equilibrated for accuracy profile")
                data_inter_new['SNAC_res_eq_des', param][i] = eq_des


                if serie == 'yes': # I differentiate my anaysis in serie or not
                    I = len(Tx_list)
                # serie number

                    # not si plan équilibré : same number of repetition per serie
                    if eq_des == 'eq':
                        J = data_intra_new.xs(Tx_list[0],level='Serie')['SNAC_res_nb',param].values[0] # I pick one of the possible ones since they are all the same
                        SCEr = 0
                        for iii in Tx_list:
                            SCE_r_Tx = data_intra_new.xs(iii, level='Serie')['SNAC_res_SCE',param].values[0]
                            SCEr = SCEr + SCE_r_Tx
                        var_r = SCEr/(I*(J-1)) # ICannot use it since J is variable (plan desequilibre) so I compute differently with the J for each serie

                    # balanced plan
                    if eq_des == 'des':
                        SCEr = 0
                        N_=0 # total number of analysis
                        n2=0 # sum of all number of analysis per concentration ^2
                        for iii in Tx_list:
                            SCE_r_Tx = data_intra_new.xs(iii, level='Serie')['SNAC_res_SCE',param].values[0]
                            SCEr = SCEr + SCE_r_Tx
                            J_Tx = data_intra_new.xs(iii, level='Serie')['SNAC_res_nb', param].values[0]
                            #var_r = SCE_r_Tx/(I * (J_Tx -1)) # todo: verify that the use of J for Tx is good or not)
                            N_ = N_+ J_Tx
                            n2 = n2 + J_Tx ** 2

                        if N_ > 0:
                            N = N_ - (n2) / N_
                            J = N / (I - 1)
                        else:
                            J=np.nan

                        var_r = SCEr / (I * (J - 1))

                elif serie == 'no':
                    I = 1
                    J = data_inter_new['SNAC_res_nb', param][i]  # repetitions
                    SCEr = SCE_FI
                    var_r = SCEr / (I * (J - 1))


                SCEb = SCE_FI - SCEr
                var_b = (SCEb / (I - 1) - var_r) / J
                if (SCEb <= 0) or (var_b <0): # security condition
                    SCEb = 0
                    var_b = 0

                var_FI = var_r + var_b

                std_kr = var_r ** 0.5
                std_kb = var_b ** 0.5
                std_FI = var_FI ** 0.5
                biais_abs = SNAC_mean - Res_mean
                biais_rel = biais_abs / Res_mean * 100
                Rf = SNAC_mean/Res_mean * 100 # taux_recouvrement moyen

                R = (std_kb ** 2) / (std_kr ** 2)
                B = ((R + 1) / (J * R + 1)) ** 0.5
                std_IT = std_FI*((1+1/(I*J*B**2))**(0.5)) # ecart-type de tolérance
                if serie == 'yes': # it gives degree = 0 when I=1 so I must separate them
                    degree = ((R +1)**2)/(((R +1/J)**2)/(I-1) + (1-1/J)/(I*J))
                elif serie == 'no':
                    degree = J-1

                k_tol = calc_student(degree, beta=0.8)
                Itol_up_rel = (SNAC_mean + k_tol*std_IT)/Res_mean*100 # interval de tolerance
                Itol_down_rel= (SNAC_mean- k_tol*std_IT)/Res_mean*100 # interval de tolerance

                # computing the uncertainties and error to announce
                Uc = std_IT
                if np.isnan(k_tol):
                    k=1 # 2 for uncertainties (95%) - use 3 for uncertainty (99%)
                else:
                    k = k_tol  # use the k of accuracy profile
                U = k * Uc
                U_rel = U/SNAC_mean *100

                error_real = max(abs(Itol_up_rel-100),abs(Itol_down_rel-100))  # relative uncertainty

                for ii in profil_param:
                    if ii in locals():
                        data_inter_new['SNAC_res_'+ii, param][i] = locals()[ii]

            except:
                for ii in profil_param:
                    data_inter_new['SNAC_res_' + ii, param][i] = np.nan
                logging.error('Something in accuracy profile for '+str(param)+', level "'+str(i)+'" went wrong')

    return data_inter_new

def accep_interval(abs_before_1 = 0.2, rel_after_1 = 20, limit_graphs=[0.1,1,1]):
    acceptability_interval_up = 100 + rel_after_1
    acceptability_interval_down = 100 - rel_after_1

    # create manual interval < 1 gAc/L or 1 g N/L

    xlist = np.linspace(limit_graphs[0],limit_graphs[2], 200)
    ylistplusrelative = []
    ylistminusrelative = []
    for i in xlist[0:]:
        if i <=limit_graphs[1]:
            abs_value = (abs_before_1 / i) * 100
            ylistplusrelative.append(100 + abs_value)
            ylistminusrelative.append(100 - abs_value)
        else:
            ylistplusrelative.append(100 + rel_after_1)
            ylistminusrelative.append(100 - rel_after_1)


#    plt.plot(xlist, ylistplusrelative)
#    plt.plot(xlist, ylistminusrelative)

    return xlist, ylistplusrelative, ylistminusrelative

def accep_interval_bisector(abs_before_1 = 0.2, rel_after_1 = 20, limit_graphs=[0.1,1,1]):
    acceptability_interval_up = 100 + rel_after_1
    acceptability_interval_down = 100 - rel_after_1

    # create manual interval < 1 gAc/L or 1 g N/L

    xlist = np.linspace(limit_graphs[0],limit_graphs[2], 200)
    ylistplusrelative = []
    ylistminusrelative = []
    for i in xlist[0:]:
        if i <=limit_graphs[1]:
            abs_value = (abs_before_1 / i) * 100
            ylistplusrelative.append(100 + abs_value)
            ylistminusrelative.append(100 - abs_value)
        else:
            ylistplusrelative.append(100 + rel_after_1)
            ylistminusrelative.append(100 - rel_after_1)


#    plt.plot(xlist, ylistplusrelative)
#    plt.plot(xlist, ylistminusrelative)

    return xlist, ylistplusrelative, ylistminusrelative


def erase_unusefull_info(data, param):
    data_new = deepcopy(data)
    variables = ['VFA', 'pls_VFA', 'TAN', 'pls_TAN', 'TAC', 'IC', 'FOS']
    dict_param_relation = {'VFA': ['VFA', 'pls_VFA', 'FOS'],  # ref : [all param with this ref]
                           'TAN': ['TAN', 'pls_TAN'],
                           'TAC': ['TAC', 'IC'],
                           'FOS': ['FOS'],
                           }
    for col_name in data_new.columns:  # erase all not used values
        if ('pH') in col_name[1] or ('cond') in col_name[1] or ('FOS_TAC') in col_name[1]:
            data_new.drop(columns=col_name, inplace=True)
        else:
            for j in variables:
                if j in col_name[1]:
                    col_name_true = j
            if col_name_true not in dict_param_relation[param]:
                # logging.info('erase :' + str(col_name) + 'param: ' + param)
                data_new.drop(columns=col_name, inplace=True)

    return data_new

def initialize_param():
    eq_des = np.nan
    J = np.nan
    I = np.nan
    SCEr = np.nan
    SCEb = np.nan
    SCE_FI = np.nan
    N = np.nan
    n2 = np.nan
    N_ = np.nan
    var_r = np.nan
    var_b = np.nan
    var_FI = np.nan
    R = np.nan
    std_kr = np.nan
    std_kb = np.nan
    std_FI = np.nan
    std_IT = np.nan
    degree = np.nan
    k_tol = np.nan
    Rf = np.nan
    Itol_up_rel = np.nan
    Itol_down_rel = np.nan
    biais_abs = np.nan
    biais_rel = np.nan
    return

# def estimator_agregation(reference=['TAC_ref'], type='range', data_loop=data_l, data_calc=data_c, others=None):
#     """this function create reference vector allowing the agregation of data following different rules:
#     average, imposed range, round, etc"""
#     if type == 'range':
#         for ref in reference:
#             new = pd.DataFrame(columns=['value'], index=All_res_twoindex.index)
#             step = 0.5
#             agr = np.arange(-1, 31, step)  # I use a bigger interval to not treat the limit case
#             for i in All_res_twoindex.index:
#                 value = round(All_res_INTRAserie['Ref_res_mean', ref].xs(i[0], level='Serie').values[0], 2)
#                 for ii in range(0, agr.size):
#                     # print(ii) # select the right interval
#                     if (value >= agr[ii] - step / 2) and (
#                             value < agr[ii] + step / 2):  # since it is filterd it goes up and only one condition is enough
#                         new['value'][i] = agr[ii]  # rounding by 3 means that no data is agregated
#                         break
#             All_res_twoindex[ref + '_Serie'] = new
#             All_res_twoindex.set_index(All_res_twoindex[ref + '_Serie'], drop=True, append=True, inplace=True)
#             All_res_twoindex.drop(ref + '_Serie', axis=1, inplace=True)
#     return

def metainfo(All_res_INTERserie_dict):
    # metainfo

    # I put in a table the information of inter data to put  in the livrable.
    All_res_INTERserie_dict_metainfo = deepcopy(All_res_INTERserie_dict)
    ref_imp = {'VFA': ['pls_VFA_geq'], 'TAN': ['TAN_geq'], 'TAC': ['TAC_geq']}
    parameter_imp = {'Ref_res_mean': 1, 'SNAC_res_std': 3, 'SNAC_res_mean': 1, 'SNAC_res_k': 1, 'SNAC_res_U': 1,
                     'SNAC_res_U_rel': 0, 'SNAC_res_error_real': 0}  # :'param' :  number --> decimel to round value
    #:format(count_pH_acid/count, ".0%" All_res_INTERserie_dict_metainfo[ref][param].style.format("{:.0%}")
    for ref in All_res_INTERserie_dict:
        if ref in ref_imp:
            for param in All_res_INTERserie_dict[ref].columns.levels[0].values:
                if param not in parameter_imp:
                    del All_res_INTERserie_dict_metainfo[ref][param]
                else:  # todo : it does not work for now
                    All_res_INTERserie_dict_metainfo[ref][param] = round_value_pd(
                        All_res_INTERserie_dict_metainfo[ref][param], decimals=parameter_imp[param])
                    for estimator in All_res_INTERserie_dict_metainfo[ref][param]:
                        if estimator not in ref_imp[ref]:
                            del All_res_INTERserie_dict_metainfo[ref][param][estimator]
        else:
            del All_res_INTERserie_dict_metainfo[ref]

    parameter_imp = {'SNAC_res_std_aggr': 1, 'SNAC_res_std_aggr_rel': 1}
    agrr = [[0, 2], [2, 10], [10, 30]]
    # b=deepcopy(All_res_INTERserie_dict_metainfo)
    c = All_res_INTERserie_dict_metainfo
    for estimator in ['IC_geq', 'TAC_geq']:
        new = pd.DataFrame(columns=['SNAC_res_std_aggr'], index=c['TAC'].index)
        new_rel = pd.DataFrame(columns=['SNAC_res_std_aggr_rel'], index=c['TAC'].index)
        for i in new.index:
            # print(i)
            if not np.isnan(i):
                for ii in agrr:  # select the right interval
                    if (i >= ii[0]) and (i < ii[1]):
                        sel = ii
                        break
                selected_data = c['TAC'].loc[(c['TAC']['SNAC_res_std'][estimator].index > sel[0]) & (
                            c['TAC']['SNAC_res_std'][estimator].index <= sel[1])]
                observed_mean = selected_data['SNAC_res_mean'][estimator].mean()
                new.loc[i]['SNAC_res_std_aggr'] = selected_data['SNAC_res_std'][
                    estimator].median()  # obseved std median of selected data
                new_rel.loc[i]['SNAC_res_std_aggr_rel'] = format(new.loc[i]['SNAC_res_std_aggr'] / observed_mean, ".1%")
        c['TAC']['SNAC_res_std_aggr', estimator] = new
        c['TAC']['SNAC_res_std_aggr_rel', estimator] = new_rel
    return All_res_INTERserie_dict_metainfo
def save_Excel(All_res_normal_metainfo,All_res_INTRAserie,All_res_normal_raw,All_res_ALLserie_dict,
               All_res_INTERserie_dict, All_res_INTERserie_dict_metainfo, All_res_INTERserie_dict_raw,
               raw_output_folder_path,file_SNAC):
    # SAVING IN EXCEL
    # put something else in the metainfo
    All_res_normal_metainfo['All']['Total conditions'] = All_res_INTRAserie.shape[
        0]  # we need to keep all the analysis in the data reference
    # I need to trier with for loop since I cannot do it with index
    for i in ['T1', 'T2', 'T3', 'T4']:
        All_res_normal_metainfo[i]['Total conditions'] = All_res_INTRAserie.loc[i, :].shape[0]
        logging.info('creating meta info from All_res_normal. not valid for other files input')

        # Saving results table
    All_res_normal = truncate_value_pd(All_res_INTRAserie, decimals=3)
    All_res_normal_raw = truncate_value_pd(All_res_normal_raw, decimals=3)

    # save all data with specific analysis
    writer = pd.ExcelWriter(raw_output_folder_path + file_SNAC.replace('.csv', '') + ' intra_results.xlsx')
    All_res_normal_raw.to_excel(writer, sheet_name='res_raw', startcol=1, startrow=1, freeze_panes=(3, 2))
    All_res_normal.to_excel(writer, sheet_name='res_analysed', startcol=1, startrow=1, freeze_panes=(3, 2))
    All_res_normal_metainfo.to_excel(writer, sheet_name='metainfo', startcol=1, startrow=1, freeze_panes=(2, 2))
    writer.save()
    writer.close()

    # save all data with global analysis
    writer = pd.ExcelWriter(raw_output_folder_path + file_SNAC.replace('.csv', '') + ' all_results.xlsx')
    for i in All_res_ALLserie_dict:
        round_value_pd(All_res_ALLserie_dict[i], decimals=2).to_excel(writer, sheet_name=str(i), startcol=1, startrow=1,
                                                                      freeze_panes=(3, 2))
    writer.save()
    writer.close()

    # save all data with inter serie analysis (complete analysis)
    writer = pd.ExcelWriter(raw_output_folder_path + file_SNAC.replace('.csv', '') + ' inter_results.xlsx')
    for i in All_res_INTERserie_dict:
        round_value_pd(All_res_INTERserie_dict[i], decimals=2).to_excel(writer, sheet_name=str(i), startcol=1,
                                                                        startrow=1, freeze_panes=(3, 3))

    for i in All_res_INTERserie_dict_metainfo:
        All_res_INTERserie_dict_metainfo[i].to_excel(writer, sheet_name=i + ' metainfo', startcol=1, startrow=1,
                                                     freeze_panes=(3, 3))

    for i in All_res_INTERserie_dict_raw:
        round_value_pd(All_res_INTERserie_dict_raw[i], decimals=2).to_excel(writer, sheet_name=str(i), startcol=1,
                                                                            startrow=1, freeze_panes=(3, 3))

    writer.save()
    writer.close()

    logging.info('Files saved correctly')
    return








