import os
from datasciencetools.base import *
from datasciencetools.eda import *
from datasciencetools.plot import *
from datasciencetools.metrics import *
from datasciencetools.modeling import *
from datasciencetools.preprocessing import *
from datasciencetools.multiblock import *
import logging
import time
import argparse
import json
import warnings
# import inputparse
import sys

def check_config(compounds, compounds_config, config):
    error = 0
    if ('cond' in config['x_dataset_type'] and compounds_config[compounds]['filters'][3][1][0] == 0) or ('cond' not in config['x_dataset_type'] and compounds_config[compounds]['filters'][3][1][0] == 1):
        error = 1
        logging.error('Conductivity signal filters should be activated if the conductivity signal is chosen, or the opposite. Please check')
    if (compounds_config[compounds]['drop']['manual_conductivity'] == True and compounds_config[compounds]['filters'][3][1][0] == 0) or (compounds_config[compounds]['drop']['manual_conductivity'] == False and compounds_config[compounds]['filters'][3][1][0] == 1):
        error = 1
        logging.error('Conductivity filter "drop" and automatic are not coherent. Please check')
    if (compounds_config[compounds]['model_type']['regression']['type'] == 'epoplsr' and compounds_config[compounds]['filters'][1][1][0] == 1) or (compounds_config[compounds]['model_type']['regression']['type'] != 'epoplsr' and compounds_config[compounds]['filters'][1][1][0] == 0):
        error = 1
        logging.error('if "epo" model is used the "indip_filter" should be [0,1], or the opposite. Please check')
    if (compounds_config[compounds]['model_type']['regression']['type'] == 'mbplsr' and 'volume_conductivity' not in config['x_dataset_type']) or (compounds_config[compounds]['model_type']['regression']['type'] != 'mbplsr' and 'volume_conductivity' in config['x_dataset_type']):
        error = 1
        logging.error('if "mbplsr" model is chosen both the volume and the conductivity signal should be used, or the opposite. Please check')
    if (compounds_config[compounds]['preprocessing']['type_data'] == 'raw' and 'sg1' not in compounds_config[compounds]['preprocessing']['steps']) or (compounds_config[compounds]['preprocessing']['type_data'] != 'raw' and 'sg1' in compounds_config[compounds]['preprocessing']['steps']):
        error = 1
        logging.error('If raw signal is used the derivative should be applied, or the opposite. Please check')

    if error == 1:
        answer_ok = False
        count = 0
        while answer_ok == False and count <= 5:
            count = count + 1
            action = input('Enter "y" to continue or "n" to stop the run')

           # understancing the user action
            if action == 'y':
                answer_ok = True
                pass
            elif action =='n' or count == 5:
                sys.exit()
            else:
                answer_ok = False
                logging.error('Input incorrect, try again: "y" or "n"')

    return


def apply_filter(data, filters_list):
    data_filtered = deepcopy(data)
    for f in filters_list:
        data_filtered = data_filtered[(data_filtered[f[0]] >= f[1][0]) & (data_filtered[f[0]] <= f[1][1])]
    return data_filtered


# def apply_filter_negative(data, filters_list):
#     data_filtered = deepcopy(data)
#     for f in filters_list:
#         data_filtered = data_filtered[~(data_filtered[f[0]] >= f[1][0]) & (data_filtered[f[0]] <= f[1][1])]
#     return data_filtered

# def data_filtering(merged_Res_analyselabo, config,args):
def data_filtering(merged_Res_analyselabo, config,custom_criteria,args):
##### Filters all the data in relation to defined filters

    indip_samples_list=[]
    if config['compute_metrics']['compute_with_indip_samples_only']['activate']:
        logging.info('# Getting list of samples where (%s !=1) in merged_Res_analyselabo for computing metrics without them'%config['compute_metrics']['compute_with_indip_samples_only']['type'])
        f = (config['compute_metrics']['compute_with_indip_samples_only']['type'], [1,1])
        indip_samples_list = merged_Res_analyselabo[~(merged_Res_analyselabo[f[0]] >= f[1][0]) & (merged_Res_analyselabo[f[0]] <= f[1][1])].index.tolist()


    # trasform True in 1 and False in 0 for ease the filtering function
    signal_list=['101_0','102_0','103_0'] # pH acid signal accepted, pH basic signal accepted, bacid conductivity accepted
    for i in signal_list:
        merged_Res_analyselabo.loc[merged_Res_analyselabo[i] == True, i] = 1
        merged_Res_analyselabo.loc[merged_Res_analyselabo[i] == False, i] = 0

    if config['filtering']['activate']:
        logging.info('# Filtering merged_Res_analyselabo with filters:%s'%config['filtering']['filters'])
        for f in config['filtering']['filters']:
            merged_Res_analyselabo = merged_Res_analyselabo[(merged_Res_analyselabo[f[0]] >= f[1][0]) & (merged_Res_analyselabo[f[0]] <= f[1][1])]
            logging.info('merged_Res_analyselabo: %s, filter: %s' % (merged_Res_analyselabo.shape, f))
            # nan_count = merged_Res_analyselabo[f[0]].isna().sum()
            # merged_Res_analyselabo = merged_Res_analyselabo[(merged_Res_analyselabo[f[0]].isna()==False)]
            # logging.info('merged_Res_analyselabo: %s because nan in filter: %s' % (merged_Res_analyselabo.shape, nan_count))
            custom_criteria = custom_criteria[custom_criteria.index.isin(merged_Res_analyselabo.index.unique())]
        logging.info('merged_Res_analyselabo: %s, custom_criteria: %s'%(merged_Res_analyselabo.shape,custom_criteria.shape))

    # drop some samples
    if config['outliers_drop']['activate']:
        if 'list' in config['outliers_drop']['method']:
            outliers_list = config['outliers_drop']['outliers']
            logging.info('# Dropping some outliers: %s' % config['outliers_drop']['outliers'])
            if 'conductivity' in args.x_dataset_type:
                outliers_list = config['outliers_drop']['outliers'] + config['outliers_drop']['outliers_conductivity']
                logging.info('# Dropping some outliers for conductivity (automatic activation): %s' % config['outliers_drop']['outliers_conductivity'])
            if config['outliers_drop']['activate_conductivity_manual']:
                outliers_list = config['outliers_drop']['outliers'] + config['outliers_drop']['outliers_conductivity']
                logging.info('# Dropping some outliers for conductivity (manual activation): %s' % config['outliers_drop']['outliers_conductivity'])
            if len(config['outliers_drop']['source']) >= 1 :
                for ii in config['outliers_drop']['source']:
                    for j in ii[1]:
                        drop_list = merged_Res_analyselabo[merged_Res_analyselabo[ii[0]] == j].index.to_list()
                        logging.info('# Dropping the remaining %s analyses related to "Project" = %s : %s' % (len(drop_list),j,drop_list))
                        outliers_list = outliers_list + drop_list
            merged_Res_analyselabo.drop(index=outliers_list, inplace=True, errors='ignore')
            custom_criteria.drop(index=outliers_list, inplace=True, errors='ignore')
        logging.info('merged_Res_analyselabo: %s' %(str(merged_Res_analyselabo.shape)))

        if 'filter' in config['outliers_drop']['method']:
            logging.info('# Dropping some outliers based on filters:%s' % config['outliers_drop']['outliers_filters'])
            for f in config['outliers_drop']['outliers_filters']:
                merged_Res_analyselabo = merged_Res_analyselabo[(merged_Res_analyselabo[f[0]] >= f[1][0]) & (merged_Res_analyselabo[f[0]] <= f[1][1])]
                custom_criteria = custom_criteria[custom_criteria.index.isin(merged_Res_analyselabo.index.unique())]
                logging.info('# Dropping some outliers based on filter: %s' % (str(f)))
        logging.info('merged_Res_analyselabo: %s, custom_criteria: %s'%(merged_Res_analyselabo.shape, custom_criteria.shape))

    return merged_Res_analyselabo, custom_criteria,indip_samples_list
    # return merged_Res_analyselabo,indip_samples_list


def data_raw_filter(merged_Res_analyselabo, measurements_vol, measurements_cond, config, custom_criteria, sub_output_path):
    logging.info('# Chosen y_variable: %s' % config['y_variable'])
    y_raw = merged_Res_analyselabo.loc[:,[config['y_variable']]]
    logging.info('y_raw: %s'%(y_raw.shape,))

    logging.info('  # Chosen x_dataset_type: %s' % config['x_dataset_type'])
    if config['x_dataset_type'] == 'volume':
        X_raw = measurements_vol
    elif config['x_dataset_type'] == 'volume_dilutioncorrected':
        logging.info('We select volume signal')
        X_raw = measurements_vol.merge(merged_Res_analyselabo.loc[:,['0_4']], left_index=True, right_index=True, how='left').dropna()
        X_raw = X_raw.iloc[:,:-1].multiply(X_raw['0_4'], axis="index")
        X_raw = X_raw.merge(merged_Res_analyselabo.loc[:, ['0_5']], left_index=True, right_index=True,how='left').dropna()
        X_raw = X_raw.iloc[:, :-1].multiply(X_raw['0_5'], axis="index")
    elif config['x_dataset_type'] == 'conductivity':
        logging.info('We select conductivity signal')
        X_raw = measurements_cond
    elif config['x_dataset_type'] == 'conductivity_dilutioncorrected':
        logging.info('We select conductivity signal')
        X_raw = measurements_cond.merge(merged_Res_analyselabo.loc[:,['0_4']], left_index=True, right_index=True, how='left').dropna()
        X_raw = X_raw.iloc[:,:-1].multiply(X_raw['0_4'], axis="index")
        X_raw = X_raw.merge(merged_Res_analyselabo.loc[:, ['0_5']], left_index=True, right_index=True,how='left').dropna()
        X_raw = X_raw.iloc[:, :-1].multiply(X_raw['0_5'], axis="index")
    elif config['x_dataset_type'] == 'custom_criteria':
        logging.info('We select custom criteria')
        X_raw = custom_criteria
    elif config['x_dataset_type'] == 'mb_volume_conductivity':
        logging.info('We select volume and conductivity signal')
        X1 = measurements_vol.merge(merged_Res_analyselabo.loc[:, ['0_4']], left_index=True, right_index=True, how='left').dropna()
        X1 = X1.iloc[:, :-1].multiply(X1['0_4'], axis="index")
        X1 = X1.merge(merged_Res_analyselabo.loc[:, ['0_5']], left_index=True, right_index=True, how='left').dropna()
        X1 = X1.iloc[:, :-1].multiply(X1['0_5'], axis="index")
        X2 = measurements_cond.merge(merged_Res_analyselabo.loc[:, ['0_4']], left_index=True, right_index=True, how='left').dropna()
        X2 = X2.iloc[:, :-1].multiply(X2['0_4'], axis="index")
        X2 = X2.merge(merged_Res_analyselabo.loc[:, ['0_5']], left_index=True, right_index=True, how='left').dropna()
        X2 = X2.iloc[:, :-1].multiply(X2['0_5'], axis="index")
        X_raw = [X1, X2]
    else:
        logging.error('Unknown x_dataset_type: %s'%config['x_dataset_type'])


    logging.info('# Merging X_raw and y_raw')
    if isinstance(X_raw, list) :
        logging.info('X_raw is multiblock, special processing chain')
        for i, block in enumerate(X_raw):
            logging.info('X_raw, block%d: %s' % (i+1, block.shape,))
        X, y = merge_multiblockXY(X_list=X_raw, Y=y_raw)
        for i, block in enumerate(X):
            logging.info('X_merged, block%d: %s' % (i+1, block.shape,))
        try:
            Xnew = []
            for block in X:
                block.columns = block.columns.astype(np.float64)
                Xnew.append(block)
                config['plot']['loadings_kind']='lineplot'
                X = Xnew.copy()
        except TypeError:
            logging.error('X columns not convertable to float')
            config['plot']['loadings_kind']='barplot'

    else:
        logging.info('X_raw is single block')
        logging.info('X_raw: %s' % (X_raw.shape,))
        merged_xy = X_raw.merge(y_raw, how='inner', left_index=True, right_index=True)
        logging.info('merged_xy: %s'%(merged_xy.shape,))

        logging.info('# Drop na values of merged_xy')
        merged_xy.dropna(inplace=True)
        logging.info('merged_xy: %s'%(merged_xy.shape,))

        # define X and errors:
        X = merged_xy.loc[:,X_raw.columns]
        y = merged_xy.loc[:,y_raw.columns]
        try:
            X.columns = X.columns.astype(np.float64)
            config['plot']['loadings_kind']='lineplot'
        except TypeError:
            logging.error('X columns not convertable to float')
            config['plot']['loadings_kind']='barplot'
        logging.info('X:%s, y:%s'%(X.shape,y.shape))

        fig, ax = plt.subplots(1,1) # todo: put into plot page
        X.T.plot(ax=ax, legend=False)
        if config['plot']['loadings_kind']=='barplot':
            ax.set_xticks(range(X.shape[1]))
            ax.set_xticklabels(X.columns, rotation=90)
        #ax.legend(fontsize='x-small')
        fig.tight_layout()
        fig.savefig(sub_output_path + 'X.png')

    if config['preprocessing']['activate']:
        if isinstance(X, list):
            Xnew = []
            for i, X_block in enumerate(X):
                logging.info('# preprocessing X block %d with following steps: %s:' % (i,config['preprocessing']['steps'].keys(),))
                for step_name, step_preprocessor in config['preprocessing']['steps'].items():
                    logging.info('>>> %s' % step_name)
                    X_block = step_preprocessor.fit(X_block).transform(X_block)
                Xnew.append(X_block)
                fig, ax = plt.subplots(1, 1) # todo: put into plot page
                X_block.T.plot(ax=ax, legend=False)
                if config['plot']['loadings_kind'] == 'barplot':
                    ax.set_xticks(range(X_block.shape[1]))
                    ax.set_xticklabels(X_block.columns, rotation=90)
                # ax.legend(fontsize='x-small')
                ax.set_title('X_preprocessed - block%d'%i)
                fig.tight_layout()
                fig.savefig(sub_output_path + 'X_preprocessed_b%d.png'%i)
            X = Xnew.copy()
        else:
            logging.info('# preprocessing X with following steps: %s:'%config['preprocessing']['steps'].keys())
            for step_name, step_preprocessor in config['preprocessing']['steps'].items():
                logging.info('>>> %s'%step_name)
                X = step_preprocessor.fit(X).transform(X)
            fig, ax = plt.subplots(1, 1)# todo: put into plot page
            X.T.plot(ax=ax, legend=False)
            if config['plot']['loadings_kind'] == 'barplot':
                ax.set_xticks(range(X.shape[1]))
                ax.set_xticklabels(X.columns, rotation=90)
            # ax.legend(fontsize='x-small')
            fig.tight_layout()
            fig.savefig(sub_output_path + 'X_preprocessed.png')

    #normalize X
    if config['normalize']:
        if isinstance(X, list):
            logging.error('Cannot normalize with multiblock methods MBPLSR') # todo: treat this case
        else:
            logging.info('# Normalizing X')
            ss = StandardScaler(with_mean=True, with_std=True)
            X = pd.DataFrame(ss.fit_transform(X), index=X.index, columns=X.columns)
            fig, ax = plt.subplots(1,1)# todo: put into plot page
            X.T.plot(ax=ax, legend=False)
            if config['plot']['loadings_kind'] == 'barplot':
                ax.set_xticks(range(X.shape[1]))
                ax.set_xticklabels(X.columns, rotation=90)
            # ax.legend(fontsize='x-small')
            fig.tight_layout()
            fig.savefig(sub_output_path + 'X_preprocessed_normalized.png')

    return X, y