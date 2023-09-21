import os
import logging
import time
import pandas
from glob import glob
import pandas as pd
from copy import deepcopy

# input_path ="C:/Users/sebas/Dropbox (BioEnTech)/CORPORATE/0 ORGANISATION DROPBOX CORPORATE/12-SNAC/2-Developpement/1-Developpement algo/4 - Analyse stat/"
# output_path = './projects/snac_eda/outputs/'

folder_name = str('T21')
input_path ="/home/silvio/Dropbox (BioEnTech)/CORPORATE/0 ORGANISATION DROPBOX CORPORATE/12-SNAC/2-Developpement/1-Developpement algo/4 - Analyse stat/"
# output_path = './projects/snac_eda/outputs/'
output_path ="/home/silvio/Dropbox (BioEnTech)/CORPORATE/0 ORGANISATION DROPBOX CORPORATE/12-SNAC/2-Developpement/1-Developpement algo/4 - Analyse stat/"+folder_name+'/'

logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)-15s][%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
            # handlers=(
            #     # logging.FileHandler(config['project_path'] + "logs/log_"+args.run_id+".log", mode='w'),
            #     logging.FileHandler(output_path + "/../logs/log_consolidateresults_"+time.strftime("%Y%m%d-%H%M%S")+".log", mode='w'),
            #     logging.StreamHandler()
            # )
)

all_metrics = []

for folder_id in [folder_name]:
    folder_fn = input_path + folder_id + '/'
    for cur_experiment_folder in glob(folder_fn + "*"):
        try:
            cur_experiment_name = cur_experiment_folder.split('/')[-1]
            logging.info(cur_experiment_name)
            cur_log = open(glob(cur_experiment_folder + '/log_*')[0], "r").readlines()

            def line2metrics(line):
                cur_metrics = {}
                temp, line = line.split('RMSEc=')
                cur_metrics['RMSEC'], line = line.split(', RMSEp=')
                cur_metrics['RMSEP'], line = line.split(', MAEc=')
                cur_metrics['MAEC'], line = line.split(', MAEp=')
                cur_metrics['MAEP'], line = line.split(', MADc=')
                cur_metrics['MADC'], line = line.split(', MADp=')
                cur_metrics['MADP'], line = line.split(', RMAEc=')
                cur_metrics['RMAEC'], line = line.split(', RMAEp=')
                cur_metrics['RMAEP'], line = line.split(', $R^{2}$c=')
                cur_metrics['R2C'], line = line.split(', $R^{2}$p=')
                cur_metrics['R2P'], line = line.split(', $r^{2}$c=')
                cur_metrics['r2C'], line = line.split(', $r^{2}$p=')
                cur_metrics['r2P'], line = line.split(',biasc=')
                cur_metrics['biasC'], line = line.split(', biasp=')
                cur_metrics['biasP'], line = line.split(', sepc=')
                cur_metrics['sepC'], line = line.split(', sepp=')
                cur_metrics['sepP'], line = line.split('\n')
                cur_metrics = {k:float(v) for k,v in cur_metrics.items()}
                return cur_metrics


            def line2metrics_specific(df, log, key, name):
                line = [line for line in log if key in line][0]
                temp, line = line.split(key)
                df[name], line = line.split('\n')
                df = {k: float(v) for k, v in df.items()}
                return df

            cur_metrics_line_pls = [line for line in cur_log if 'Metrics with model:' in line][0] # I add: to ditinguish from "Metrics with model without ignored samples"
            cur_metrics_pls = line2metrics(cur_metrics_line_pls)

            cur_metrics_pls = line2metrics_specific(cur_metrics_pls,cur_log,'nb.train with model:','nb_train')
            cur_metrics_pls = line2metrics_specific(cur_metrics_pls, cur_log, 'nb.test with model:', 'nb_test')
            cur_metrics_pls = line2metrics_specific(cur_metrics_pls, cur_log, 'Components number with model:', 'nb_comp')
            cur_metrics_pls_df = pd.DataFrame(index= [cur_experiment_name], columns = cur_metrics_pls.keys(),data=[cur_metrics_pls.values()])

            cur_metrics_line_snac = [line for line in cur_log if 'Metrics with SNAC' in line][0]
            cur_metrics_snac = line2metrics(cur_metrics_line_snac)

            cur_metrics_snac_df = pd.DataFrame(index=[cur_experiment_name.replace('mbplsr', 'snac').replace('epoplsr', 'snac').replace('plsr', 'snac')], columns=cur_metrics_snac.keys(),
                                              data=[cur_metrics_snac.values()])

            try:
                cur_metrics_line_stacked = [line for line in cur_log if 'Metrics with stacked model:' in line][0]
                cur_metrics_stacked = line2metrics(cur_metrics_line_stacked)
                cur_metrics_stacked_df = pd.DataFrame(index=[cur_experiment_name.replace('mbplsr', 'stacking').replace('epoplsr', 'stacking').replace('plsr', 'stacking')], columns=cur_metrics_stacked.keys(),
                                                  data=[cur_metrics_stacked.values()])
                # if cur_metrics_stacked_df is empty:
                #     cur_metrics_stacked_df
                # if not isinstance(cur_metrics_stacked_df, type(None))
                cur_metrics = pd.concat([cur_metrics_pls_df, cur_metrics_snac_df, cur_metrics_stacked_df],axis=0)
            except:
                logging.info('no Stacking in %s - error' % cur_experiment_folder)
                cur_metrics = pd.concat([cur_metrics_pls_df, cur_metrics_snac_df], axis=0)
                pass

            all_metrics.append(cur_metrics)
        except:
            logging.info('%s - error in this folder'%cur_experiment_name)
            pass

def parser(s):
    folder_name, y_name, phrange, method, X = s.split('_')[:5]
    if len(s.split('_'))>5:
        other = '_'.join(s.split('_')[5:])
    else:
        other=''
    return tuple([folder_name, y_name, phrange, method, X, other])

all_metrics_df = pd.concat(all_metrics, axis=0)
all_metrics_df.index = pd.MultiIndex.from_tuples(all_metrics_df.index.map(parser), names=['folder_name', 'y_name', 'phrange', 'method', 'X', 'other'])
all_metrics_df.reset_index(inplace=True)

# chose which snac metrics keep
SNAC_metrics = all_metrics_df[all_metrics_df['method']=='snac'].drop_duplicates(subset=['y_name'])
SNAC_metrics['X'] = 'Volume & Conductivity '
# erase all snac metrics
all_metrics_df.drop(index=all_metrics_df[all_metrics_df['method']=='snac'].index, inplace=True)
# join
all_metrics_df = pd.concat([all_metrics_df, SNAC_metrics], ignore_index=True)


all_metrics_df_annexe = pd.pivot_table(all_metrics_df, values=['RMSEP', 'MAEP', 'MADP', 'RMAEP', 'R2P', 'r2P', 'biasP', 'sepP','nb_train','nb_test','nb_comp'], index=['X','other', 'method'], columns=['y_name', 'phrange'])
all_metrics_df_annexe = all_metrics_df_annexe.swaplevel(i=0,j='y_name', axis=1).sort_index(axis=1)

all_metrics_df_annexe.to_excel(output_path + 'metrics_annexe.xlsx', sheet_name='metrics_annexe',
                        startcol=1, startrow=1, freeze_panes=(0,2))


# all_metrics_df_article = pd.pivot_table(all_metrics_df, values=['RMSEP', 'MADP', 'R2P','nb_comp'], index=['X','other', 'method'], columns=['y_name', 'phrange'])
# all_metrics_df_article = all_metrics_df_article.swaplevel(i=0,j='y_name', axis=1).sort_index(axis=1)

all_metrics_article_df = deepcopy(all_metrics_df)
all_metrics_article_df.rename(columns={"phrange": "pH range", "X": "Signal","method": "Model","y_name": "Param."}, inplace = True)
all_metrics_article_df['pH range'][all_metrics_article_df['pH range'] == '38-58'] = '3.8-5.8'
all_metrics_article_df['pH range'][all_metrics_article_df['pH range'] == '825-1025'] = '8.25.-10.25'
all_metrics_article_df['Signal'][all_metrics_article_df['Signal'] == 'volume-dilutioncorrected'] = 'Volume'
all_metrics_article_df['Signal'][all_metrics_article_df['Signal'] == 'conductivity-dilutioncorrected'] = 'Conductivity'
all_metrics_article_df['Signal'][all_metrics_article_df['Signal'] == 'mb-volume-conductivity'] = 'Volume & Conductivity'
all_metrics_df_article = pd.pivot_table(all_metrics_article_df, values=['RMSEP', 'MADP', 'R2P','nb_comp'], index=['Signal', 'Model'], columns=['Param.', 'pH range'])
all_metrics_df_article = all_metrics_df_article.swaplevel(i=0,j='Param.', axis=1).sort_index(axis=1)

all_metrics_df_article.to_excel(output_path + 'metrics_article.xlsx', sheet_name='metrics_article',
                        startcol=1, startrow=1, freeze_panes=(0,2))

# with pd.ExcelWriter('metrics.xlsx', engine='xlsxwriter') as writer:  # doctest: +SKIP
#     all_metrics_df_article.to_excel(writer, sheet_name='Sheet_name_1')
#     all_metrics_df_annexe.to_excel(writer, sheet_name='Sheet_name_2')
# writer.save()

logging.info('----------------- Run end ---------------------')