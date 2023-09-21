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
from datasciencetools.metrics import *
from matplotlib import ticker
from data_filter import *
import os
from pathlib import Path
from datasciencetools.plot import *
import matplotlib.pyplot as plt

# set paths
folder_name = str('T21')
input_path ="/home/silvio/Dropbox (BioEnTech)/CORPORATE/0 ORGANISATION DROPBOX CORPORATE/12-SNAC/2-Developpement/1-Developpement algo/4 - Analyse stat/"+folder_name
output_path ="/home/silvio/Dropbox (BioEnTech)/CORPORATE/0 ORGANISATION DROPBOX CORPORATE/12-SNAC/2-Developpement/1-Developpement algo/4 - Analyse stat/"+folder_name+'/Article_graphs'

dpi_value = 80
fontsize = 12
fontname='Calibri'

VFA_light = 'orange'
VFA_dark = 'firebrick'
TAN_light ='plum'
TAN_dark ='indigo'
# color
bis_color = 'black'
VFA_color = VFA_light
VFA_pka_color = VFA_dark
TAN_color = TAN_light
TAN_pka_color = TAN_dark
TAN_train_color = TAN_dark
TAN_test_color = TAN_light
VFA_train_color = VFA_dark
VFA_test_color = VFA_light
TAN_volume_color = TAN_dark
TAN_cond_color = TAN_light
TAN_DDM_color = TAN_dark
TAN_KDM_color = TAN_light
VFA_volume_color = VFA_dark
VFA_cond_color = VFA_light
VFA_DDM_color = VFA_dark
VFA_KDM_color = VFA_light

import shutil
if os.path.exists(output_path):
    shutil.rmtree(output_path)
    os.makedirs(output_path)
else:
    os.makedirs(output_path)

def parser(s):
    print(s)
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

# print(get_super('GeeksforGeeks'))
def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


# print('H{}SO{}'.format(get_sub('2'), get_sub('4')))

model_and_SNAC_data = {}
model_coeff = {}
stacking_coeff = {}
x_model_pHfiltered_vol = {}
x_model_pHfiltered_cond = {}
x_model_all_vol = {}
x_model_all_cond = {}

list_file= {'model_and_SNAC_data':model_and_SNAC_data,'model_coeff':model_coeff,'stacking_coeff':stacking_coeff,
            'x_model_pHfiltered_vol':x_model_pHfiltered_vol,'x_model_pHfiltered_cond':x_model_pHfiltered_cond,
            'x_model_all_vol':x_model_all_vol,'x_model_all_cond':x_model_all_cond}
folder = Path(input_path).glob('*')
for i in folder:
    # print(i)
    name1 = str(i).split('/')[-1]
    for j in list_file.items():
        obj = {name1: pd.DataFrame()}
        j[1].update(obj)

    file = Path(os.path.join(i,'Data_article')).glob('*')
    for ii in file:
        # print(ii)
        name2 = str(ii).split('/')[-1].replace('.csv','')
        list_file[name2][name1] = pd.read_csv(ii, sep=';', decimal='.', index_col=0)
        # list_file[name2][name1] = pd.read_csv(ii, sep=';', decimal='.', header=None, index_col=0)


# chose what to plot
def plot_or_not(comp, phrange, model, signal, other, filter=True):
    if filter == True:
        y_n = False
        if comp == 'TAN' and model =='mbplsr' and phrange =='825-1025':
            y_n = True
        if comp == 'VFA'and model =='plsr' and phrange =='43-53' and signal =='volume-dilutioncorrected':
            y_n = True
    else:
        y_n = True
    return y_n


# check coherence dans fichier: ex size dataframe

# Je vais chercher le donnée VFA et TAN labo parmi tous les fichiers
labo_analysis_VFA = pd.DataFrame()
labo_analysis_TAN = pd.DataFrame()
for fol in model_and_SNAC_data:
    if 'Article_graphs' not in fol and 'metrics' not in fol and 'description' not in fol:
        comp, phrange, model, signal, other = parser(fol)
        if comp == 'VFA' and model == 'plsr':
            print (fol)
            labo_analysis_VFA['values'] = model_and_SNAC_data[fol]['y_VFA_labo']
            break
for fol in model_and_SNAC_data:
    if 'Article_graphs' not in fol and 'metrics' not in fol and 'description' not in fol:
        comp, phrange, model, signal, other = parser(fol)
        if comp == 'TAN' and model == 'mbplsr':
            print (fol)
            labo_analysis_TAN['values'] = model_and_SNAC_data[fol]['y_TAN_labo']
            break

if labo_analysis_VFA.isnull().values.any():
    logging.error('Nan in labo analysis VFA')
    labo_analysis_VFA.dropna()
else:
    logging.info('No Nan in labo analysis VFA. %s values' %labo_analysis_VFA.size)

labo_analysis = pd.DataFrame(index=['number', 'min', 'max', 'mean', 'median', 'std'], columns=['VFA', 'TAN'])
labo_analysis['VFA']['number'] = labo_analysis_VFA.size
labo_analysis['VFA']['min'] = labo_analysis_VFA['values'].min().round(2)
labo_analysis['VFA']['max'] = labo_analysis_VFA['values'].max().round(2)
labo_analysis['VFA']['max'] = labo_analysis_VFA['values'].max().round(2)
labo_analysis['VFA']['mean'] = labo_analysis_VFA['values'].mean().round(2)
labo_analysis['VFA']['median'] = labo_analysis_VFA['values'].median().round(2)
labo_analysis['VFA']['std'] = labo_analysis_VFA['values'].std().round(2)

if labo_analysis_TAN.isnull().values.any():
    logging.error('Nan in labo analysis TAN')
    labo_analysis_TAN.dropna()
else:
    logging.info('No Nan in labo analysis TAN. %s values' % labo_analysis_TAN.size)
labo_analysis['TAN']['number'] = labo_analysis_TAN.size
labo_analysis['TAN']['min'] = labo_analysis_TAN['values'].min().round(2)
labo_analysis['TAN']['max'] = labo_analysis_TAN['values'].max().round(2)
labo_analysis['TAN']['max'] = labo_analysis_TAN['values'].max().round(2)
labo_analysis['TAN']['mean'] = labo_analysis_TAN['values'].mean().round(2)
labo_analysis['TAN']['median'] = labo_analysis_TAN['values'].median().round(2)
labo_analysis['TAN']['std'] = labo_analysis_TAN['values'].std().round(2)

labo_analysis.to_excel(input_path + '/Labo analysis description.xlsx',startcol=1, startrow=1, freeze_panes=(0,2))

# graph plot
for fol in model_and_SNAC_data:
    if 'Article_graphs' not in fol and 'metrics' not in fol and 'description' not in fol:
        comp, phrange, model, signal, other = parser(fol)
        y_n = plot_or_not(comp, phrange, model, signal, other, filter=True)

        # define units
        if comp == 'TAN':
            unit_comp = 'gN/L'
            unit_comp = 'gN L'+get_super('-1')
        elif comp == 'VFA':
            unit_comp = 'gAc_eq/L'
            unit_comp = 'gAc_eq L'+get_super('-1')

        if comp == 'TAN':
            train_color = TAN_train_color
            test_color = TAN_test_color
            volume_color = TAN_volume_color
            cond_color = TAN_cond_color
            DDM_color = TAN_DDM_color
            KDM_color = TAN_KDM_color

        elif comp == 'VFA':
            train_color = VFA_train_color
            test_color = VFA_test_color
            volume_color = VFA_volume_color
            cond_color = VFA_cond_color
            DDM_color = VFA_DDM_color
            KDM_color = VFA_KDM_color

        plt.rcParams.update({'font.size': fontsize})
        if y_n == True:

        # graph pH and conductivity signal
            if model == 'mbplsr':
                logging.info('plotting graph "pH and conductivity signals" of :'+fol)
                data1 = x_model_all_vol[fol].T
                data2 = x_model_all_cond[fol].T
                # data1.rename(columns={'nan': 'pH'})
                # data1.columns.values[0] = 'pH'
                # data2.columns.values[0] = 'pH'
                data1['pH'] = data1.index
                data1.reset_index(drop=True, inplace=True)
                data2['pH'] = data2.index
                data2.reset_index(drop=True, inplace=True)
                data1 = data1.astype({'pH': float})
                data2 = data2.astype({'pH': float})
                # look for ids for ticks
                # list = [3, 4, 5, 6, 7, 8, 9, 10]
                # ids_data1 = []
                # ids_data2 = []
                # for ii in list:
                #     ids_data1.append(data1[data1['pH'] >= ii].index[0])
                #     ids_data2.append(data1[data2['pH'] >= ii].index[0])

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), dpi = dpi_value)
                ax1.plot(data1['pH'], data1.iloc[:,0:-1])
                ax2.plot(data2['pH'], data2.iloc[:,0:-1], ls='-')
                # craw line for pka
                ax1.axvline(4.8, 0, 1, ls='--', label='VFA_pka_th.', color = VFA_pka_color)
                ax1.axvline(9.25, 0, 1,ls='--', label='TAN_pka_th.', color= TAN_pka_color)
                ax2.axvline(4.8, 0, 1,ls='--', label='VFA_pka_th.', color= VFA_pka_color)
                ax2.axvline(9.25, 0, 1,ls= '--', label='TAN_pka_th.', color= TAN_pka_color)

                list = [3.0,4.0,4.3,4.8, 5.0,5.3, 6.0, 7.0, 8.25, 8.0, 9,9.25, 10.0, 10.25]
                ax1.xaxis.set_ticks(list)
                ax2.xaxis.set_ticks(list)
                ax1.set_xticklabels(ax1.get_xticks(), rotation=90)
                ax2.set_xticklabels(ax2.get_xticks(), rotation=90)
                # ax2.xaxis.set_ticks(ids_data2)
                # ax1.set_title('d(volume)/d(pH)')
                # ax2.set_title('d(conductivity)/d(pH)')
                ax1.set_title('\u03B2'+'\u0307')
                ax2.set_title('\u03C3'+'\u0307')

                ax2.xaxis.set_label_text('pH', fontsize = fontsize, fontname = fontname)
                ax1.yaxis.set_label_text('mL pH'+get_super('-1'), fontsize = fontsize, fontname = fontname)
                ax2.yaxis.set_label_text('mS cm'+get_super('-1')+'pH'+get_super('-1'), fontsize = fontsize, fontname = fontname)
                ax1.legend(loc='upper left')
                # ax2.legend(loc='upper left')
                # ax2.set_title('conductivity')
                # ax1.xaxis.set_ticklabels(['x=8.25', 'x=10.24'], rotation = 90, color = 'red', fontsize = 8, style = 'italic', verticalalignment = 'center')
                ax1.axvspan(4.3, 5.3, color=VFA_color, alpha=0.5)
                ax1.axvspan(8.25, 10.25, color=TAN_color, alpha=0.5)
                ax2.axvspan(4.3, 5.3, color=VFA_color, alpha=0.5)
                ax2.axvspan(8.25, 10.25, color=TAN_color, alpha=0.5)
                ax1.grid(True)
                ax2.grid(True)
                fig.savefig(output_path + '/_X_raw_all_'+fol+'.png')
                plt.close('all')
    #
    #             # fig, (ax1,ax2) = plt.subplots(2, 1,figsize=(12,12), sharex=True)
    #             # data1 = x_model_all_vol[fol].T
    #             # data2 = x_model_all_cond[fol].T
    #             # ax1.plot(data1)
    #             # ax2.plot(data2)
    #             # ax1.xaxis.set_ticks([0,100,200,300])
    #             # ax1.set_title('Volume')
    #             # ax2.set_title('Conductivity')
    #             # ax2.xaxis.set_label_text('pH')
    #             # ax1.yaxis.set_label_text('mL?')
    #             # ax2.yaxis.set_label_text('mS/cm?')
    #             # ax1.plot([9.0, 9.0], [data1.min().min(), data1.max().max()], 'b--', lw=2) # Blue dashed straight line
    #             # ax1.plot([5.8, 5.8], [data1.min().min(), data1.max().max()], 'b--', lw=2) # Blue dashed straight line
    #             # # ax2.set_title('conductivity')
    #             # # ax1.xaxis.set_ticklabels(['x=8.25', 'x=10.24'], rotation = 90, color = 'red', fontsize = 8, style = 'italic', verticalalignment = 'center')
    #             # fig.savefig(output_path + '/_X_raw_pHfiltered'+fol+'.png')
    #             # plt.close('all')


        # save train test distribution

            train =model_and_SNAC_data[fol]['y_'+comp+'_labo'][model_and_SNAC_data[fol]['y_'+comp+'_model_train'].dropna().index]
            test = model_and_SNAC_data[fol]['y_'+comp+'_labo'][model_and_SNAC_data[fol]['y_'+comp+'_model_test'].dropna().index]
            fig, ax = plt.subplots(1, 1)
            maxs = max(train.max(),test.max())
            mins = min(train.min(),test.min())
            bins = np.linspace(mins,maxs,15)
            # ax.hist(train, bins=bins, density=False, histtype='bar', align='left', alpha=.5, facecolor='blue', label='train', edgecolor = "grey")
            # ax.hist(test, bins=bins, density=False, histtype='bar', align='left', alpha=.5, facecolor='red', label='test', edgecolor = "grey")
            ax.hist(train, bins=bins, density=False, histtype='bar', align='right', rwidth=0.85, alpha=1.0,
                    facecolor= train_color, hatch='///', label='train', edgecolor="aliceblue")
            ax.hist(test, bins=bins, density=False, histtype='bar', align='right', rwidth=0.55, alpha=1.0,
                    facecolor= test_color, label='test', edgecolor="aliceblue")
            # ax.hist(train, bins=bins, density=False, histtype='stepfilled', align='left', rwidth =0.7 ,alpha=1.0, facecolor='blue', label='train', edgecolor = "grey")
            # ax.hist(test, bins=bins, density=False, histtype='step', align='left',hatch='/*/' ,alpha=1.0, facecolor='red', label='test', edgecolor = "grey")
            ax.set_xlabel(unit_comp, labelpad=-1)  # , weight='bold')
            ax.set_ylabel('Number of observations', fontsize = fontsize, fontname = fontname)  # , weight='bold')
            ax.grid(True)
            ax.legend(loc='upper right')
            plt.title(comp+' - '+model, fontsize = fontsize, fontname = fontname)
            fig.savefig(output_path + '/Histograms_'+fol+'.png')
            # desc = stats.describe(np.array(y))
            # desc_text = 'nobs=%d\n$\mu=%.2f$\n$\sigma=%.2f$\nmin=%.2f\nmax=%.2f' % (
            #     desc.nobs, desc.mean, np.sqrt(desc.variance), desc.minmax[0][0], desc.minmax[1][0])
            # ax.text(text_xy[0], text_xy[1], desc_text, horizontalalignment='right',
            #         verticalalignment='top', transform=ax.transAxes, color=facecolor,
            #         bbox=dict(boxstyle="square,pad=0.3", fc='white', alpha=0.3), size='x-small')


        # graphs comparaison
            model_train_ids = model_and_SNAC_data[fol]['y_' + comp + '_model_train'].dropna().index
            model_test_ids = model_and_SNAC_data[fol]['y_' + comp + '_model_test'].dropna().index
            stacking_train_ids = model_and_SNAC_data[fol]['y_' + comp + '_stacking_train'].dropna().index
            stacking_test_ids = model_and_SNAC_data[fol][
                'y_' + comp + '_stacking_test'].dropna().index  # todo check that the Ids n'aient pas de duplicats

            labo = model_and_SNAC_data[fol]['y_' + comp + '_labo']
            SNAC = model_and_SNAC_data[fol]['y_' + comp + '_SNAC']
            model_train = model_and_SNAC_data[fol]['y_' + comp + '_model_train']
            model_test = model_and_SNAC_data[fol]['y_' + comp + '_model_test']
            stacking_train = model_and_SNAC_data[fol]['y_' + comp + '_stacking_train']
            stacking_test = model_and_SNAC_data[fol]['y_' + comp + '_stacking_test']

            max_SNAC = max(SNAC.max(), labo.max())
            max_model = max(model_test.max(), model_train.max(), labo.max())
            max_stacking = max(stacking_test.max(), stacking_train.max(), labo.max())

            if (model_train_ids == stacking_train_ids).all() and (model_test_ids == stacking_test_ids).all(): # check if le train and test are coherant pour tous
                logging.info('train and test ids of model and stacking are the same')

                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 18), dpi = dpi_value, sharex=True)

                # fig.set_figwidth(18)
                # fig = plt.figure(constrained_layout=True)
                # fig.set_figheight(6)
                # fig.set_figwidth(18)
                # figg = fig.add_gridspec(ncols=3, nrows=1, width_ratios=[1,1,1], height_ratios=[1])
                # figg = fig.add_gridspec(ncols=3, nrows=1)
                # ax1 = fig.add_subplot(figg[0, 0])
                # ax2 = fig.add_subplot(figg[0, 1])
                # ax3 = fig.add_subplot(figg[0, 2])
                # plot_lineplot_cv(fig=fig1, ax=ax1)
                ax1.plot([0, max_SNAC], [0, max_SNAC], '--', color= bis_color, label='bisector', lw=1)
                ax2.plot([0, max_model], [0, max_model], '--', color=bis_color, label='bisector', lw=1)
                ax3.plot([0, max_stacking], [0, max_stacking], '--', color=bis_color, label='bisector', lw=1)
                # ax1.scatter(labo, SNAC, marker='+', color='blue')
                ax1.scatter(labo[model_train_ids], SNAC[model_train_ids], marker='+', color=train_color, label='model train')
                ax1.scatter(labo[model_test_ids], SNAC[model_test_ids], marker='+', color=test_color, label='model test')
                ax2.scatter(labo[model_train_ids], model_train[model_train_ids], marker='+', color=train_color, label='train')
                ax2.scatter(labo[model_test_ids], model_test[model_test_ids], marker='+', color=test_color, label='test')
                ax3.scatter(labo[stacking_train_ids], stacking_train[stacking_train_ids], marker='+', color=train_color,
                            label='train')
                ax3.scatter(labo[stacking_test_ids], stacking_test[stacking_test_ids], marker='+', color=test_color,
                            label='test')
                ax1.legend(loc='upper left')
                ax2.legend(loc='upper left')
                ax3.legend(loc='upper left')
                # ax3.xaxis.set_ticks([0, 100, 200, 300])
                # fig.set_title('SNAC - Model - Stacking')
                ax1.set_title('SNAC', fontsize = fontsize, fontname = fontname)
                ax2.set_title(model, fontsize = fontsize, fontname = fontname)
                ax3.set_title('Stacking ', fontsize = fontsize, fontname = fontname)
                # ax1.xaxis.set_label_text('Observed (labo) - '+unit_comp, fontsize = fontsize, fontname = fontname)
                # ax2.xaxis.set_label_text('Observed (labo) - '+unit_comp, fontsize = fontsize, fontname = fontname)
                ax3.xaxis.set_label_text('Observed (labo) - '+unit_comp, fontsize = fontsize, fontname = fontname)
                ax1.yaxis.set_label_text('Predicted - '+unit_comp, fontsize = fontsize, fontname = fontname)
                ax2.yaxis.set_label_text('Predicted - ' + unit_comp, fontsize=fontsize, fontname=fontname)
                ax3.yaxis.set_label_text('Predicted - ' + unit_comp, fontsize=fontsize, fontname=fontname)
                ax1.grid(True)
                ax2.grid(True)
                ax3.grid(True)
                add_metrics(ax1, labo[model_test_ids], SNAC[model_test_ids])
                add_metrics(ax2, labo[model_test_ids], model_test[model_test_ids])
                add_metrics(ax3, labo[model_test_ids], stacking_test[stacking_test_ids])
                # ax2.yaxis.set_label_text('Predicted')
                # ax3.yaxis.set_label_text('Predicted')
                fig.suptitle(comp+' predictions', fontsize = fontsize, fontname = fontname)
                fig.savefig(output_path + '/Prediction_'+fol+'.png')
                plt.close('all')


            # SNAC vs labo
                # simple
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi = dpi_value)
                ax1.scatter(labo, SNAC, marker='+', label='SNAC')
                ax1.plot([0, max_SNAC], [0, max_SNAC], '--', color=bis_color, label='bisector', lw=1)
                ax1.set_title(comp+' - SNAC', fontsize = fontsize, fontname = fontname)
                ax1.set_xlabel('Observed (labo) - '+unit_comp, fontsize = fontsize, fontname = fontname)
                ax1.set_ylabel('Predicted - '+unit_comp, fontsize = fontsize, fontname = fontname)
                ax1.grid(True)
                ax1.legend(loc='upper left')
                add_metrics(ax1, labo, SNAC)
                fig.savefig(output_path + '/SNAC predictions_' + fol + '.png')
                plt.close('all')


                # labo partagé par labo type
                n_labo = model_and_SNAC_data[fol].nunique()[comp+'_labo']
                laboratories = model_and_SNAC_data[fol].drop_duplicates([comp + '_labo'])[comp + '_labo']
                labo_ids = {}
                for i in laboratories:
                    obj = {i: model_and_SNAC_data[fol][model_and_SNAC_data[fol][comp + '_labo'] == i].index.to_list()}
                    labo_ids.update(obj)

                # SNAC = model_and_SNAC_data[fol]['y_' + comp + '_SNAC']
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
                ax1.plot([0, max_SNAC], [0, max_SNAC], '--', color=bis_color, label='bisector', lw=1)
                for i in laboratories:
                    # labo = model_and_SNAC_data[fol]['y_' + comp + '_labo']
                    SNAC_x = SNAC[labo_ids[i]]
                    labo_x = labo[labo_ids[i]]
                    ax1.scatter(labo_x, SNAC_x, marker='+', label=i)

                SNAC = model_and_SNAC_data[fol]['y_' + comp + '_SNAC']
                fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
                ax1.plot([0, max_SNAC], [0, max_SNAC], '--', color=bis_color, label='bisector', lw=1)
                for i in laboratories:
                    # labo = model_and_SNAC_data[fol]['y_' + comp + '_labo']
                    SNAC_x = SNAC[labo_ids[i]]
                    labo_x = labo[labo_ids[i]]
                    ax1.scatter(labo_x, SNAC_x, marker='+', label=i)


                # # labo chose on methods and labo
                # n_labo = model_and_SNAC_data[fol].nunique()[comp + '_labo']
                # n_method = model_and_SNAC_data[fol].nunique()[comp + '_method']
                # laboratories = model_and_SNAC_data[fol].drop_duplicates([comp + '_labo'])[comp + '_labo']
                # methods = model_and_SNAC_data[fol].drop_duplicates([comp + '_method'])[comp + '_method']
                # labo_ids = {}
                # for j in methods:
                #     lab = model_and_SNAC_data[fol][model_and_SNAC_data[fol][comp + '_method'] == j].drop_duplicates(
                #         [comp + '_labo'])[comp + '_labo']
                #     for i in lab:
                #         obj = {j + '__' + i: model_and_SNAC_data[fol][
                #             (model_and_SNAC_data[fol][comp + '_method'] == j) & (
                #                         model_and_SNAC_data[fol][comp + '_labo'] == i)].index.to_list()}
                #         labo_ids.update(obj)
                #
                # fig, ax1 = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi_value)
                # ax1.plot([0, max_SNAC], [0, max_SNAC], '--', color='red', label='bisector', lw=1)
                #
                # markers = {'+': None, 'v': None, 's': None, 'd': None, '*': None, 'o': None, 'p': None, 'd': None}
                # colors = {'purple': , 'royalblue': None, 'deeppink': None, 'maroon': None, 'gold': None, forestgreen': None, 'k': None,
                #           'slateblue': None}
                # for i in labo_ids:
                #     meth_lab = i.split('__')
                #
                #     marker_ok = 0
                #     for m in markers:
                #         if markers[m] == meth_lab[0]:
                #             marker_ok = 1
                #     if marker_ok == 0:
                #         for m in markers:
                #             if markers[m] == None:
                #                 markers[m] = meth_lab[0]
                #                 break
                #
                #     color_ok = 0
                #     for m in colors:
                #         if colors[m] == meth_lab[1]:
                #             color_ok = 1
                #     if color_ok == 0:
                #         for m in colors:
                #             if colors[m] == None:
                #                 colors[m] = meth_lab[1]
                #                 break
                #
                # for i in labo_ids:
                #     meth_lab = i.split('__')
                #     for j in markers:
                #         if markers[j] == meth_lab[0]:
                #             mm = j
                #     for jj in colors:
                #         if colors[jj] == meth_lab[1]:
                #             cc = jj
                #     SNAC_x = SNAC[labo_ids[i]]
                #     labo_x = labo[labo_ids[i]]
                #     ax1.scatter(labo_x, SNAC_x, marker=mm, color=cc, label=i)

                ax1.set_title(comp+' - SNAC', fontsize = fontsize, fontname = fontname)
                ax1.set_xlabel('Observed (labo) - '+unit_comp, fontsize = fontsize, fontname = fontname)
                ax1.set_ylabel('Predicted - '+unit_comp, fontsize = fontsize, fontname = fontname)
                ax1.grid(True)
                ax1.legend(loc='upper left')
                add_metrics(ax1, labo, SNAC , position = 0.55)
                fig.savefig(output_path + '/SNAC predictions_laboratories_' + fol + '.png')
                plt.close('all')


            # bcoeff model and stacking
            if model == 'plsr':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi = dpi_value)
                pH = model_coeff[fol]['pH']
                bcoeff = model_coeff[fol]['bcoeff_model_signal']
                ax1.plot(pH, bcoeff, color=volume_color, label='Volume', lw=1)
                ax1.set_title(comp+' - '+model, fontsize = fontsize, fontname = fontname)
                ax1.set_xlabel('pH', fontsize = fontsize, fontname = fontname)
                ax1.set_ylabel('Beta-coefficient', fontsize = fontsize, fontname = fontname)
                ax1.legend(loc='upper left')

            elif model == 'mbplsr':
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi = dpi_value)
                pH = model_coeff[fol]['pH']
                bcoeff_vol = model_coeff[fol]['bcoeff_model_vol']
                bcoeff_cond = model_coeff[fol]['bcoeff_model_cond']
                # ax1.scatter(labo, SNAC, marker='+', label='SNAC')
                ax1.plot(pH, bcoeff_vol, ls ='-',color=volume_color, label='Volume', lw=1)
                ax1.plot(pH, bcoeff_cond, ls ='-.',color=cond_color, label='Conductivity', lw=1)
                ax1.set_title(comp+' - '+model, fontsize = fontsize, fontname = fontname)
                ax1.set_xlabel('pH', fontsize = fontsize, fontname = fontname)
                ax1.set_ylabel('Beta-coefficient', fontsize = fontsize, fontname = fontname)
                ax1.legend(loc='upper left')

            barWidth = 0.2
            y1 = stacking_coeff[fol]['model']
            y2 = stacking_coeff[fol]['SNAC']
            r1 = range(len(y1))
            r2 = [x + barWidth for x in r1]
            ax2.bar(r1, y1, width=barWidth, label=model, color=[DDM_color for i in y1], hatch='/',
                    edgecolor=['aliceblue' for i in y1], linewidth=1)
            ax2.bar(r2, y2, width=barWidth, label='SNAC', color=[KDM_color for i in y1], hatch='//',
                    edgecolor=['aliceblue' for i in y1], linewidth=1)
            ax2.set_xlim([barWidth*(-2),barWidth*3])
            ax2.xaxis.set_ticks([])
            ax2.set_title(comp+' - '+model, fontsize = fontsize, fontname = fontname)
            ax2.set_ylabel('Weight', fontsize = fontsize, fontname = fontname)
            ax2.legend(loc='upper left')
            fig.savefig(output_path + '/Bcoeff & weights_' + fol + '.png')


plt.close('all')


