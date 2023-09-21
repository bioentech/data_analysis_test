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

def plot_PCA(X, y, reductor, config, sub_output_path):
    logging.info('plotting explained variance')
    fig, ax = reductor.plot_explained_variance()
    fig.savefig(sub_output_path + 'pca_explained_variance.png')

    logging.info('plotting all scores (colored by y)')
    fig, axes = reductor.plot_scores_2d_all(color_values=y[config['y_variable']].values, cmap='viridis', color_label=config['y_variable'], labels=None, plot_ellipse=True) #without labels
    fig.savefig(sub_output_path + 'pca_scores_all_coloredy.png')

    logging.info('plotting all scores (colored by site)')
    fig, axes = reductor.plot_scores_2d_all(color_values=y.index.map(lambda s:s.split('_')[0]).values, cmap='tab10', color_label='Site', labels=None, plot_ellipse=True) #without labels
    fig.savefig(sub_output_path + 'pca_scores_all_coloredsite.png')

    logging.info('plotting all loadings')
    fig, axes = reductor.plot_loadings_all(figsize=(10, 6), var_names=X.columns, kind= config['plot']['loadings_kind'])
    fig.savefig(sub_output_path + 'pca_loadings_all.png')

    logging.info('compute score and orthogonal distance')
    fig, ax = reductor.plot_diagnostic(X=X, add_confidence_intervals=True, q=0.99)
    fig.savefig(sub_output_path + 'pca_sdod.png')
    return

def plot_dataset_graphs(merged_Res_analyselabo,compounds,config, custom_criteria, sub_output_path):
    if config['plot']['plot_dataset_graphs']:
        logging.info('plotting relationship relative/absolute errors and absolute value for VFA and TAN')
        # relationship relative error and VFA/TAN analyse labo
        for error_type in ['relative', 'absolute']:
            cur_x_label = '%s_geq_y' % compounds
            cur_y_label = '%s_%s_error' % (compounds, error_type)
            x = merged_Res_analyselabo.loc[:, [cur_x_label]]
            y = merged_Res_analyselabo.loc[:, [cur_y_label]]
            # y = y.loc[y['%s_relative_error'%compounds]<2.,:].copy()
            xy_merged = x.merge(y, left_index=True, right_index=True, how='inner').dropna()
            x = xy_merged.loc[:, x.columns]
            y = xy_merged.loc[:, y.columns]

            if config['plot']['label_outliers']:
                fig, ax = relate_xy(x=x.values, y=y.values,
                                    add_reg_line=config['plot']['relatexy_params']['add_reg_line'],
                                    add_qreg_line=config['plot']['relatexy_params']['add_qreg_line'], labels=x.index,
                                    reg_params=config['plot']['relatexy_params']['reg_params'],
                                    qreg_params=config['plot']['relatexy_params']['qreg_params']
                                    )
            else:
                fig, ax = relate_xy(x=x.values, y=y.values,
                                    add_reg_line=config['plot']['relatexy_params']['add_reg_line'],
                                    add_qreg_line=config['plot']['relatexy_params']['add_qreg_line'],  # labels=x.index,
                                    reg_params=config['plot']['relatexy_params']['reg_params'],
                                    qreg_params=config['plot']['relatexy_params']['qreg_params']
                                    )
            ax.set_xlabel(cur_x_label)
            ax.set_ylabel(cur_y_label)
            fig.savefig(sub_output_path + '%s_error_%s_absolute_value.png' % (error_type, compounds))


    if config['plot']['plot_dataset_graphs']:
        logging.info('Get links between errors (pairplot)')
        fig, axes = pairplot_diag(
            merged_Res_analyselabo.loc[:, [compounds + '_absolute_error', compounds + '_relative_error']].dropna())
        fig.savefig(sub_output_path + 'pairplot_errors.png')

        if custom_criteria.shape[1] <= 10:
            logging.info('Get links between custom_criteria and TAN/VFA absolute/relative errors (pairplot)')
            fig, axes = pairplot_diag(X=custom_criteria.dropna(),
                                      y=merged_Res_analyselabo.loc[:, [compounds + '_absolute_error']].dropna(),
                                      pearsonr_threshold=config['plot']['pearsonr_threshold'])
            fig.savefig(sub_output_path + 'pairplot_' + compounds + '_absolute_error.png')
            fig, axes = pairplot_diag(X=custom_criteria.dropna(),
                                      y=merged_Res_analyselabo.loc[:, [compounds + '_relative_error']].dropna(),
                                      pearsonr_threshold=config['plot']['pearsonr_threshold'])
            fig.savefig(sub_output_path + 'pairplot_' + compounds + '_relative_error.png')
        else:
            logging.info('Get links between custom_criteria (heatmap)')
            fig, axes = heatmap_diag(X=custom_criteria.dropna())
            fig.savefig(sub_output_path + 'heatmap_X.png')
            logging.info('Get links between custom_criteria and TAN/VFA absolute/relative errors (heatmap)')
            fig, axes = heatmap_diag(X=custom_criteria.dropna(),
                                     y=merged_Res_analyselabo.loc[:, [compounds + '_absolute_error']].dropna())
            fig.savefig(sub_output_path + 'heatmap_' + compounds + '_absolute_error.png')
            fig, axes = heatmap_diag(X=custom_criteria.dropna(),
                                     y=merged_Res_analyselabo.loc[:, [compounds + '_relative_error']].dropna())
            fig.savefig(sub_output_path + 'heatmap_' + compounds + '_relative_error.png')
    return


def plot_snac_labo(merged_Res_analyselabo,compounds,config, custom_criteria, sub_output_path):
    # plot snac and analyse labo together
    logging.info('plot SNAC vs lab values for '+compounds)
    y_observed_name = '%s_geq_y'%compounds # lab
    y_predicted_name = '%s_geq_x'%compounds # snac
    data = merged_Res_analyselabo.dropna(subset=[y_observed_name, y_predicted_name])
    fig, ax = plot_pred_obs(data[y_predicted_name], data[y_observed_name], fig=None, ax=None, figsize=(10,10), marker= '+', color='steelblue', alpha=.8, color_values=None,
                      cmap='viridis', color_label='', labels=data.index, plot_legend=True,
                      add_identity_line=True,
                            # identity_range=[np.concatenate((data[y_observed_name], data[y_predicted_name])).min()-0.5,
                            #                 np.concatenate((data[y_observed_name], data[y_predicted_name])).max()+0.5],
                            # identity_range=[-0.25,10.0],
                            # label_only_outliers=False,
                            q_outliers=95, label_size=8,
                            add_reg_line=config['plot']['predobs_params']['add_reg_line'], reg_params=config['plot']['predobs_params']['reg_params'],
                            add_qreg_line=config['plot']['predobs_params']['add_qreg_line'],  qreg_params=config['plot']['predobs_params']['qreg_params'])

    # add metrics:
    metrics = dict()
    metrics['rmse'] = np.sqrt(mean_squared_error(data[y_observed_name], data[y_predicted_name]))
    metrics['mae'] = mean_absolute_error(data[y_observed_name], data[y_predicted_name])
    metrics['mad'] = median_absolute_error(data[y_observed_name], data[y_predicted_name])
    metrics['rmae'] = mean_absolute_percentage_error(data[y_observed_name], data[y_predicted_name])
    metrics['R2'] = r2_score(data[y_observed_name], data[y_predicted_name])
    metrics['r2'] = pearson_correlation_squared(data[y_observed_name], data[y_predicted_name])
    metrics['bias'] = bias(data[y_observed_name], data[y_predicted_name])
    metrics['sep'] = standard_error_prediction(data[y_observed_name], data[y_predicted_name])
    ax.text(0.05, 0.95, 'RMSE=%.2f\nMAE=%.2f\nMAD=%.2f\nRMAE=%.2f\n$R^{2}$=%.2f\n$r^{2}$=%.2f\n$bias$=%.2f\n$sep$=%.2f' % (metrics['rmse'],metrics['mae'],metrics['mad'],metrics['rmae'],metrics['R2'],metrics['r2'],metrics['bias'],metrics['sep']), ha='left', va='top', size=8, transform=ax.transAxes)
    ax.text(0.05, 0.98, 'nb. ech. = ' + str(data[y_observed_name].shape), ha='left', va='top', size=8, transform=ax.transAxes)

    ax.set_title('SNAC vs. Analyse Labo: %s'%compounds)
    ax.set_xlabel('SNAC')
    ax.set_ylabel('Analyse labo')
    fig.savefig(sub_output_path + 'SNAC_analyselabo_%s.png'%compounds)

    return


def plot_pls_epo(config, X, sub_output_path,regression_type,gscv, fig1, ax1, fig2, ax2, train_ids, test_ids):
    fig1.savefig(sub_output_path + '%s_cv.png' % regression_type)
    fig2.savefig(sub_output_path + '%s_predobs.png' % regression_type)

    if regression_type in ['plsr', 'epoplsr']:

        if regression_type == 'plsr':
            plsr_object = gscv.best_estimator_
        elif regression_type == 'epoplsr':
            plsr_object = gscv.best_estimator_['plsr']

        if plsr_object.n_components > 1:
            logging.info('plotting explained variance')
            fig, ax = plsr_object.plot_explained_variance()
            fig.savefig(sub_output_path + '%s_explained_variance.png' % regression_type)

            logging.info('plotting all scores')
            fig, axes = plsr_object.plot_scores_2d_all(X)
            fig.savefig(sub_output_path + '%s_scores.png' % regression_type)

            logging.info('plotting scores per site')
            for i in range(1, plsr_object.n_components):
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                fig, ax = plsr_object.plot_scores_2d(pc1=0, pc2=i, fig=fig, ax=ax,
                                                     color_values=X.index.map(lambda s: s.split('_')[0]), cmap='tab20')
                fig.tight_layout()
                fig.savefig(sub_output_path + '%s_scores_sites_f%d.png' % (regression_type, i))

        logging.info('plotting all loadings')
        fig, axes = plsr_object.plot_loadings_all(var_names=X.columns, kind=config['plot']['loadings_kind'])
        fig.savefig(sub_output_path + '%s_loadings.png' % regression_type)

        logging.info('plotting bcoefficients')
        fig, ax = plsr_object.plot_bcoefs(var_names=X.columns, kind=config['plot']['loadings_kind'])
        fig.savefig(sub_output_path + '%s_bcoefs.png' % regression_type)

        logging.info('plotting orthogonal and score distances')
        fig, axes = plsr_object.plot_diagnostic(X, labels=X.index)
        fig.savefig(sub_output_path + '%s_sdod.png' % regression_type)
        plt.close('all')
    else:
        logging.info('plotting for MB method')
        mb_estimator = gscv.best_estimator_

        logging.info('plotting all loadings')
        # fig, ax = mb_estimator.plot_loadings(pc=0, kind=config['plot']['loadings_kind'])
        all_fig_axes = mb_estimator.plot_loadings_all_mb(n_comp=config['plot']['mb_params']['loadings_max_n_comp'],
                                                         kind=config['plot']['loadings_kind'],
                                                         var_names=[b.columns for b in X])
        for i, (f, a) in enumerate(all_fig_axes):
            f.savefig(sub_output_path + '%s_loadings_comp%d.png' % (regression_type, i))

        logging.info('plotting bcoefficients')
        # fig, ax = mb_estimator.plot_bcoefs(kind=config['plot']['loadings_kind'])
        # fig, axes = mb_estimator.plot_bcoefs_mb(self=est, kind=config['plot']['loadings_kind'], var_names= [b.columns for b in X], sharex=False)
        fig, ax = mb_estimator.plot_bcoefs_mb(kind=config['plot']['loadings_kind'], var_names=[b.columns for b in X],
                                              sharex=True)
        fig.savefig(sub_output_path + '%s_bcoefs.png' % regression_type)

        logging.info('plotting block contributions to each components')
        fig, ax = mb_estimator.plot_block_contributions(figsize=(6, 3))
        fig.savefig(sub_output_path + '%s_blockcontributions.png' % regression_type)
    return

def plot_stacking_MB(y_train, y_test, y_train_pred, y_test_pred, stacking, sub_output_path):
    # plot results:
    fig_stacking, ax_stacking = plt.subplots(1, 1, figsize=(6, 6))
    xy_range = [np.concatenate(
        (y_train.values.reshape(-1, 1), y_train_pred.reshape(-1, 1), y_test.values.reshape(-1, 1),
         y_test_pred.reshape(-1, 1))).min(),
                np.concatenate(
                    (y_train.values.reshape(-1, 1), y_train_pred.reshape(-1, 1), y_test.values.reshape(-1, 1),
                     y_test_pred.reshape(-1, 1))).max()]
    fig_stacking, ax_stacking = plot_pred_obs(y_train.values.reshape(-1, 1), y_train_pred, fig=fig_stacking,
                                              ax=ax_stacking,
                                              color='steelblue', identity_range=xy_range)
    fig_stacking, ax_stacking = plot_pred_obs(y_test.values.reshape(-1, 1), y_test_pred, fig=fig_stacking,
                                              ax=ax_stacking, color='red',
                                              add_identity_line=False)
    ax_stacking.legend(['Identity line', 'Train', 'Test'], loc='lower left', bbox_to_anchor=(-0.1, 1.), ncol=3,
                       fontsize=8, frameon=True)

    # add metrics:
    metrics = compute_metrics(y_train, y_test, y_train_pred, y_test_pred)
    logging.info(
        'Metrics with stacked model: RMSEc=%.2f, RMSEp=%.2f, MAEc=%.2f, MAEp=%.2f, MADc=%.2f, MADp=%.2f, RMAEc=%.2f, RMAEp=%.2f, $R^{2}$c=%.2f, $R^{2}$p=%.2f, $r^{2}$c=%.2f, $r^{2}$p=%.2f,biasc=%.2f, biasp=%.2f, sepc=%.2f, sepp=%.2f' % (
            metrics['rmsec'], metrics['rmsep'],
            metrics['maec'], metrics['maep'], metrics['madc'], metrics['madp'], metrics['rmaec'],
            metrics['rmaep'],
            metrics['R2c'], metrics['R2p'], metrics['r2c'], metrics['r2p'], metrics['biasc'], metrics['biasp'],
            metrics['sepc'], metrics['sepp']))

    ax_stacking.text(0.05, 0.95, 'RMSEc=%.2f, RMSEp=%.2f\n'
                                 'MAEc=%.2f, MAEp=%.2f\n'
                                 'MADc=%.2f, MADp=%.2f\n'
                                 'RMAEc=%.2f, RMAEp=%.2f\n'
                                 '$R^{2}$c=%.2f, $R^{2}$p=%.2f\n'
                                 '$r^{2}$c=%.2f, $r^{2}$p=%.2f\n'
                                 'biasc=%.2f, biasp=%.2f\n'
                                 'sepc=%.2f, sepp=%.2f'
                     % (
                         metrics['rmsec'], metrics['rmsep'],
                         metrics['maec'], metrics['maep'],
                         metrics['madc'], metrics['madp'],
                         metrics['rmaec'], metrics['rmaep'],
                         metrics['R2c'], metrics['R2p'],
                         metrics['r2c'], metrics['r2p'],
                         metrics['biasc'], metrics['biasp'],
                         metrics['sepc'], metrics['sepp'],
                     ), ha='left', va='top', size=8, transform=ax_stacking.transAxes)
    ax_stacking.text(0.05, 0.98, 'nb. train: %.0f, nb. test: %.0f' % (y_train.shape[0], y_test.shape[0]),
                     ha='left',
                     va='top', size=8,
                     transform=ax_stacking.transAxes)
    fig_stacking.savefig(sub_output_path + 'stackingmbplsr_predobs.png')

    fig_stacking2, ax_stacking2 = plt.subplots(1, 1)
    pd.DataFrame(stacking.final_estimator_.coef_, index=['MBPLSR', 'SNAC'],
                 columns=['Model weights in stacked regressor']).T.plot(kind='bar', ax=ax_stacking2)
    ax_stacking2.set_xticklabels(ax_stacking2.get_xticklabels(), rotation=0)
    fig_stacking2.savefig(sub_output_path + 'stackingmbplsr_weights.png')
    return


def plot_stacking_PLS(y_train, y_test, y_train_pred, y_test_pred, stacking, sub_output_path):
    # plot results:
    fig_stacking, ax_stacking = plt.subplots(1, 1, figsize=(6, 6))
    xy_range = [np.concatenate(
        (y_train.values.reshape(-1, 1), y_train_pred.reshape(-1, 1), y_test.values.reshape(-1, 1),
         y_test_pred.reshape(-1, 1))).min(),
                np.concatenate(
                    (y_train.values.reshape(-1, 1), y_train_pred.reshape(-1, 1), y_test.values.reshape(-1, 1),
                     y_test_pred.reshape(-1, 1))).max()]
    fig_stacking, ax_stacking = plot_pred_obs(y_train.values.reshape(-1, 1), y_train_pred, fig=fig_stacking,
                                              ax=ax_stacking,
                                              color='steelblue', identity_range=xy_range)
    fig_stacking, ax_stacking = plot_pred_obs(y_test.values.reshape(-1, 1), y_test_pred, fig=fig_stacking,
                                              ax=ax_stacking, color='red',
                                              add_identity_line=False)
    ax_stacking.legend(['Identity line', 'Train', 'Test'], loc='lower left', bbox_to_anchor=(-0.1, 1.), ncol=3,
                       fontsize=8, frameon=True)

    # add metrics:
    metrics = compute_metrics(y_train, y_test, y_train_pred, y_test_pred)
    logging.info(
        'Metrics with stacked model: RMSEc=%.2f, RMSEp=%.2f, MAEc=%.2f, MAEp=%.2f, MADc=%.2f, MADp=%.2f, RMAEc=%.2f, RMAEp=%.2f, $R^{2}$c=%.2f, $R^{2}$p=%.2f, $r^{2}$c=%.2f, $r^{2}$p=%.2f,biasc=%.2f, biasp=%.2f, sepc=%.2f, sepp=%.2f' % (
            metrics['rmsec'], metrics['rmsep'],
            metrics['maec'], metrics['maep'], metrics['madc'], metrics['madp'], metrics['rmaec'],
            metrics['rmaep'],
            metrics['R2c'], metrics['R2p'], metrics['r2c'], metrics['r2p'], metrics['biasc'], metrics['biasp'],
            metrics['sepc'], metrics['sepp']))

    ax_stacking.text(0.05, 0.95, 'RMSEc=%.2f, RMSEp=%.2f\n'
                                 'MAEc=%.2f, MAEp=%.2f\n'
                                 'MADc=%.2f, MADp=%.2f\n'
                                 'RMAEc=%.2f, RMAEp=%.2f\n'
                                 '$R^{2}$c=%.2f, $R^{2}$p=%.2f\n'
                                 '$r^{2}$c=%.2f, $r^{2}$p=%.2f\n'
                                 'biasc=%.2f, biasp=%.2f\n'
                                 'sepc=%.2f, sepp=%.2f'
                     % (
                         metrics['rmsec'], metrics['rmsep'],
                         metrics['maec'], metrics['maep'],
                         metrics['madc'], metrics['madp'],
                         metrics['rmaec'], metrics['rmaep'],
                         metrics['R2c'], metrics['R2p'],
                         metrics['r2c'], metrics['r2p'],
                         metrics['biasc'], metrics['biasp'],
                         metrics['sepc'], metrics['sepp'],
                     ), ha='left', va='top', size=8, transform=ax_stacking.transAxes)
    ax_stacking.text(0.05, 0.98, 'nb. train: %.0f, nb. test: %.0f' % (y_train.shape[0], y_test.shape[0]), ha='left',
                     va='top', size=8,
                     transform=ax_stacking.transAxes)
    fig_stacking.savefig(sub_output_path + 'stacking_predobs.png')

    fig_stacking2, ax_stacking2 = plt.subplots(1, 1)
    pd.DataFrame(stacking.final_estimator_.coef_, index=['PLSR', 'SNAC'],
                 columns=['Model weights in stacked regressor']).T.plot(kind='bar', ax=ax_stacking2)
    ax_stacking2.set_xticklabels(ax_stacking2.get_xticklabels(), rotation=0)
    fig_stacking2.savefig(sub_output_path + 'stacking_weights.png')
    return

