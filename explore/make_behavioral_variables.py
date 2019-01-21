# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 15:02:42 2016

@author: pipolose
"""
from __future__ import division
import helper_functions as hf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.covariance import MinCovDet
from os.path import join as opj
import os
from scipy.stats import mannwhitneyu
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
import matplotlib as mpl
from colorsys import rgb_to_hls

mpl.rcParams['pdf.fonttype'] = 42

'''
USER-SET PARAMETERS
'''
do_percent_change = False  # Set to true to use percent change instead of abs units
use_only_ton_visits = False
n_components = 10
do_save = False
out_dir = '/data2/polo/figures/'
save_plots = False


'''
FUNCTION DEFINITIONS
'''

div_pal = sns.color_palette("RdBu", 7)
diverging_cmap = LinearSegmentedColormap.from_list('diverging_sns',
                                                   div_pal[::-1])

def compute_slopes_df(subject_list, subjects_dict, do_percent_change=False,
                      use_only_ton_visits=False):
    '''
    Computes a DataFrame with the slopes of each subject in each task
    '''
    slopes_dlist = []
    visits_dlist = []
    right_handed_dict = {'left': False, 'right': True, 'mixed': False}
    for su_name in subject_list:
        su = subjects_dict[su_name]
        last_visit = sorted(su.visdy.keys(), key=lambda x: x[::-1])[-1]
        su_slope_dict = {'subjid': su.subjid,
                         'group': su.group,
                         'sex': su.demographics['sex']-1,
                         'age': su.visdy[last_visit]/365. +
                         su.demographics['age'],
                         'eligible': su.eligible,
                         'right_handed': right_handed_dict[
                             su.demographics['handness']],
                         'CAP': su.CAP[last_visit],
                         'CAG': np.nanmean(su.pheno_df.caghigh)}
        su_visits_dict = {'subjid': su.subjid}
        cmp = su.cog_motor_performance
        if use_only_ton_visits:
            cmp = cmp.loc[cmp['visit'].str.endswith('ton')]

        y_cols = [c for c in cmp.columns if c not in [u'visdy', u'visit']]
        x = np.stack((cmp.visdy/365., np.ones(cmp.visdy.shape)), axis=1)
        motor_tasks = ['grip_var', 'paced_tap']
        for col_name in y_cols:
            try:
                y = cmp[col_name].values
                if col_name in motor_tasks:
                    perf_sign = -1
                else:
                    perf_sign = 1
                ok_idx = ~np.isnan(y)
                visits_used = int(ok_idx.sum())
                assert np.min(y[ok_idx]) > 0
                m = np.linalg.lstsq(x[ok_idx], perf_sign * y[ok_idx])[0][0]
                if do_percent_change:
                    m = m / np.abs(np.mean(y[ok_idx]))
            except:
                m = np.nan
                visits_used = int(np.sum(~np.isnan(cmp[col_name].values)))
            su_slope_dict.update({col_name: m})
            su_visits_dict.update({col_name: visits_used})
        slopes_dlist.append(su_slope_dict)
        visits_dlist.append(su_visits_dict)
    slopes_df = pd.DataFrame(slopes_dlist).set_index('subjid')
    visits_used_df = pd.DataFrame(visits_dlist).set_index('subjid')
    return slopes_df, visits_used_df


def plot_corr_matrix(corr_mat, col_names=None, title=None):
    '''
    Take a correlation matrix and makes a nice plot of it
    '''
    sns.set_style('dark')
    ih = plt.imshow(corr_mat, interpolation='nearest', cmap=diverging_cmap,
                    vmin=-1, vmax=1)
    if col_names:
        plt.yticks(np.arange(corr_mat.shape[0]), col_names)
        plt.xticks(np.arange(corr_mat.shape[0]), col_names,
                   rotation='vertical')
    if title:
        plt.title(title)
    cb = plt.colorbar()
    cb.set_ticks(np.linspace(-1, 1, 5))
    plt.show()
    sns.set_style('darkgrid')
    return ih


def corr_from_cov(cov_mat):
    '''
    Computes a correlation matrix from a covariance matrix
    '''
    D_inv = np.diag(1/np.sqrt(cov_mat.diagonal()))
    corr_mat = D_inv.dot(cov_mat.dot(D_inv))
    return corr_mat


def PC_df_heatmap(pc_weights, fract_ordered_eigval, n_max=None, annot=False,
                  cmap=diverging_cmap, **kwargs):
    '''
    Takes a DataFrame with PC loadings of each task
    '''
    if not n_max or (n_max > pc_weights.shape[1]):
        n_max = pc_weights.shape[1]
    ah = sns.heatmap(pc_weights.ix[:, :n_max], center=0, annot=annot,
                     cmap=cmap,
                     **kwargs)
    for l in ah.get_yticklabels():
        l.set_rotation(0)
    plt.xlabel('Explained variance (%)')
    ah.set_xticklabels(["{:2.1f}".format(f) for f in fract_ordered_eigval])
    plt.show()
    return ah


def PCs_from_cov(cov_mat, task_names, n_components=None, convert_2_corr=False,):
    '''
    Computes Principal Components loadings (in DataFrame format if task names
    are provided, else as array)
    from a covariance matrix
    '''
    if convert_2_corr:
        cov_mat = corr_from_cov(cov_mat)
    eig_val, eig_vects = np.linalg.eig(cov_mat)
    idx_order = np.argsort(eig_val)[::-1]
    if not n_components:
        n_components = eig_val.shape[0]
    ordered_eigvect = eig_vects[:, idx_order][:, :n_components]
    s_idx = np.abs(ordered_eigvect).argmax(axis=0)
    ordered_eigvect = ordered_eigvect * np.tile(np.sign(ordered_eigvect[s_idx, np.arange(ordered_eigvect.shape[1])]),
                                      (ordered_eigvect.shape[0], 1))

    fract_ordered_eigval = eig_val[idx_order][:n_components] /\
        eig_val.sum() * 100
    pc_weights = pd.DataFrame()
    if task_names:
        for pc_idx in range(n_components):
            pc_weights['PC_{}'.format(pc_idx)] = pd.Series(
                dict(zip(task_names, ordered_eigvect[:, pc_idx])))
        fract_ordered_out = pd.Series(dict(zip(pc_weights.columns,
                                               fract_ordered_eigval)))
    else:
        pc_weights = ordered_eigvect[:, n_components]
        fract_ordered_out = fract_ordered_eigval
    return pc_weights, fract_ordered_out


def save_variables_to_csv(out_dir, do_percent_change, save_dict):
    if do_percent_change:
        suffix = 'percent'
    else:
        suffix = 'abs'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for df_name, df_used in save_dict.viewitems():
        fn = df_name + '_' + suffix + '.csv'
        df_used.to_csv(opj(out_dir, fn))


def save_all_figures(out_dir, use_figure_subdir=True, format_used='pdf'):
    if use_figure_subdir:
        od = opj(out_dir, 'figures')
    else:
        od = out_dir
    if not os.path.isdir(od):
        os.makedirs(od)
    fig_list = [plt.figure(f) for f in plt.get_fignums()]
    for f_idx, fig in enumerate(fig_list):
        fname = opj(od, 'figure_{}.{}'.format(f_idx, format_used))
        fig.savefig(fname, dpi=300, bbox_inches='tight', format=format_used)



'''
ANALYSIS SCRIPT
'''

if __name__ == '__main__':
    np.random.seed(0) # Fix random seed to allow exact reproducible results
    '''Load data'''

    TRACK_ON_data_dir = ('/data1/chdi_disks/Disk2/IBM_SOW3-part2/'
                         'Updated TRACK phenotypic dataset')

    subject_list, subjects_dict, gton, mri_TON, visit_dict = \
        hf.make_Track_ON_subjects(TRACK_ON_data_dir)

    subject_list = np.array(subject_list)

    '''Compute slopes'''

    slopes_df, visits_used_df = compute_slopes_df(subject_list, subjects_dict,
                                                  do_percent_change,
                                                  use_only_ton_visits)
    if do_save:
        visits_used_df.to_csv(opj(out_dir, 'n_visits_used.csv'))

    '''Filter good subjects and separate control and preHD slopes'''

    task_names = [u'cancelation', u'count_backwards', u'grip_var',
                  u'indirect_circle_trace', u'map_search', u'mental_rotation',
                  u'paced_tap', u'sdmt', u'spot_change', u'stroop']

    #Converted.csv not uploaded due to privacy agreement of sbuject data:
    converted_subjids = np.loadtxt('converted.csv', dtype=str)

    slopes_df['converted'] = pd.Series(dict(zip(slopes_df.index,
        [s in converted_subjids for s in slopes_df.index])))

    inclusion_dict = {'slopes_in_all_tasks': ~slopes_df[task_names].
                      isnull().any(axis=1),
                      'righthanded': slopes_df['right_handed'],
                      'eligible': slopes_df['eligible'] == 1,
                      'is_control_or_has_CAP': (slopes_df['group'] ==
                          'control') | (~slopes_df['CAP'].isnull()),
                      'is_control_or_has_CAG': (slopes_df['group'] ==
                          'control') | (~slopes_df['CAG'].isnull()),
                      'never_converted': ~slopes_df['converted']}

    ok_subjects = reduce(lambda x, y: x & y, inclusion_dict.values())
    inclusion_df = pd.DataFrame.from_dict(inclusion_dict)
    if do_save:
        inclusion_df.to_csv(opj(out_dir, 'inclusion_df.csv'))

    raw_slopes_ok_subjs = slopes_df[ok_subjects]

    control_data = raw_slopes_ok_subjs[raw_slopes_ok_subjs['group'] == 'control']
    control_slopes = control_data[task_names]
    preHD_data = raw_slopes_ok_subjs[raw_slopes_ok_subjs['group'] == 'preHD']
    preHD_slopes = preHD_data[task_names]

    '''
    PCA Representation of raw slopes
    '''

    all_slopes = raw_slopes_ok_subjs[task_names]
    rs = RobustScaler()
    scaled_all_slopes = (1.34896) * rs.fit_transform(all_slopes)
    mcd = MinCovDet() #random_state=1982)
    mcd.fit(scaled_all_slopes)
    all_slopes_corr = corr_from_cov(mcd.covariance_)
    plot_corr_matrix(all_slopes_corr, col_names=task_names,
                     title='All corr, raw')

    all_pcs, all_var_explained = PCs_from_cov(mcd.covariance_, task_names,
                                              n_components=n_components,
                                              convert_2_corr=True)

    # Properly centered DataFrame of scaled slopes
    stds = pd.Series(dict(zip(task_names, np.sqrt(mcd.covariance_.diagonal()))))
    rn_slopes = pd.DataFrame(dict(zip(task_names, scaled_all_slopes.T))).\
        set_index(all_slopes.index) / stds -\
        pd.Series(dict(zip(task_names, mcd.location_)))

    all_pc_vals = rn_slopes.dot(all_pcs)

    HD_idx = (raw_slopes_ok_subjs['group'] == 'preHD')
    ctrl_idx = (raw_slopes_ok_subjs['group'] == 'control')

    # Make scatter plot
    pcs_used = ['PC_0', 'PC_1']
    to_scat = all_pc_vals[pcs_used]
    plt.figure()
    plt.scatter(*to_scat.loc[HD_idx].values.T, color='blue')
    plt.hold('on')
    plt.scatter(*to_scat.loc[ctrl_idx].values.T, color='red')
    plt.xlabel(pcs_used[0])
    plt.ylabel(pcs_used[1])
    plt.legend(['preHD', 'Controls'])
    plt.title('All slopes, uncorrected on robust PCs')
    plt.show()

    # Make heatmap of PC loadings
    plt.figure()
    task_order = all_pcs['PC_0'].sort_values(ascending=False).index
    ah = PC_df_heatmap(all_pcs.ix[task_order], all_var_explained)
    ah.set_title('Robust PCs uncorrected, all slopes (Robust scaled)')

    '''Removal of covariate effects'''

    control_covariates = ['age', 'sex']
    ols = OLS#pd.stats.ols.OLS
    detrended_control_slopes = pd.DataFrame()
    control_betas = pd.DataFrame()

    for col_name in task_names:
        model = ols(control_slopes[col_name],
                sm.add_constant(control_data[control_covariates]))
        z = model.fit()
        detrended_control_slopes[col_name] = z.resid
        control_betas[col_name] = z.params #z.beta

    mcd_ctrl = MinCovDet()
    mcd_ctrl.fit(detrended_control_slopes)
    ctrl_scales = pd.Series(dict(zip(task_names, np.sqrt(mcd_ctrl.covariance_.
                                                         diagonal()))))
    ctrl_centers = pd.Series(dict(zip(task_names, mcd_ctrl.location_)))
    z_scored_control_slopes = (detrended_control_slopes - ctrl_centers)\
        / ctrl_scales

    '''
    Removing effect of age and sex in preHD
    '''

    preHD_covariates = preHD_data[control_covariates].copy()
    preHD_covariates['const'] = 1
    preHD_normal_detrended = preHD_slopes - preHD_covariates.dot(control_betas)
    z_scored_preHD_norm_detrended_slopes = (preHD_normal_detrended - ctrl_centers)\
        / ctrl_scales

    all_detrended_scaled_slopes = pd.concat([z_scored_control_slopes,
                                            z_scored_preHD_norm_detrended_slopes]).sort_index()
    mcd = MinCovDet()#random_state=1982)
    mcd.fit(all_detrended_scaled_slopes)
    all_detrended_pcs, all_det_var_explained = PCs_from_cov(mcd.covariance_,
                                                            task_names,
                                                            n_components=
                                                            n_components,
                                                            convert_2_corr=True)

    stds = pd.Series(dict(zip(task_names, np.sqrt(mcd.covariance_.
                                                  diagonal()))))
    rn_slopes = (all_detrended_scaled_slopes -\
        pd.Series(dict(zip(task_names, mcd.location_)))) / stds

    all_detrended_pc_vals = rn_slopes.dot(all_detrended_pcs)

    #Quantify difference between control and pre-HD:
    PC = all_detrended_pc_vals['PC_0']
    PC.groupby(raw_slopes_ok_subjs['group'] == 'preHD')
    rr = mannwhitneyu(PC[raw_slopes_ok_subjs['group'] == 'preHD'],
                  PC[raw_slopes_ok_subjs['group'] != 'preHD'])
    print(rr)

    # Make scatter plot
    pcs_used = ['PC_0', 'PC_1']
    to_scat = all_detrended_pc_vals[pcs_used]
    plt.figure()
    plt.scatter(*to_scat.loc[HD_idx].values.T, color='blue')
    plt.hold('on')
    plt.scatter(*to_scat.loc[ctrl_idx].values.T, color='red')
    plt.xlabel(pcs_used[0])
    plt.ylabel(pcs_used[1])
    plt.legend(['preHD', 'Controls'])
    plt.title('All slopes, corrected for crontrol age/sex, PCs')
    plt.show()

    # Make heat map of PC loadings
    plt.figure()
    task_order = all_detrended_pcs['PC_0'].sort_values(ascending=False).index
    ah = PC_df_heatmap(all_detrended_pcs.ix[task_order], all_det_var_explained,
                       n_max=5, annot=all_detrended_pcs.ix[task_order].values[:,:5], fmt='.1g')
    ah.set_title('Robust PCs age/sex corrected, all slopes (Robust scaled)')

    '''
    Removal of CAP score effect
    '''
    PreHD_covariates = ['CAP', 'CAG']
    ols = OLS # pd.stats.ols.OLS
    deCAPed_preHD_slopes = pd.DataFrame()
    preHD_betas = pd.DataFrame()


    for col_name in task_names:
        model = ols(z_scored_preHD_norm_detrended_slopes[col_name],
                sm.add_constant(preHD_data[PreHD_covariates]),
                hasconst=True)
        z = model.fit()
        deCAPed_preHD_slopes[col_name] = z.resid
        preHD_betas[col_name] = z.params

    mcd = MinCovDet()#random_state=666)
    mcd.fit(deCAPed_preHD_slopes)
    deCAPed_pcs, deCAPed_det_var_explained = PCs_from_cov(mcd.covariance_,
                                                            task_names,
                                                            n_components=
                                                            n_components,
                                                            convert_2_corr=False)

    stds = pd.Series(dict(zip(task_names, np.sqrt(mcd.covariance_.diagonal()))))

    # Notice that this is the rescaling expected by PCs produced by mcd method:
    rn_CAP_slopes = (deCAPed_preHD_slopes -\
        pd.Series(dict(zip(task_names, mcd.location_)))) / stds #Normalized preHD slopes
    deCAPed_pc_vals = rn_CAP_slopes.dot(deCAPed_pcs)

    # Robust scale PC projections
    scaler = RobustScaler()
    h  = (1.34896) * scaler.fit_transform(deCAPed_pc_vals)
    deCAPed_pc_vals_robust_scaled = pd.DataFrame(dict(zip(deCAPed_pc_vals.columns, h.T)),index=deCAPed_pc_vals.index)

    plt.figure()
    task_order = deCAPed_pcs['PC_0'].sort_values(ascending=False).index
    ah = PC_df_heatmap(deCAPed_pcs.ix[task_order], deCAPed_det_var_explained,
                       n_max=5, annot=deCAPed_pcs.ix[task_order].values[:,:5], fmt='.1g')
    ah.set_title('PreHD PCs CAP, age/sex corrected (Robust scaled)')

    '''
    Save results as csv file
    '''

    if do_save:
        save_dict = {'deCAPed_pc_vals': deCAPed_pc_vals_robust_scaled,
                     'deCAPed_pc_loadings': deCAPed_pcs,
                     'deCAPed_preHD_slopes': rn_CAP_slopes,  # deCAPed_preHD_slopes,
                     'all_detrended_pc_vals': all_detrended_pc_vals,
                     'all_detrended_pc_loadings': all_detrended_pcs,
                     'all_detrended_scaled_slopes': all_detrended_scaled_slopes,
                     'raw_slopes_ok_subjs': raw_slopes_ok_subjs}
        save_variables_to_csv(out_dir, do_percent_change, save_dict)

    '''
    Save plots
    '''
    if save_plots:
        save_all_figures(out_dir)
