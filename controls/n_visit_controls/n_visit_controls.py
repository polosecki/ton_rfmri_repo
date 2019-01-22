#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 16:00:46 2019

@author: pipolose
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib2 import Path
from polyML import polyssifier_3 as ps
from scipy.stats import pearsonr, mannwhitneyu, kruskal
from collections import OrderedDict

'''
USER OPTIONS
'''
out_dir = Path('/data2/polo/figures')
do_save = False

'''
SOURCE DATA
'''
task_dir = Path('/data2/polo/half_baked_data/slopes/abs')
single_task_slope_csv = task_dir / 'raw_slopes_ok_subjs_abs.csv'
corrected_single_task_csv = task_dir / 'deCAPed_preHD_slopes_abs.csv'
n_visit_csv = task_dir / 'n_visits_used.csv'
in_mat = Path().cwd().parent / 'VBM_controls' /\
    'TON_log_deg_maps_local_gm_corrected.mat'

'''
LOAD
'''
source = 'python'
subject_list = ps.load_subject_list(in_mat.as_posix(), source=source)
n_df = pd.read_csv(n_visit_csv, index_col='subjid')
slope_df = pd.read_csv(single_task_slope_csv, index_col='subjid')

task_names = n_df.columns.values

behav_n_imag = [s for s in subject_list if s in slope_df.index]

x = slope_df.loc[behav_n_imag]['group'] == 'preHD'
preHD_idx = x.loc[x].index.values
n_df.corrwith(slope_df.loc[preHD_idx][task_names])
task_corrs = OrderedDict()

corr_dict = OrderedDict()
corr_dict['task'] = task_names
p_vals = []
corr_vals = []

for task in task_names:
    x = n_df.loc[preHD_idx][task]
    y = slope_df.loc[preHD_idx][task]
    corr, p = pearsonr(x, y)
    p_vals.append(p)
    corr_vals.append(corr)

corr_dict['p-value'] = p_vals
corr_dict['corr'] = corr_vals
corr_dict['variance-explained'] = [100*(p**2) for p in corr_vals]
corr_dict['Bonferroni p-val'] = [p*len(p_vals) for p in p_vals]


raw_corr_df = pd.DataFrame(corr_dict)
if do_save:
    raw_corr_df.to_csv(out_dir / 'slope_n_visit_correlation.csv')


with sns.axes_style("darkgrid"):
    sns.set(font='helvetica')
    fh, ah = plt.subplots()
    fh.set_size_inches([4.64, 6.68])
    axes = n_df.loc[preHD_idx].hist(ax=ah, sharex=True, sharey=True,
                                    bins=np.arange(2, 9) - .5, layout=(5, 2))

for ax in axes.flatten():
    old_title = ax.get_title()
    new_title = old_title.replace('_', ' ').title()
    ax.set_title(new_title)
    if ax.get_xticklabels():
        ax.set_xticks(np.arange(2, 8))
        ax.set_xlabel('N visits included')
    if ax.get_yticklabels():
        ax.set_ylabel('Counts')


if do_save:
    if not out_dir.exists():
        out_dir.mkdir()
    fh.savefig((out_dir / 'n_visit_hist.pdf').as_posix(),
               bbox_inches='tight')

'''
Finally show group by n_visits association or not
'''

csv_format = ('/data1/chdi_results/polo/polyML/results/degree/'
              'bct/thres07/non_smooth/happy_sad/{}'
              '/SAGA_log_elastic_predictions.csv')
task_list = ['cancelation',
             'sdmt', 'grip_var', 'stroop', 'map_search', 'spot_change',
             'mental_rotation', 'mental_rotation', 'count_backwards',
             'paced_tap']

subgroup_dict = OrderedDict()
labels = slope_df.loc[subject_list]['group']  # Pandas Series
preHD_subjs = labels[labels == 'preHD'].index

count_dict = OrderedDict()
subg_visit_dict = OrderedDict()

subgroups_compared = [-1, 0, 1] #[-1, 1]  #
for task in task_list:
    csv_fn = Path(csv_format.format(task))
    pred_df = pd.read_csv(csv_fn, index_col='subjid',
                          usecols=['subjid', 'labels']).astype(int)
    subgroups = pd.DataFrame(np.zeros(preHD_subjs.shape),
                             index=preHD_subjs, columns=[task]).astype(int)
    subgroups.at[pred_df.index, [task]] = pred_df['labels']
    subgroups['n_visits'] = n_df.loc[subgroups.index.values][task]
    subgroup_dict[task] = subgroups
    listed_visits = []
    for sg in subgroups_compared:
        sg_subjects = (subgroups[task] == sg)
        listed_visits.append(subgroups['n_visits'].loc[sg_subjects])
    if len(subgroups_compared) == 2:
        subg_visit_dict[task] = mannwhitneyu(*listed_visits,
                                             alternative='two-sided')
    else:
        subg_visit_dict[task] = kruskal(*listed_visits)

mann_df = pd.DataFrame(subg_visit_dict, index=['statistic', 'p-value']).T
print('Effect of n_visits in subgroup assignement:')
print(mann_df)

if do_save:
    mann_df.to_csv(out_dir / 'subgroup_{}_n_visit_effect.csv'.
                   format(len(subgroups_compared)))
