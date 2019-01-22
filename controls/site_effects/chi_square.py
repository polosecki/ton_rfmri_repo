#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 14:25:39 2018

@author: pipolose
"""
# from scipy.stats import chisquare
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
import numpy as np
import pandas as pd
from polyML import polyssifier_3 as ps
from pathlib2 import Path
from collections import OrderedDict


do_save = False
out_dir = Path('/data2/polo/figures')
csv_dir = Path('/data2/polo/half_baked_data')
dem_csv = csv_dir / 'TON_baseline_demo.csv'
site_csv = csv_dir / 'TON_siteid.csv'

in_mat = Path().cwd().parent / 'VBM_controls' /\
    'TON_log_deg_maps_local_gm_corrected.mat'

source = 'python'
subject_list = ps.load_subject_list(in_mat.as_posix(), source=source)



sites_df = pd.read_csv(site_csv, index_col='subjid')
dem_df = pd.read_csv(dem_csv, index_col='subjid')
dem_df['site'] = sites_df.loc[dem_df.index]
dem_df['count'] = 1
used_dem = dem_df.loc[subject_list]

'''
Group-site associations
'''
freq = used_dem.groupby(['site', 'group']).count()['count']
g, p_value, dof, expctd = chi2_contingency(freq.unstack())

'''
CAG/CAP-site associations
'''
site_ids = used_dem['site'].unique()
obs = {'CAG': [], 'CAP': []}
p_val_dict = {'CAG': [], 'CAP': []}
for sid in site_ids:
    this_sid_obs = (used_dem['group'] == 'preHD') & (used_dem['site'] == sid)
    for key in obs.keys():
        obs[key].append(used_dem.loc[this_sid_obs][key].values)
for key in p_val_dict.keys():
    p_val_dict[key] = kruskal(*obs[key])
# freq = used_dem[used_dem['group'] == 'preHD'].groupby('site').mean()['CAG']


csv_format = ('/data1/chdi_results/polo/polyML/results/degree/'
              'bct/thres07/non_smooth/happy_sad/{}'
              '/SAGA_log_elastic_predictions.csv')
task_list = ['PC_0', 'PC_1', 'PC_2', 'PC_3', 'PC_4', 'cancelation',
             'sdmt', 'grip_var', 'stroop', 'map_search', 'spot_change',
             'mental_rotation', 'mental_rotation', 'count_backwards',
             'paced_tap']

# subgroup_dict = OrderedDict()
count_dict = OrderedDict()
for task in task_list:
    csv_fn = Path(csv_format.format(task))
    pred_df = pd.read_csv(csv_fn, index_col='subjid',
                          usecols=['subjid', 'labels']).astype(int)
    subgroups = pd.Series(np.zeros((used_dem['group'] == 'preHD').sum()),
                          index=used_dem.index[used_dem['group'] == 'preHD'])\
        .astype(int)
    subgroups.loc[pred_df.index] = pred_df['labels']
    count_dict[task] = subgroups.value_counts()
#    subgroup_dict[task] = subgroups

# subgroup_dict['site'] = used_dem.loc[subgroups.index]['site']
# subgroup_df = pd.DataFrame(subgroup_dict)

subgroup_counts_df = pd.DataFrame(count_dict).T

if do_save:
    subgroup_counts_df.to_csv(out_dir / 'subgroups_count.csv')
