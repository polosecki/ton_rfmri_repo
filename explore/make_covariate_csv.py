#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:24:49 2018

@author: pipolose
"""

# =============================================================================
# Making a simple csv with age and sex for TON subjects at baseline.
# =============================================================================


import numpy as np
import pandas as pd
import helper_functions as hf

out_fn = '/data2/polo/half_baked_data/TON_baseline_demo.csv'
do_save = True
# Create subject_list and such using explore.helper_functions:
TRACK_ON_data_dir = ('/data1/chdi_disks/Disk2/IBM_SOW3-part2/'
                     'Updated TRACK phenotypic dataset')

subject_list, subjects_dict, gton, mri_TON, visit_dict = \
    hf.make_Track_ON_subjects(TRACK_ON_data_dir)

subject_list = np.array(subject_list)

gton.set_index('subjid', inplace=True)
base_visit = 'visit1_ton'
gton_visdy = gton['visdy']
gton_age = gton['age']

base_age = gton_age.copy()
cag = pd.Series(data=np.nan, index=base_age.index)
cap = pd.Series(data=np.nan, index=base_age.index)
sex = pd.Series(data=np.nan, index=base_age.index)
group = pd.Series(data=np.nan, index=base_age.index).astype(str)


for subjid in base_age.index:
    su = subjects_dict[subjid]
    base_age.at[subjid] = gton_age.loc[subjid] +\
        (su.visdy[base_visit] - gton_visdy[subjid]) / 365.
    cag.at[subjid] = su.pheno_df['caghigh'].mean()
    cap.at[subjid] = base_age.at[subjid] * (cag.at[subjid] - 35.5) * 100. / 627
    sex.at[subjid] = su.demographics['sex'] - 1  # i.e. 1 is male, 0 is female
    group.at[subjid] = su.group


out_df = pd.DataFrame.from_dict({'age': base_age,
                                 'CAG': cag,
                                 'CAP': cap,
                                 'sex': sex,
                                 'group': group})
if do_save:
    out_df.to_csv(out_fn)
