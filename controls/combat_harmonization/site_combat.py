#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:41:06 2018

@author: pipolose
"""

import numpy as np
import pandas as pd
from explore import helper_functions as hf
from polyML import polyssifier_3 as ps
from scipy.io import savemat
from pathlib2 import Path
from feature_making.harmonization_tools import combat_harmonize_site as chs

# Create subject_list and such using explore.helper_functions:
TRACK_ON_data_dir = ('/data1/chdi_disks/Disk2/IBM_SOW3-part2/'
                     'Updated TRACK phenotypic dataset')

subject_list, subjects_dict, gton, mri_TON, visit_dict = \
    hf.make_Track_ON_subjects(TRACK_ON_data_dir)

do_save = True

in_mat = Path('/data2/polo/code/MLtool/TON_resting_classification/proc_data/'
              'non_smooth_non_subsamp/thres07/'
              'TON_Resting_State_bct_degrees_log.mat')
source = 'matlab'

data, labels = ps.load_data(str(in_mat))[:2]
subject_list = ps.load_subject_list(str(in_mat), source=source)

runs_per_subj = float(data.shape[0]) / subject_list.shape[0]
assert (runs_per_subj - np.floor(runs_per_subj)) == 0
runs_per_subj = int(runs_per_subj)
samples_subj_list = np.empty((data.shape[0],)).astype(str)
for six, subjid in enumerate(subject_list):
    samples_subj_list[runs_per_subj * six: runs_per_subj * (six+1)] = subjid


sites = np.array([subjects_dict[subjid].siteid
                  for subjid in samples_subj_list])


for k in range(runs_per_subj):
    z = samples_subj_list[k::runs_per_subj]
    samples_subj_list[k::runs_per_subj] = np.array([(r + str(k)) for r in z])

site_df = pd.DataFrame({'siteid': sites}, index=samples_subj_list)
harmonized_data = chs(data, samples_subj_list, site_df)

if do_save:
    fname = Path.cwd() / (in_mat.stem + '_combat.mat')
    with open(fname.as_posix(), 'wb') as f:
        savemat(f, {'data': np.hstack((harmonized_data, labels[:, None])),
                    'fmri_subjects': subject_list})
