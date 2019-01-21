#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:02:56 2017

@author: pipolose
"""
import numpy as np
import pandas as pd
import helper_functions as hf

TRACK_ON_data_dir = ('/data1/chdi_disks/Disk2/IBM_SOW3-part2/'
                     'Updated TRACK phenotypic dataset')

subject_list, subjects_dict, gton, mri_TON, visit_dict = \
    hf.make_Track_ON_subjects(TRACK_ON_data_dir)

subject_list = np.array(subject_list)

siteid = np.zeros(subject_list.shape)

for sidx, su in enumerate(subject_list):
    siteid[sidx] = subjects_dict[su].siteid

out_df = pd.DataFrame({'subjid': subject_list,
                       'siteid': siteid.astype(int)}).set_index('subjid')

out_fn = '/data2/polo/half_baked_data/TON_siteid.csv'
