# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:47:32 2016

@author: pipolose
"""

import pandas as pd
import glob
import os
import explore.helper_functions as hf
import numpy as np
from collections import defaultdict as defd

'''
BASIC DATA LOADING
'''
# Create subject_list and such using explore.helper_functions:
TRACK_ON_data_dir = ('/data1/chdi_disks/Disk2/IBM SOW3-part2/'
                     'Updated TRACK phenotypic dataset')

subject_list, subjects_dict, gton, mri_TON, visit_dict = \
    hf.make_Track_ON_subjects(TRACK_ON_data_dir)

outlier_csv = '/data2/polo/half_baked_data/TON_rsfMRI/outlier_runs.csv'

'''
Get subjects in TONf directory
'''
TONf_dir = '/data1/cooked/TONf'
dl = glob.glob(os.path.join(TONf_dir, 'R*'))

subjs_in_TONf = np.array([sl.split('/')[-1:][0] for sl in dl])
subjs_in_TONf.sort()


'''
Get the subject clinical groups (1: PreHD, -1: Control)
'''
gr_list = [subjects_dict[st].group for st in subjs_in_TONf]
gr_dict = {'preHD': 1, 'control': -1}
gr_num = np.array([gr_dict[grv] for grv in gr_list])

'''
Get site info of each subject
'''
siteid = np.array([subjects_dict[st].siteid for st in subjs_in_TONf])
age = np.array([subjects_dict[st].demographics['age'] for st in subjs_in_TONf])
sex = np.array([subjects_dict[st].demographics['sex'] for st in subjs_in_TONf])

'''
Get first and last visit of each subject
'''
def get_visit(subject_ID, TON_dir, which_visit, exclusion_csv= None,
              scan_subtype='Resting_State'):
    from os.path import join as opj, isdir as isdir
    if exclusion_csv:
        exclusion_df = pd.read_csv(exclusion_csv)
    dict_of_lists = {'first': [1, 2, 3], 'last': [3, 2, 1]}
    for visit_no in dict_of_lists[which_visit]:
        if not exclusion_csv:
            excluded = False
        else:
            excluded = ((exclusion_df['subjid'] == subject_ID) &
                        (exclusion_df['visit'] == visit_no)).any()
        if isdir(opj(TON_dir, subject_ID, 'visit_' + str(visit_no),
                     scan_subtype)) and not excluded:
            return visit_no
        if visit_no is dict_of_lists[which_visit][-1]:
            return 0

def subject_is_complete(subject_ID, TON_dir, which_visit,
                        scan_subtype='Resting_State'):
    from os.path import join as opj, isdir as isdir
    visit_list = [1, 2, 3]
    is_complete = True
    for vis_no in visit_list:
        if not isdir(opj(TON_dir, subject_ID, 'visit_' + str(vis_no),
                         scan_subtype)):
                             is_complete = False
    return is_complete

def get_sdmt_slope(subject):
    vis_names = subject.pheno_df.index.tolist()
    vy = np.array([subject.visdy[vn]/365. for vn in vis_names])
    sdmt = np.array([subject.pheno_df.loc[vn]['SDMT'] for vn in vis_names])
    try:
        m, b = np.linalg.lstsq(np.stack((vy, np.ones(vy.shape)),axis=1),sdmt)[0]
    except:
        m, b = np.nan, np.nan
    return m, b

sdmt_slopes = np.array([get_sdmt_slope(subjects_dict[st])[0] for st in subjs_in_TONf])


first_visit = np.array([get_visit(st, TONf_dir, 'first',
                                  exclusion_csv=outlier_csv)
                        for st in subjs_in_TONf])
last_visit = np.array([get_visit(st, TONf_dir, 'last',
                                 exclusion_csv=outlier_csv)
                       for st in subjs_in_TONf])


regress_names = ['yto', 'CAP','TMS','SDMT', 'category']
first_reg_dict = {name: np.empty(first_visit.shape) * np.nan for name in regress_names}
last_reg_dict = {name: np.empty(first_visit.shape) * np.nan for name in regress_names}

first_reg_dict['category'] = np.empty(first_visit.shape).astype(str)
last_reg_dict['category'] = np.empty(first_visit.shape).astype(str)


#first_yto = np.empty(first_visit.shape) * np.nan
#last_yto = np.empty(first_visit.shape) * np.nan

for name in regress_names:
    first_vect = first_reg_dict[name]
    last_vect = last_reg_dict[name]
    for s_i, s_name in enumerate(subjs_in_TONf):
        if first_visit[s_i] != 0:
            first_vect[s_i] = subjects_dict[s_name].pheno_df.loc['visit' + str(first_visit[s_i]) + '_ton'][name]
        if last_visit[s_i] != 0:
            last_vect[s_i] = subjects_dict[s_name].pheno_df.loc['visit' + str(last_visit[s_i]) + '_ton'][name]

'''
Make Dataframe and save
'''
ok_subjs = first_visit != 0

reg_select_dict_first = {}
reg_select_dict_last = {}

for name in regress_names:
    reg_select_dict_first['first_' + name] = first_reg_dict[name][ok_subjs]
    reg_select_dict_last['last_' + name] = last_reg_dict[name][ok_subjs]

dict_for_df = {'subject_IDs': subjs_in_TONf[ok_subjs],
               'group': gr_num[ok_subjs],
               'site_id': siteid[ok_subjs],
               'first_visit': first_visit[ok_subjs],
               'last_visit': last_visit[ok_subjs],
               'sdmt_slope' : sdmt_slopes[ok_subjs],
               'age': age[ok_subjs],
               'sex': sex[ok_subjs]}
dict_for_df.update(reg_select_dict_first)
dict_for_df.update(reg_select_dict_last)

out_df = pd.DataFrame(dict_for_df).set_index('subject_IDs').sort_index()

if 'first_category' in out_df.columns:
    cats = defd(int)
    cats.update({'control': 0, 'preHD-A': 1, 'preHD-B': 2, 'HD1': 3, 'HD2': 4})
    cols_checked = ['first_category', 'last_category']
    for colname in cols_checked:
        zz=out_df[colname]
        zz.replace(cats,inplace=True)
        pd.to_numeric(out_df[colname],errors='coerce')
fn = ['first_' + name for name in regress_names]
ln = ['last_' + name for name in regress_names]
sl = zip(fn, ln)
dn = [n for tupy in sl for n in tupy]

do_save = False
if do_save:
    out_df[['group', 'site_id', 'first_visit', 'last_visit', 'sdmt_slope',
            'age', 'sex'] + dn].\
          to_csv('TON_info_for_MLtools.csv')


do_Q_stats = True
if do_Q_stats:
    #Q1_subjs_IDs = np.genfromtxt('Q3_subject_list.csv', dtype=str)
    from scipy.io import loadmat
#    mname = '/data1/polo/half_baked_data/DTI/DTI_nFA_data_filtered.mat'
    mname = ('/data2/polo/code/MLtool/TON_resting_classification/proc_data/'
             'non_smooth_non_subsamp/thres07/'
             'TON_Resting_State_bct_degrees_log_norm_median.mat')
    source = 'matlab'
    #source = 'python'
    zz = loadmat(mname)['fmri_subjects']
    if source == 'matlab':
        Q1_subjs_IDs = np.array([s[0][0] for s in zz])
    elif source == 'python':
        Q1_subjs_IDs = np.array([str(s[0]) for s in zz[0]])
    in_Q1 = np.array([sid in Q1_subjs_IDs for sid in subjs_in_TONf])
    so_in_Q1 = np.array([subjects_dict[sid] for sid in subjs_in_TONf[in_Q1]])
    age = np.array([so.demographics['age'] for so in so_in_Q1])
    sex = np.array([so.demographics['sex'] for so in so_in_Q1])

    dict_for_df = {'subjids': subjs_in_TONf[in_Q1],
                   'group': gr_num[in_Q1],
                   'age': age,
                   'sex': sex}
    do_CAP_CAG = True
    if do_CAP_CAG:
        cap_arr = np.zeros(so_in_Q1.shape)
        cag_arr = np.zeros(so_in_Q1.shape)
        yto_arr = np.zeros(so_in_Q1.shape)
        for six, so in enumerate(so_in_Q1):
            fv = 'visit{}_ton'.format(out_df.loc[so.subjid]['first_visit'])
            # cap_arr[six] = so.pheno_df.loc[fv]['CAP']
            cag_arr[six] = so.pheno_df.loc[fv]['caghigh']
            cap_arr[six] = so.pheno_df.loc[fv]['age'] *\
                           (so.pheno_df.loc[fv]['caghigh']-35.5) *100./627
            yto_arr[six] = so.pheno_df.loc[fv]['yto']
        dict_for_df['CAP'] = cap_arr
        dict_for_df['CAG'] = cag_arr
        dict_for_df['yto'] = yto_arr
    Q1df = pd.DataFrame(dict_for_df).set_index('subjids').sort_index()

    Q1df.groupby(by='group').count()
    Q1df[Q1df.group==-1].groupby(by='sex').count()
    Q1df[Q1df.group==1].groupby(by='sex').count()
    Q1df[Q1df.group==1].age.mean()
    Q1df[Q1df.group==1].age.std()
    Q1df[Q1df.group==-1].age.std()
    Q1df[Q1df.group==-1].age.mean()
    Q1df[Q1df.group==1].yto.mean()
    Q1df[Q1df.group==1].yto.std()

