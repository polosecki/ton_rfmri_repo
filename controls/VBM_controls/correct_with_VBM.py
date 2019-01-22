#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 12:12:51 2018

@author: pipolose
"""

import numpy as np
from pathlib2 import Path
import nibabel as nib
from polyML import polyssifier_TON as ps
from nilearn import image as nlim
from polyML.make_feature_stability_plot import where_matlab_ordered
import matplotlib.pyplot as plt
from scipy.io import savemat


in_mat = ('/data2/polo/code/MLtool/TON_resting_classification/proc_data/'
          'non_smooth_non_subsamp/thres07/TON_Resting_State_bct_degrees_log'
          '.mat')
mask_fn = ('/data2/polo/code/MLtool/TON_resting_classification'
           '/masks/Resting_State_TON_mask_strict.nii.gz')

gm_mat = (Path.cwd() / 'TON_gm_maps.mat').as_posix()

cooked_dir = Path('/data1/cooked/TONf')
out_dir = cooked_dir / 'rsfmri_ton_controls'

source = 'matlab'
do_save = False
load_gm = True

data, labels = ps.load_data(in_mat)[:2]
subject_list = ps.load_subject_list(in_mat, source=source)

runs_per_subj = float(data.shape[0]) / subject_list.shape[0]
assert (runs_per_subj - np.floor(runs_per_subj)) == 0
runs_per_subj = int(runs_per_subj)

gm_path_list = []

for subjid in subject_list:

    supath = cooked_dir / subjid
#   last_visit_vbm = sorted(supath.glob('visit*/T1W/vbm'))[-1]
    try:
        last_visit_vbm = sorted(supath.glob('visit*/T1W/vbm/cross-sect/T2mask_dataset/' + subjid + '*'))[-1].parent
    except IndexError:
        last_visit_vbm = Path('/')
    try:
        gm_last_visit_path = (last_visit_vbm).\
            glob('*normalized.nii.gz').next()
    except StopIteration:
        gm_last_visit_path = ''
    gm_path_list.append(gm_last_visit_path)


needs_recomputation = ~np.array(gm_path_list).astype(bool)
assert ~np.any(needs_recomputation)

if not load_gm:
    if not out_dir.exists():
        out_dir.mkdir()

    gm_data = np.zeros_like(data)
    gm_data[:] = np.nan

    mask_vol = nib.load(mask_fn)
    mask_arr = np.squeeze(mask_vol.get_data()).astype(bool)


    for sidx, vbm_path in enumerate(gm_path_list):
        resamp_img = nlim.resample_to_img(vbm_path.as_posix(), mask_fn,
                                          interpolation='linear')
        resamp_data = resamp_img.get_data()
        flat_map = resamp_data.swapaxes(0, 1)[where_matlab_ordered
                                              (mask_arr.swapaxes(0, 1))]
        start_row = runs_per_subj * sidx
        stop_row = start_row + runs_per_subj
        gm_data[start_row:stop_row, :] = flat_map
else:
    gm_data = ps.load_data(gm_mat)[0]

if do_save:
    with open(gm_mat, 'wb') as f:
        savemat(f, {'data': np.hstack((gm_data, labels[:,None])),
                    'fmri_subjects': subject_list})

'''
Global detrending by removing projection to same subject GM map:
    pro: -it ensures you are not just seeing a copy of the gm map
         -No need to look at other subjects
    con: -effects of gm might be local, and need comparisons across subjects
          for that
'''

gm_norm = np.linalg.norm(gm_data, axis=1, keepdims=True)
gm_scaled = gm_data / gm_norm

prod = np.array([np.dot(data[i,:],gm_scaled[i,:])
                 for i in range(data.shape[0])])

data_corrected = data - prod[:,None] * gm_scaled

if do_save:
    fname = Path.cwd() / 'TON_log_deg_maps_global_gm_corrected.mat'
    with open(fname.as_posix(), 'wb') as f:
        savemat(f, {'data': np.hstack((data_corrected, labels[:,None])),
                    'fmri_subjects': subject_list})


plotted_mat = [data, gm_data, data_corrected]
fh, ax_arr = plt.subplots(len(plotted_mat),1)

for aix, ax in enumerate(ax_arr):
    ax.imshow(plotted_mat[aix], interpolation='nearest', aspect='auto')

'''
Local detrending of gm effects across subjects by estimating linear
effects of gm on each voxel across subjects.
'''

n_pred = 2 # GM + offset
#Independent variable: gm + offset
X_h = np.ones((data.shape[0], n_pred))
#Dependent variable: data
y_h = np.ones((data.shape[0], 1))
#Learned coefficients: predictors x features (voxel)
out_coeff = np.zeros((n_pred, data.shape[1]))
this_coeff = np.zeros((2, 1))
corrected_coeff = np.zeros((2, 1))
local_detrended_data = np.zeros_like(data)

local_censored_detrended_data = np.zeros_like(data)

'''
Note the range of slopes is unusually large: saving truncated slope values
'''

for cidx in range(data.shape[1]):
    y_h[:,0] = data[:, cidx]
    X_h[:, 0] = gm_data[:, cidx] - gm_data[:, cidx].mean()
    this_coeff[:, 0] = np.linalg.lstsq(X_h, y_h)[0][:, 0]
    if gm_data[:, cidx].mean() < .01:
        corrected_coeff[0, 0] = 0
        corrected_coeff[1, 0] = this_coeff[1, 0]
        print('a')
    else:
        corrected_coeff = this_coeff
    local_detrended_data[:, cidx] = data[:, cidx] - X_h.dot(this_coeff).flatten()
    out_coeff[:, cidx] = this_coeff.flatten()

    local_censored_detrended_data[:, cidx] = data[:, cidx] -\
        X_h.dot(s).flatten()


fh, ax = plt.subplots(1)
ax.plot(gm_data.mean(axis=0), out_coeff[0,:], '.')
ax.set_title('GM close to 0 leads to wildly changing betas')

if do_save:
    fname = Path.cwd() / 'TON_log_deg_maps_local_gm_corrected.mat'
    with open(fname.as_posix(), 'wb') as f:
        savemat(f, {'data': np.hstack((local_detrended_data, labels[:,None])),
                    'fmri_subjects': subject_list})
    fname = Path.cwd() / 'TON_log_deg_maps_local_gm_censor_corrected.mat'
    with open(fname.as_posix(), 'wb') as f:
        savemat(f, {'data': np.hstack((local_censored_detrended_data,
                                       labels[:,None])),
                    'fmri_subjects': subject_list})

fh, axs = plt.subplots(2,1)
im_list= [data, data - out_coeff[0,:] * (gm_data - gm_data.mean(axis=0))]

for aix, ax in enumerate(axs):
    ax.imshow(im_list[aix], interpolation='nearest', aspect='auto')


