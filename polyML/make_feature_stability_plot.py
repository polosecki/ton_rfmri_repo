# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:18:26 2016

@author: pipolose
"""

'''
This file contains functions for converting classifier wieghts into
NIFTI format
'''

from os.path import join as opj
import pickle
import numpy as np
import scipy
from timeit import default_timer
import nibabel as nb
from nilearn import plotting
from matplotlib import cm
from matplotlib import pyplot as plt


'''
USER-DEFINED VARIABLES
'''
map_type = 'weight'  #'selection_freq' #

out_dir = '/data1/chdi_results/polo/polyML/results/degree/bct/thres07/non_smooth/cross-site-CV/happy_sad/PC_0'
#'/data1/chdi_results/polo/polyML/results/degree/bct/thres07/non_smooth/cross-site-CV/age-sex-corrected'

classifier_used = 'SAGA_log_elastic'
n_folds = 4 #4 sites
uni_select = 'ttest'
thres_used = 2

'''
The 3-D mask that was used for flattening the data to 1-D
'''
brain_mask_fname = ('/data2/polo/code/MLtool/TON_resting_classification'
                    '/masks/Resting_State_TON_mask_strict.nii.gz')

used_ntop_indx = None  # If none, pick classifier with highest TopKFeatures val
feature_origin = 'matlab'  # language used to save the feature matrix



'''
FUNCTION DEFINITIONS
'''
def where_matlab_ordered(A):
    A_swapped = A.T
    swapped_idx = np.where(A_swapped)
    ok_idx = swapped_idx[::-1]
    return ok_idx

def vector2nii(in_vect,mask_fn,out_fn,source_language='matlab'):
    mask_vol_obj = nb.load(mask_fn)
    brain_mask = mask_vol_obj.get_data()
    bm_per_language = {'python': brain_mask,
                       'matlab': brain_mask.swapaxes(0, 1)}
    bm_used=bm_per_language[source_language]
    ind_to_vox_func_dict = {'python': np.where, 'matlab': where_matlab_ordered}
    br_idx = ind_to_vox_func_dict[source_language](bm_used==1)
    out_vol = bm_used.astype(float).copy()
    out_vol[br_idx] = in_vect
    out_header = mask_vol_obj.header
    out_affine = mask_vol_obj.affine
    out_vol_per_language = {'python': out_vol,
                            'matlab': out_vol.swapaxes(0, 1)}
    out_img = nb.Nifti1Image(out_vol_per_language[source_language],
                             out_affine, header=out_header)
    nb.save(out_img, out_fn)


def make_sel_freq_map(univar_ranking_fn, classifier_fn, used_ntop_indx=None):

    '''
    Data load
    '''
    rd = scipy.io.loadmat(univar_ranking_fn, mat_dtype=True)
    rank_per_fold = rd['rank_per_fold']

    with open(classifier_fn, 'rb') as po:
        try:
            clf_dict = pickle.load(po)
        except UnicodeDecodeError:
            clf_dict = pickle.load(po, encoding='latin1')
    fcs = clf_dict['FittedClassifiers']

    if used_ntop_indx is None:
        used_ntop_indx = np.array(fcs).shape[1] - 1
    fcs_per_split = np.array(fcs).T[used_ntop_indx, :]

    '''
    Computation
    '''

    used_vox_per_split = np.array([~np.isclose(this_fc.coef_, 0).astype(bool)
                                   for this_fc in fcs_per_split]).squeeze()

    used_vox_idx_per_split = [set(np.array(rank_per_fold[split]
                                       [used_vox_per_split[split]])) #Removed a -1 from here
                          for split in range(n_folds)]

    selection_ranking = np.zeros((len(rank_per_fold[0]),), dtype=float)
    st = default_timer()
    for idx_set in used_vox_idx_per_split:
        for k in range(selection_ranking.shape[0]):
            if k in idx_set:
                selection_ranking[k] += 1
    et = default_timer()
    print('Elapsed time during stability ranking: {:.2f}'.format(et-st))
    selection_ranking /= n_folds
    return selection_ranking

def make_avg_weight_map(univar_ranking_fn, classifier_fn, used_ntop_indx=None,
                        folds_used=None):

    '''
    Data load
    '''
    rd = scipy.io.loadmat(univar_ranking_fn, mat_dtype=True)
    rank_per_fold = rd['rank_per_fold']

    with open(classifier_fn, 'rb') as po:
        try:
            clf_dict = pickle.load(po)
        except UnicodeDecodeError:
            clf_dict = pickle.load(po, encoding='latin1')

    fcs = clf_dict['FittedClassifiers']

    if used_ntop_indx is None:
        used_ntop_indx = np.array(fcs).shape[1] - 1
    fcs_per_split = np.array(fcs).T[used_ntop_indx, :]
    weights_per_split = np.array([this_fc.coef_ for this_fc in fcs_per_split]).squeeze()



    '''
    Make a feature ranking
    '''
    nfeats = weights_per_split.shape[1]
    mean_weights = np.zeros((len(rank_per_fold[0]),), dtype=float)
    st = default_timer()
    if folds_used is None:
        folds_used = np.arange(weights_per_split.shape[0])
    elif isinstance(folds_used, list):
        folds_used = np.array(folds_used)
    for fold_idx, idx_from_fold in enumerate(rank_per_fold):
        if fold_idx in folds_used:
            idx_to_use = idx_from_fold[:nfeats]
            mean_weights[idx_to_use] += weights_per_split[fold_idx, :]
    et = default_timer()
    print('Elapsed time during stability ranking: {:.2f}'.format(et-st))
    mean_weights /= folds_used.shape[0]
    mean_weights /= (mean_weights.std())
    return mean_weights


if __name__ == "__main__":


    '''
    Make a feature ranking or weight map
    '''
    feature_map_func_dict = {'weight': make_avg_weight_map,
                             'selection_freq': make_sel_freq_map}

    unifile = opj(out_dir, '_'.join([uni_select, 'ind',
                                 str(n_folds), 'folds.mat']))
    classifier_fn = opj(out_dir, classifier_used + '.pkl')
    result_map = feature_map_func_dict[map_type](univar_ranking_fn=unifile,
                                                 classifier_fn=classifier_fn,
                                                 used_ntop_indx=used_ntop_indx)

    '''
    Save map as NIFTI
    '''
    out_fn = opj(out_dir, '_'.join([classifier_used.replace(' ', '_'),
                                    map_type, 'nfolds',
                                    str(n_folds) + '.nii.gz']))

    vector2nii(result_map, brain_mask_fname,
               out_fn,source_language=feature_origin)

    '''
    Show the resulting map:
    '''
    # Plotting with nilearn:
    # http://nilearn.github.io/plotting/index.html
    # http://nilearn.github.io/auto_examples/01_plotting/plot_demo_plotting.html

    cmap_func_dict = {'weight': cm.bwr,
                      'selection_freq': cm.autumn}
    pg = plotting.plot_glass_brain(out_fn, title='Th={:.2f}'.
                              format(thres_used),
                              threshold=thres_used,
                              display_mode='lyrz',
                              plot_abs=False)

    pv = plotting.plot_stat_map(out_fn, threshold=thres_used,
                           cmap=cmap_func_dict[map_type],
                           black_bg=True)

    pg.savefig(opj(out_dir, '_'.join([classifier_used.replace(' ', '_'),
                                      map_type, 'nfolds', str(n_folds),
                                      'glasbrain.pdf'])))
    pv.savefig(opj(out_dir, '_'.join([classifier_used.replace(' ', '_'),
                                      map_type, 'nfolds', str(n_folds),
                                      '3plane.pdf'])))

    plt.show()
