# -*- coding: utf-8 -*-
"""
This file calls functions of polyssifier. It is problem-dependent
(as opposed to the relatively stable polyssifier functions)

@author: pipolose
"""

import polyssifier_3 as ps
import logging
from sklearn.model_selection import GroupKFold
from os import path
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd

'''
INPUT PARAMETERS
'''

mname = ('/data2/polo/code/MLtool/TON_resting_classification/proc_data/'
         'non_smooth_non_subsamp/thres07/'
         'TON_Resting_State_bct_degrees_log_norm_median.mat')

ksplit = 'site' #'LOO'
numTopVars = [10000]
runs_per_subj = 2

do_preHD_age_sex_detrend = True

classify_happy_sad = False
happy_sad_label = 'count_backwards' #'PC_'#'count_backwards'#'sdmt'  # 'PC_5'  # 'grip_var'
do_happy_sad_detrend_CV = True
out_dir = '/data1/chdi_results/polo/polyML/results/degree/bct/thres07/non_smooth/cross-site-CV'

if do_preHD_age_sex_detrend:
    out_dir += '/age-sex-corrected/'
if classify_happy_sad:
    out_dir += '/happy_sad/' + happy_sad_label

    # If false, load classifier results instead of running:
compute_results = True

NAMES = ["SAGA_log_elastic"]


if __name__ == "__main__":
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    '''
    Initializing logger to write to file and stdout
    '''
    logging.basicConfig(format="[%(asctime)s:%(module)s:%(levelname)s]:%(message)s",
                        filename=path.join(out_dir, 'log.log'),
                        filemode='w',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    ch = logging.StreamHandler(logging.sys.stdout)
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    '''
    DATA LOADING
    '''
    data, labels, data_file = ps.load_data(mname)
    logging.info("Trying to load data")
    if np.any(np.isnan(data)):
        h = np.nonzero(np.isnan(data))
        data[h[0], h[1]] = 0
        logging.warning('Nan values were removed from data')
    if np.any(np.isinf(data)):
        h = np.nonzero(np.isinf(data))
        data[h[0], h[1]]=0
        logging.warning('Inf values were removed from data')
        # Load list of subject names
    zz = loadmat(mname)['fmri_subjects']
    subject_list = np.array([s[0][0] for s in zz])

        # Use the full brain:
    numTopVars.append(data.shape[1])
    filename_base = path.splitext(path.basename(mname))[0]
    covariate_detrend_params = None
    if classify_happy_sad:
        logging.info("Performing Happy Sad filtering")
        import TON_tools3 as TON_tools
        in_csv = '/data2/polo/half_baked_data/slopes/abs/raw_slopes_ok_subjs_abs.csv'

        # Define the control variables:
        healthy_ctrl_vars = ['age', 'sex']
        preHD_ctrl_vars = ['CAG', 'CAP']

        if do_happy_sad_detrend_CV:
            covariate_detrend_params = {'data': data.copy(),
                                        'preHD_healthy_labels': labels,
                                        'subject_list': subject_list,
                                        'in_csv': in_csv,
                                        'healthy_ctrl_vars': healthy_ctrl_vars,
                                        'preHD_ctrl_vars': preHD_ctrl_vars,
                                        'runs_per_subj': runs_per_subj,
                                        'add_intercept': True,
                                        'idx_classified': []}
        else:
            # Apply detrending to data
            detrended_data = TON_tools.detrend_data(data, labels,
                                                    subject_list, in_csv,
                                                    healthy_ctrl_vars,
                                                    preHD_ctrl_vars,
                                                    runs_per_subj=2)[0]
            data = detrended_data
            covariate_detrend_params = None

        # Make happy_sad labels
        in_csv = '/data2/polo/half_baked_data/slopes/abs/deCAPed_preHD_slopes_abs.csv'
        #in_csv = '/data2/polo/half_baked_data/slopes/abs/deCAPed_pc_vals_abs.csv'
        out_labels = TON_tools.make_binary_levels_from_behav_file(
            labels, subject_list, happy_sad_label, in_csv, thres=.5,
            runs_per_subj=None)
        keep_idx = ~np.isnan(out_labels)
        if covariate_detrend_params:
            covariate_detrend_params['idx_classified'] = np.where(keep_idx)[0]
        # Remove unused runs/subjects
        data, labels = data[keep_idx, :], out_labels[keep_idx]
        logging.info("Data was filtered: shape is {}".format(data.shape))

    else:
        if do_preHD_age_sex_detrend:
            in_csv = '/data2/polo/half_baked_data/slopes/abs/raw_slopes_ok_subjs_abs.csv'
            healthy_ctrl_vars = ['age', 'sex']
            preHD_ctrl_vars = []
            covariate_detrend_params = {'data': data.copy(),
                                'preHD_healthy_labels': labels,
                                'subject_list': subject_list,
                                'in_csv': in_csv,
                                'healthy_ctrl_vars': healthy_ctrl_vars,
                                'preHD_ctrl_vars': preHD_ctrl_vars,
                                'runs_per_subj': runs_per_subj,
                                'add_intercept': True,
                                'idx_classified': np.arange(data.shape[0])}
        keep_idx = ~np.isnan(labels)
    '''
    Make an array with subject name for each run
    '''
    subj_per_run = np.empty((data.shape[0],), dtype='<U10')
    for k in range(runs_per_subj):
        subj_per_run[k::runs_per_subj] = subject_list[keep_idx[::runs_per_subj]]

    if ksplit == 'LOO':
        ksplit = data.shape[0] / runs_per_subj
    elif ksplit == 'site':
        site_df = pd.read_csv('/data2/polo/half_baked_data/TON_siteid.csv').set_index('subjid')
        site_labels = site_df.loc[subj_per_run].values.flatten()
        ksplit = np.unique(site_labels).shape[0]

    logging.info("Folds are {}".format(ksplit))
    logging.info("Test subjects per split:{}".format(data.shape[0] /
                                                     (ksplit*runs_per_subj)))
    try:
        input("Press Enter to continue...")
    except SyntaxError:
        pass



    '''
    CLASSIFIER AND PARAM DICTS
    '''
    classifiers, params = ps.make_classifiers(NAMES)  # data.shape, ksplit)

    '''
    Make subject-wise folds
    '''
    subject_labels = np.zeros(data.shape[0])
    assert np.remainder(data.shape[0], runs_per_subj) == 0
    for it in range(runs_per_subj):
        subject_labels[it::runs_per_subj] = np.arange(data.shape[0] /
                                                      runs_per_subj)
    group_kfold = GroupKFold(n_splits=ksplit)
    if 'site_labels' in locals():
        split_gen = list(group_kfold.split(data, labels, site_labels))
    else:
        split_gen = list(group_kfold.split(data, labels, subj_per_run))

    # kf = KFold(labels.shape[0], n_folds=ksplit)
    # Extract the training and testing indices from the k-fold object,
    # which stores fold pairs of indices.
    fold_pairs = [(tr, ts) for (tr, ts) in split_gen]
    assert len(fold_pairs) == ksplit

    '''
    RANK VARIABLES FOR EACH FOLD (does ttest unless otherwise specified)
    '''

    rank_per_fold = ps.get_rank_per_fold(data, labels, fold_pairs,
                                         save_path=out_dir, parallel=True,
                                         covariate_detrend_params=covariate_detrend_params)

    '''
    COMPUTE SCORES
    '''

    score = {}
    dscore = []
    totalErrs = []
    if compute_results:
        for name in NAMES:
            mdl = classifiers[name]
            param = params[name]
            # get_score runs the classifier on each fold,
            # each subset of selected top variables and does a grid search for
            # classifier-specific parameters (selects the best)
            clf, allConfMats, allTotalErrs,\
                allFittedClassifiers, predictions, cont_predictions = \
                ps.get_score(data, labels, fold_pairs, name, mdl, param,
                             numTopVars=numTopVars,
                             rank_per_fold=rank_per_fold, parallel=True,
                             rand_iter=-1,
                             covariate_detrend_params=covariate_detrend_params)

            # save classifier object and results to file
            ps.save_classifier_results(name, out_dir, allConfMats,
                                       allTotalErrs)
            ps.save_classifier_object(clf, allFittedClassifiers, name, out_dir)
            ps.save_classifier_predictions_per_sample(name, out_dir,
                                                      predictions, cont_predictions,
                                                      fold_pairs,
                                                      labels, subj_per_run)
            # Append classifier results to list of all results
            dscore.append(allConfMats)
            totalErrs.append(allTotalErrs)

        '''
        First do some saving of total results
        '''
        ps.save_combined_results(NAMES, dscore, totalErrs,
                                 numTopVars, out_dir, filename_base)

    ps.plot_errors(NAMES, numTopVars, dscore, totalErrs,
                   filename_base, out_dir, compute_results, format_used='pdf')

    for name in NAMES:
        ps.plot_bootstrap_confusion_from_prediction_csv(name, out_dir,
                                                        n_resamples=int(1e5),
                                                        do_chance=True,
                                                        alpha=5,
                                                        chance_alpha=0,
                                                        save_fig=True,
                                                        format_used='pdf',
                                                        fig_title='full',
                                                        replacement=False)
        ps.plot_bootstrap_auc_from_prediction_csv(name, out_dir,
                                                  n_resamples=int(1e5),
                                                  do_chance=True, alpha=5,
                                                  save_fig=True,
                                                  format_used='pdf',
                                                  fig_title='auc',
                                                  replacement=False)

    logging.shutdown()
