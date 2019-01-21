# -*- coding: utf-8 -*-
"""
This function takes a data file and removes correlations
Created on Fri Oct 14 18:53:40 2016

@author: pipolose
"""
import numpy as np
import scipy.io
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import logging
import sys
if sys.version_info[0] < 3:
    import bootstrap_tools as bt
else:
    from . import bootstrap_tools as bt


def load_data(data_file, data_pattern='*.mat'):
    """
    Loads the data from multiple sources if provided.

    Parameters
    ----------
    data_file: str
    data_pattern: str

    Returns
    -------
    data: array_like
    """

    dataMat = scipy.io.loadmat(data_file, mat_dtype=True)
    data = dataMat['data']

    logging.info("Data loading complete. Shape is %r" % (data.shape,))
    return data[:, :-1], data[:, -1], data_file


def detrend_data(data, preHD_healthy_labels, subject_list,
                 in_csv, healthy_ctrl_vars=[],
                 preHD_ctrl_vars=[], runs_per_subj=None, add_intercept=True,
                 healthy_label=-1, **kwargs):
    '''
    Inputs: data array (no labels in it),list with pre_HD/healthy labels,
            ordered list of subject names, input csv file with control
            variables (as made by make_behavioral_variables)
    Optional: number of runs pers subject, otherwise this is inferred from data
    '''

    if not runs_per_subj:
        runs_per_subj = np.floor(data.shape[0]/len(subject_list)).astype(int)
    assert np.remainder(data.shape[0], len(subject_list)) == 0

    # Add intercept
    if ('intercept' not in healthy_ctrl_vars) and add_intercept\
            and healthy_ctrl_vars:
        healthy_ctrl_vars.append('intercept')
    if ('intercept' not in preHD_ctrl_vars) and add_intercept\
            and preHD_ctrl_vars:
        preHD_ctrl_vars.append('intercept')

    # Load regressors
    temp = pd.read_csv(in_csv, dtype={'subjid': str})
    temp.index = temp.index.astype(str)

    cov_df = temp.groupby('subjid').mean().loc[subject_list].copy()
    cov_df['intercept'] = 1
    assert cov_df.index.name == 'subjid'

    if healthy_ctrl_vars:
        cov_per_subj = cov_df[healthy_ctrl_vars].values
        X_hp = np.empty((data.shape[0], len(healthy_ctrl_vars)))
        # Repeat subject regressors for each run
        for i in range(runs_per_subj):
            X_hp[i::runs_per_subj, :] = cov_per_subj

        # Compute age/sex and intercept coefficient from healthy subjects
        X_h = X_hp[preHD_healthy_labels == healthy_label, :]
        y_h = data[preHD_healthy_labels == healthy_label, :]
        coeff_h = np.linalg.lstsq(X_h, y_h)[0]
        assert not np.any(np.isnan(coeff_h))
        data_deaged = data - np.dot(X_hp, coeff_h)
    else:
        data_deaged = data
        coeff_h = np.array([])

    # Compute CAP/CAG and intercept coefficient from preHD subjects
    if preHD_ctrl_vars:
        cov_per_subj = cov_df[preHD_ctrl_vars].values
        X_hp = np.empty((data.shape[0], len(preHD_ctrl_vars)))
        # Repeat subject regressors for each run
        for i in range(runs_per_subj):
            X_hp[i::runs_per_subj, :] = cov_per_subj
        X_p = X_hp[preHD_healthy_labels == -healthy_label, :]
        y_p = data_deaged[preHD_healthy_labels == -healthy_label, :]
        coeff_p = np.linalg.lstsq(X_p, y_p)[0]
        assert not np.any(np.isnan(coeff_p))
        data_decaped = data_deaged - np.dot(X_hp, coeff_p)
    else:
        data_decaped = data_deaged
        coeff_p = np.array([])
    return data_decaped, coeff_h, coeff_p


def make_binary_levels_from_behav_file(preHD_healthy_labels, subject_list,
                                       used_behav_column, in_csv, thres=.5,
                                       runs_per_subj=None):

    '''
    Takes in a csv file, loads the used_behav_column, set a threshold on the
    behavior, assing labels to the subjects in sbuject_list
    '''

    if not runs_per_subj:
        runs_per_subj = np.floor(preHD_healthy_labels.shape[0] /
                                 len(subject_list)).astype(int)
    assert np.remainder(preHD_healthy_labels.shape[0], len(subject_list)) == 0

    behav = pd.read_csv(in_csv, dtype={'subjid': str}, index_col='subjid').\
        loc[subject_list][used_behav_column].copy()

    subject_labels = pd.Series(preHD_healthy_labels[::runs_per_subj],
                               subject_list)

    behav_label = pd.Series(np.nan, subject_labels.index)
    happy_subjs = (behav > thres) & (subject_labels == 1)
    sad_subjs = (behav < -thres) & (subject_labels == 1)
    behav_label[happy_subjs] = -1
    behav_label[sad_subjs] = 1

    label_vals = behav_label.values

    out_labels = np.zeros(runs_per_subj*label_vals.shape[0])
    # Repeat subject regressors for each run
    for i in range(runs_per_subj):
        out_labels[i::runs_per_subj] = label_vals
    return out_labels


class TON_feature_detrender(BaseEstimator, TransformerMixin):
    '''
    This class is meant to implement detrending in a way that allows
    for cross-validation using separate fit and transform methods in the
    manner of sklearn

    Initial arugments:
    ------------------
    data: feature data matrix with all subjects
    preHD_healthy_labels: array with binary +1,-1 disease/control labals
    subject_list: list or subject names, ordered as in data matrix
    in_csv: absolute path of csv file with regreesion covariates,
            as generated by make_behavioral_variables
    runs_per_subject: Optional. It is assumed all subjs. have equal number of runs
    add_intercept: (Default: True) If intercept term is to be used when detrending
    idx_classified: indicates the indices in data matrix that are part of the CV classification
    (e.g., this includes only a subset of disease subject indices to which the train and
    test indices used in methods will make reference to. This idx_classified parameter
    is for translating)
    healthy_label: The value used in preHD_healthy_labels to indicate healthy
    subjects
    rankedVars: Rank of features (optional)
    numVars: Number of top k ranked features to use (only if rank is provided)
    '''

    def __init__(self, data, preHD_healthy_labels, subject_list,
                 in_csv, idx_classified, healthy_ctrl_vars=[],
                 preHD_ctrl_vars=[], runs_per_subj=None, add_intercept=True,
                 healthy_label=-1, rankedVars=None, numVars=None):

        if not runs_per_subj:
            runs_per_subj = np.floor(data.shape[0]/len(subject_list)).\
                astype(int)
        assert np.remainder(data.shape[0], len(subject_list)) == 0

        self.data = data  # Input data matrix containing both test and train
        self.subject_list = subject_list  # List of subject names
        self.in_csv = in_csv  # path to input data frame
        self.preHD_healthy_labels = preHD_healthy_labels
        self.healthy_ctrl_vars = healthy_ctrl_vars
        self.preHD_ctrl_vars = preHD_ctrl_vars
        self.runs_per_subj = runs_per_subj
        self.add_intercept = add_intercept
        self.idx_classified = idx_classified
        self.healthy_label = healthy_label
        self.rankedVars = rankedVars
        self.numVars = numVars

    def fit(self, train_idx, test_idx=np.array([])):
        '''
        Fits detrending weights for each feature.
        Inputs:
        ------
        train_idx: Array, contains the RUN (not subject) INDICES to be used
        '''

        run_idx = np.union1d(self.idx_classified[train_idx],
                             np.where(self.preHD_healthy_labels ==
                                      self.healthy_label)[0])
        if test_idx.shape[0] > 0:
            run_idx = np.setdiff1d(run_idx, self.idx_classified[test_idx])
        subj_train_idx = run_idx[np.mod(run_idx, self.runs_per_subj
                                        ) == 0] / self.runs_per_subj

        fit_dict = self.get_params()
        if self.rankedVars is not None:
            used_cols = np.squeeze(self.rankedVars)[:self.numVars]
        else:
            used_cols = np.arange(self.data.shape[1])
        fit_dict['data'] = self.data[np.ix_(run_idx, used_cols)]
        fit_dict['subject_list'] = self.subject_list[subj_train_idx]
        fit_dict['preHD_healthy_labels'] = self.preHD_healthy_labels[run_idx]
        self.beta_h, self.beta_p = detrend_data(**fit_dict)[1:]

    def transform(self, test_idx):
        '''
        Applies detrending to test set, from weights learned before using
        fit method.

        Inputs:
        ------
        test_idx: Array, contains the RUN (not subject) INDICES to be detrended
        Output:
        decaped_data
        '''
        run_idx = self.idx_classified[test_idx]
        # Compute subject indices from run indices
        subj_test_idx = run_idx[np.mod(run_idx, self.runs_per_subj) == 0] /\
            self.runs_per_subj

        temp = pd.read_csv(self.in_csv, dtype={'subjid': str})
        temp.index = temp.index.astype(str)
        cov_df = temp.groupby('subjid').mean().\
            loc[self.subject_list[subj_test_idx]].copy()
        if self.add_intercept:
            cov_df['intercept'] = 1
        if self.rankedVars is not None:
            used_cols = np.squeeze(self.rankedVars)[:self.numVars]
        else:
            used_cols = np.arange(self.data.shape[1])
        in_data = self.data[np.ix_(run_idx, used_cols)]

        # First, remove trends expected in healthy brains
        if self.beta_h.shape != (0,):
            cov_per_subj = cov_df[self.healthy_ctrl_vars].values
            X_hp = np.empty((in_data.shape[0], self.beta_h.shape[0]))
            # Repeat subject regressors for each run
            for i in range(self.runs_per_subj):
                X_hp[i::self.runs_per_subj, :] = cov_per_subj
            data_deaged = in_data - np.dot(X_hp, self.beta_h)
        else:
            data_deaged = in_data

        # Now, remove trends expected in non-healthy brains
        if self.beta_p.shape != (0,):
            cov_per_subj = cov_df[self.preHD_ctrl_vars].values
            X_p = np.empty((in_data.shape[0], self.beta_p.shape[0]))
            # Repeat subject regressors for each run
            for i in range(self.runs_per_subj):
                X_p[i::self.runs_per_subj, :] = cov_per_subj
            data_decaped = data_deaged - np.dot(X_p, self.beta_p)
        else:
            data_decaped = data_deaged
        return data_decaped


def provide_pheno_distributions_from_df(results_df, pheno_vars, n_resamples,
                                        runs_per_subj=0, shuffle=False,
                                        replacement=True, stat_fn='median'):
    if not shuffle and not replacement:
        replacement = True
        warnings.warn('Replacement was automatically set to True because'
                      'shuffle is False')

    if runs_per_subj == 0:
        non_subj_cols = [col for col in results_df.columns if col != 'subjid']
        run_counts = results_df.groupby('subjid')[non_subj_cols[0]]\
            .count().unique()
        runs_per_subj = run_counts[0]
        assert run_counts.shape == (1,)
    assert np.remainder(results_df.shape[0], runs_per_subj) == 0
    n_samples = results_df.shape[0]
    perf = []
    for t in range(n_resamples):
        ix_used = bt.provide_rand_idx(n_samples,
                                      runs_per_subj,
                                      replacement=replacement)
        resampled_table = results_df.iloc[ix_used]
        this_result = resampled_table[pheno_vars]
        # This dict key is the shuffle input logical value
        group_column = {True: results_df['prediction'].values,
                        False: resampled_table['prediction'].values}
        if stat_fn == 'median':
            it_out = this_result.groupby(group_column[shuffle]).median()
        elif stat_fn == 'mean':
            it_out = this_result.groupby(group_column[shuffle]).mean()
        else:
            raise NotImplementedError(('Stat function {} hasn\'t'
                                       ' been incorporated into this code'
                                       ).format(stat_fn))
        perf.append((t, it_out.diff().iloc[-1]))
    out_dist = pd.DataFrame.from_items(perf).T
    return out_dist


def provide_subgroup_preserving_ranking_correlation_dist(rank_df,
                                                         subgroup_names,
                                                         runs_per_subj,
                                                         n_iter,
                                                         shuffle=False,
                                                         replacement=True):
    out_corrs = np.zeros((n_iter,))
    for it in range(n_iter):
        new_behav_ranks = []
        new_svm_ranks = []
        for sg in subgroup_names:
            sub_rank = rank_df[rank_df['subgroup'] == sg]
            ix_used = bt.provide_rand_idx(sub_rank.shape[0],
                                          runs_per_subj,
                                          replacement=replacement)
            resampled_table = sub_rank.iloc[ix_used]
            behav_column = {True: sub_rank['behav_rank'].values,
                            False: resampled_table['behav_rank'].values}
            new_behav_ranks.append(behav_column[shuffle])
            new_svm_ranks.append(resampled_table['weight_rank'].values)
        x = np.concatenate(new_svm_ranks)
        y = np.concatenate(new_behav_ranks)
        out_corrs[it] = scipy.stats.spearmanr(x, y)[0]
    return out_corrs
