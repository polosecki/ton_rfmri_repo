#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:19:48 2018

@author: pipolose
"""

import numpy as np
import pandas as pd
from combat import combat as cb


def combat_harmonize_site(data, subject_list, site_df, covariates_df=None):
    '''
    Uses the combat approach to harmonized data

    Parameters
    ----------
    data: array (n_samples x n_features)
    subject_list: list of subject id's (len: n_samples)
    site_df: dataframe index must be called subjid and contain all subjids in
        subject_list. Column must be called siteid
    covariates_df: dataframe, optional
        Contains covariates of interest to keep

    Returns
    -------
    harmonized_data: array (n_samples x n_features)
    '''
    batch = site_df.loc[subject_list]['siteid']

    feat_df = pd.DataFrame(data.T)
    feat_df.columns = subject_list
    harmonized_data = cb.combat(feat_df, batch, model=covariates_df).values.T
    if np.isnan(data).sum() > np.isnan(harmonized_data).sum():
        Warning('New nan data have been introduced by the harmonization '
                'method')
    return harmonized_data
