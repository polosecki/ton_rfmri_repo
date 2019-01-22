#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:57:57 2017

@author: pipolose
"""

from plotting_tools.plot_overlay import plot_n_save_3plane
import matplotlib.pyplot as plt
from matplotlib import cm


in_dict = {}
in_dict['do_save'] = True # True #
in_dict['formats_used'] = ['pdf', 'png']
in_dict['out_dir'] = '/data1/polo/figures'

pv = []
'''
Plot median log degree wilcoxon
'''
in_dict['in_dir'] = ('/data1/polo/code/MLtool/TON_resting_classification/2016_Q4/paper_volumes/TON_Resting_State_bct_degrees_log_norm_mean_happy_sad')
in_dict['in_fn'] = 'bct_degrees_log_PC_3_pmap.nii.gz'
in_dict['cut_coords'] = [-22, 4, 36]#[20, 3, -5]
in_dict['area'] = 'L-internal_capsule'
in_dict['threshold']=3
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))


'''
Plot mean logistic weights
'''
in_dict['in_dir'] = ('/data1/chdi_results/polo/polyML/results/degree/bct/thres07/non_smooth/happy_sad/PC_3')
in_dict['in_fn'] = 'SAGA_log_elastic_weight_nfolds_46.nii.gz'
in_dict['cut_coords'] = [-22, 4, 36]#[-18, 15, 0]
in_dict['area'] = 'L-internal_capsule'
in_dict['threshold'] = 2.5
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))
