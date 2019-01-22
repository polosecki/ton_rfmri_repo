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
in_dict['in_fn'] = 'bct_degrees_log_sdmt_pmap_FDRmasked.nii.gz'
in_dict['cut_coords'] = [-13, 17, -8]#[20, 3, -5]
in_dict['area'] = 'L-accumbens'
in_dict['threshold']=0
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))


in_dict['in_dir'] = ('/data1/polo/code/MLtool/TON_resting_classification/2016_Q4/paper_volumes/TON_Resting_State_bct_degrees_log_norm_mean_happy_sad')
in_dict['in_fn'] = 'bct_degrees_log_sdmt_pmap_FDRmasked.nii.gz'
in_dict['cut_coords'] = [29, -20, 40]#[20, 3, -5]
in_dict['area'] = 'R-WM'
in_dict['threshold']=0
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))


in_dict['in_dir'] = ('/data1/polo/code/MLtool/TON_resting_classification/2016_Q4/paper_volumes/TON_Resting_State_bct_degrees_log_norm_mean_happy_sad')
in_dict['in_fn'] = 'bct_degrees_log_sdmt_pmap_FDRmasked.nii.gz'
in_dict['cut_coords'] = [-59, -40, -4]#[20, 3, -5]
in_dict['area'] = 'L-STS'
in_dict['threshold']=0
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))

in_dict['in_dir'] = ('/data1/polo/code/MLtool/TON_resting_classification/2016_Q4/paper_volumes/TON_Resting_State_bct_degrees_log_norm_mean_happy_sad')
in_dict['in_fn'] = 'bct_degrees_log_grip_var_pmap_FDRmasked.nii.gz'
in_dict['cut_coords'] = [-32, -20, 30]#[20, 3, -5]
in_dict['area'] = 'L-WM'
in_dict['threshold']=0
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))
