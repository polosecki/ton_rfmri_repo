#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:57:57 2017

@author: pipolose
"""

from  polyML.plot_overlay import plot_n_save_3plane
import matplotlib.pyplot as plt
from matplotlib import cm


in_dict = {}
in_dict['do_save'] = True # True #
in_dict['formats_used'] = ['pdf', 'png']
in_dict['out_dir'] = '/data2/polo/figures'

pv = []
'''
Plot median log degree wilcoxon
'''
in_dict['in_dir'] = ('/data2/polo/code/MLtool/TON_resting_classification/2018_Q2/TON_log_deg_maps_local_gm_corrected_combat_smooth_happy_sad')
in_dict['in_fn'] = 'bct_degrees_log_sdmt_pmap_FDRmasked.nii.gz'
in_dict['cut_coords'] = [-13, 17, -8]#[20, 3, -5]
in_dict['area'] = 'L-accumbens'
in_dict['threshold']=0
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))


in_dict['in_dir'] = ('/data2/polo/code/MLtool/TON_resting_classification/2018_Q2/TON_log_deg_maps_local_gm_corrected_combat_smooth_happy_sad')
in_dict['in_fn'] = 'bct_degrees_log_sdmt_pmap_FDRmasked.nii.gz'
in_dict['cut_coords'] = [29, -20, 40]#[20, 3, -5]
in_dict['area'] = 'R-WM'
in_dict['threshold']=0
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))


in_dict['in_dir'] = ('/data2/polo/code/MLtool/TON_resting_classification/2018_Q2/TON_log_deg_maps_local_gm_corrected_combat_smooth_happy_sad')
in_dict['in_fn'] = 'bct_degrees_log_sdmt_pmap_FDRmasked.nii.gz'
in_dict['cut_coords'] = [-59, -40, -4]#[20, 3, -5]
in_dict['area'] = 'L-STS'
in_dict['threshold']=0
in_dict['symmetric_cbar'] = True
pv.append(plot_n_save_3plane(in_dict))


