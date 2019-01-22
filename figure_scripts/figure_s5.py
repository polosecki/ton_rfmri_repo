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
in_dict['out_dir'] = '/data2/polo/figures'

pv = []

'''
Plot mean logistic weights
'''
in_dict['in_dir'] = ('/data1/chdi_results/polo/polyML/results/TON_rsfmri_scirep/degree/thres07/non_smooth_no_norm/local_gm_corrected_not_CV/combat/happy_sad/PC_0')
in_dict['in_fn'] = 'SAGA_log_elastic_weight_nfolds_51.nii.gz'
in_dict['cut_coords'] = [-55, -40, 1]
in_dict['area'] = 'L-STS'
in_dict['threshold'] = 2.5
in_dict['symmetric_cbar'] = True
in_dict['draw_cross'] = {'linewidth': .5}
pv.append(plot_n_save_3plane(in_dict))
