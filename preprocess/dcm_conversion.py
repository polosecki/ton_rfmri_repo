# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:20:08 2016

@author: pipolose
"""

from __future__ import division
import numpy as np
import explore.helper_functions as hf
import conversion_tools as ct

import nipype.pipeline.engine as pe
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import IdentityInterface

'''
INPUT PARAMETERS
'''
sink_bdir = '/data1/cooked/TONf/'
wf_bdir = '/data1/cooked/pipelines/'
used_subj_idx = range(388)
visit_list = [1, 2, 3]
subtype = 'Resting State'  # 'T1W'
use_multicore = False

'''
BASIC DATA LOADING
'''
# Create subject_list and such using explore.helper_functions:
TRACK_ON_data_dir = ('/data1/chdi_disks/Disk2/IBM SOW3-part2/'
                      'Updated TRACK phenotypic dataset')

subject_list, subjects_dict, gton, mri_TON, visit_dict = \
    hf.make_Track_ON_subjects(TRACK_ON_data_dir)

subject_list = np.array(subject_list)

# Specify a subset of subjects through local indexing variable or use all:
if 'used_subj_idx' in locals():
    used_subj = subject_list[used_subj_idx]
else:
    used_subj = subject_list

'''
Conversion Node and Workflow definitions
---------------------
'''

# Define object generating node:
sub_obj_prov_node = ct.sub_obj_prov_node
sub_obj_prov_node.inputs.subj_dict = subjects_dict

# Conversion node
conversion_wf = ct.conversion_wf

'''
Workflow definition
-------------------
'''

# Define Workflow
format_conversion = pe.Workflow(name='format_conversion')
format_conversion.base_dir = wf_bdir

# Define the identity interface
infosource = pe.Node(IdentityInterface(fields=['subject_id',
                                               'visit',
                                               'subtype']),
                     name="infosource")
infosource.iterables = [('subject_id', used_subj),
                        ('visit', visit_list)]
infosource.inputs.subtype = subtype

# Rename Node:
ren_node = ct.ren_node

# Define the datasink
datasink = pe.Node(DataSink(base_directory=sink_bdir),
                   name="datasink")
substitutions = [('reoriented', '')]
datasink.inputs.substitutions = substitutions
datasink.inputs.regexp_substitutions = [('_subject_id_R[0-9X]+_', ''),
                                        ('_reorient[0-9]',
                                         infosource.inputs.
                                         subtype.replace(' ', '_'))]

format_conversion.connect([(infosource, sub_obj_prov_node, [('subject_id', 'subj_name')]),
                           (infosource, conversion_wf, [('visit', 'dcm_prov_node.visit'),
                                                        ('subtype', 'dcm_prov_node.subtype')]),
                           (sub_obj_prov_node, conversion_wf, [('subj_obj', 'dcm_prov_node.subj_obj')]),
                           (infosource, ren_node, [('subject_id', 'subject_id'),
                                                   ('visit', 'visit'),
                                                   ('subtype', 'subtype')]),
                           (conversion_wf, ren_node, [('reorient.out_file', 'input_vol')]),
                           (sub_obj_prov_node, ren_node, [('subj_obj', 'subj_obj')]),
                           (infosource, datasink, [('subject_id', 'container')]),
                           (ren_node, datasink, [('out_file', 'reoriented')]),
                           ])

format_conversion.write_graph(graph2use='flat', format='svg')
format_conversion.write_graph(graph2use='colored', format='svg')

if use_multicore:
    format_conversion.run('MultiProc', plugin_args={'n_procs': 48})
else:
    format_conversion.run()
