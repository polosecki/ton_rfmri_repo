# -*- coding: utf-8 -*-
"""
    Resting State fMRI (rsfMRI) preprocessing
    Created on Thu Feb 11 11:37:50 2016
@author: pipolose

Based off:
http://nipy.org/nipype/0.9.2/users/examples/rsfmri_fsl_compcorr.html
(this is the one that comes in nipype.workflows.rsfmri.fsl.resting.create_resting_preproc)
"""

import resting_wf as rsfmri
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import IdentityInterface, Function
import nipype.interfaces.io as nio
import explore.helper_functions as hf
import numpy as np



'''
INPUT PARAMETERS
'''
TR = 3.0
local_base_dir = '/data1/pipelines'
cooked_dir = '/data1/cooked/TONf'
visit_list = [1,2 ,3]
use_multicore = 24


'''
BASIC DATA LOADING
'''
# Create subject_list and such using explore.helper_functions:
TRACK_ON_data_dir = ('/data1/chdi_disks/Disk2/IBM_SOW3-part2/'
                     'Updated TRACK phenotypic dataset')

subject_list, subjects_dict, gton, mri_TON, visit_dict = \
    hf.make_Track_ON_subjects(TRACK_ON_data_dir)

subject_list = np.array(subject_list)

# Specify a subset of subjects through local indexing variable or use all:
if 'used_subj_idx' in locals():
    used_subj = subject_list[used_subj_idx]
else:
    used_subj = subject_list


''' IdentityInterface: The beginning of all workflows: '''
infosource = pe.Node(IdentityInterface(fields=['subject_id',
                                               'visit',
                                               'subtype']),
                     name="infosource")
infosource.iterables = [('subject_id', used_subj),
                        ('visit', visit_list)]
infosource.inputs.subtype = 'Resting_State'

''' Datagrabber node: '''
datag = pe.Node(nio.DataGrabber(infields=['subject_id',
                                          'visit',
                                          'subtype']),
                name='datag')
datag.inputs.base_directory = cooked_dir
datag.inputs.template = ('{0}/visit_{1}/{2}/'
                         '{0}_visit_{1}_{2}_S??????.nii.gz')
datag.inputs.sort_filelist = True

'''Load Resting State WorkFlow: '''
restingflow = rsfmri.create_resting_preproc(name='restingflow')
restingflow.inputs.inputspec.num_noise_components = 6
# In FSL Sigmas are in volumes, not seconds. Therefore this is the TR-dependent
# sigma in volumes if one wants a TR-indepdendent sigma in seconds:
restingflow.inputs.inputspec.highpass_sigma = 200/(2*TR) # 0.01 Hz
restingflow.inputs.inputspec.lowpass_sigma = 12.5/(2*TR) # 0.16 Hz

'''Define the datasink: '''
datasink = pe.Node(nio.DataSink(base_directory=cooked_dir),
                   name="datasink")
datasink.inputs.parameterization = False

''' Set up function for replacing volume names: '''


def get_substitutions(subject_id, visit, subtype, input_volume):
    '''Replace output names of files with more meaningful ones
    '''
    import re
    from os.path import join as opj
    scan_id = re.search('_S[0-9]+', input_volume).group()[1:]
    template = opj('visit_{}'.format(visit), subtype,
                   '{0}_visit_{1}_{2}_{3}'.format(subject_id, visit,
                                                  subtype, scan_id))
    mcf_template = '_'.join([subject_id, 'visit', str(visit),
                             subtype, scan_id, 'st_start_excised_mcf'])
    mcf_target = opj('visit_{}'.format(visit), subtype, mcf_template)

    plot_source = 'plot.' + template + '_realigned'
    intensity_source = 'global_intensity.' + template + '_realigned'
    substitutions = [('detrend_regfilt_filt',
                      template + '_corrcomped'),
                     ('mask.vol0000_warp_merged_smooth.',
                      template + '_fmask.'),
                     ('mask.vol0000_warp_merged_smooth_maths',
                      template + '_noisyvoxels'),
                     ('vol0000_warp_merged.',
                      template + '_realigned.'),
                     (mcf_template, mcf_target),
                     ('art.vol0000_warp_merged_outliers',
                      template + '_outliers'),
                     ('_subject_id_', ''),
                     (plot_source, template + '_mcplot'),
                     (intensity_source, template + '_intensity')
                     ]
    return substitutions


name_substituter = pe.Node(Function(input_names=['subject_id',
                                                 'visit',
                                                 'subtype',
                                                 'input_volume'],
                                    output_names=['substitutions'],
                                    function=get_substitutions),
                           name='name_substituter')



def pick_rsvol_from_list(vol_abs_names):
    '''
    set up function for picking the desired rs-fMRI out of possibly more
    than one in a given subject and visit:
    '''
    import nibabel as nib
    import numpy as np
    import re
    if type(vol_abs_names) is str:
        used_vol = vol_abs_names
    else:
        n_frames = np.zeros(len(vol_abs_names), dtype=int)
        for used_idx, used_vol in enumerate(vol_abs_names):
            img = nib.load(used_vol)
            n_frames[used_idx] = img.header.get_data_shape()[3]  # 4th dim

        mx = max(n_frames)
        mx_idx = [idx for idx, val in enumerate(n_frames) if val == mx]

        if len(mx_idx) == 1:
            used_vol = vol_abs_names[mx_idx[0]]
        else:
            # If more than one identical scan was acquired,
            # pick the last one:
            canditate_vols = vol_abs_names[mx_idx]
            scan_ids = [int(re.search('S[0-9]+', c_vol).group()[1:])
                        for c_vol in canditate_vols]
            used_vol = canditate_vols[np.argmax(scan_ids)]

    return used_vol

rsvol_picker = pe.Node(Function(input_names=['vol_abs_names'],
                                output_names=['used_vol'],
                                function=pick_rsvol_from_list),
                       name='rsvol_picker')


'''
DEFINE RS WORKFLOW
'''

resting = pe.Workflow(name="resting")
resting.base_dir = local_base_dir
resting.connect([(infosource, datag, [('subject_id', 'subject_id'),
                                      ('visit', 'visit'),
                                      ('subtype', 'subtype')]),
                 (datag, rsvol_picker, [('outfiles', 'vol_abs_names')]),
                 (rsvol_picker, restingflow, [('used_vol', 'inputspec.func')]),
                 (rsvol_picker, name_substituter, [('used_vol',
                                                    'input_volume')]),
                 (infosource, name_substituter, [('subject_id', 'subject_id'),
                                                 ('visit', 'visit'),
                                                 ('subtype', 'subtype')]),
                 (name_substituter, datasink, [('substitutions',
                                                'substitutions')]),
                 (infosource, datasink, [('subject_id', 'container')]),
                 (restingflow, datasink, [('outputspec.noise_mask_file',
                                           '@noisefile'),
                                          ('outputspec.filtered_file',
                                           '@filteredfile'),
                                          ('outputspec.motion_rms_files',
                                           '@motion_rms_files'),
                                          ('outputspec.motion_par_file',
                                           '@motion_par_file'),
                                          ('outputspec.realigned_file',
                                           '@realigned_file'),
                                          ('outputspec.outlier_files',
                                           '@outlier_files'),
                                          ('outputspec.intensity_files',
                                           '@intensity_files'),
                                          ('outputspec.outlier_plots',
                                           '@outlier_plots'),
                                          ('outputspec.mask_file',
                                           '@mask_file'),
                                          ])
                 ])

if __name__ == '__main__':
    resting.config['execution']['remove_unnecessary_outputs'] = 'true'
    resting.write_graph(graph2use='flat', format='svg')
    resting.write_graph(graph2use='colored', format='svg')
    if use_multicore > 1:
        resting.run('MultiProc', plugin_args={'n_procs': use_multicore})
    else:
        resting.run()
