# -*- coding: utf-8 -*-
"""
Script to do spatial registration of both structural and functional MRI volumes
to MNI space. The generated images have the same resolution as the originally
acquired ones. It makes use of 2 workflows: 'registration_func_wf.py' and
'registration_struct_wf.py'.

Created on Fri Feb 19 02:52:17 2016

@author: ecastrow
"""

import registration_struct_wf as regstruct
import registration_func_wf as regfunc
import nipype.pipeline.engine as pe
from nipype.interfaces.utility import IdentityInterface, Function
import nipype.interfaces.io as nio
import explore.helper_functions as hf
import numpy as np


"""
INPUT PARAMETERS
"""
grab_bdir = '/data1/cooked/TONf/'
sink_bdir = '/data1/cooked/TONf/'
wf_bdir = '/data1/pipelines/'
frac_thresh = 0.4
visit_list = [1, 2, 3]
sMRI_stype = 'T1W'
fMRI_stype = 'Resting_State'
use_multicore = 4          # either False or number of cores

"""
BASIC DATA LOADING
"""
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
    used_subj = subject_list#  used_subj = ['R181040562'] #subject_list  # ['R013297352']

used_subj = subject_list[used_subj_idx]

""" IdentityInterface: The beginning of all workflows """
infosource = pe.Node(IdentityInterface(fields=['subject_id',
                                               'visit',
                                               'sMRI_stype',
                                               'fMRI_stype',
                                               'frac_thresh']),
                     name="infosource")
infosource.iterables = [('subject_id', used_subj),
                        ('visit', visit_list)]
infosource.inputs.sMRI_stype = sMRI_stype
infosource.inputs.fMRI_stype = fMRI_stype
infosource.inputs.frac_thresh = frac_thresh

""" Datagrabber node """
datag = pe.Node(nio.DataGrabber(infields=['subject_id', 'visit', 'sMRI_stype',
                                          'fMRI_stype'],
                                outfields=['struct', 'func_r', 'func_c']),
                name='datag')
datag.inputs.base_directory = grab_bdir
datag.inputs.template = '*'
datag.inputs.sort_filelist = True
datag.inputs.field_template = dict(struct=('{0}/visit_{1}/{2}/'
                                           '{0}_visit_{1}_{2}_S??????'
                                           '.nii.gz'),
                                   func_r=('{0}/visit_{1}/{3}/'
                                           '{0}_visit_{1}_{3}_S??????'
                                           '_realigned.nii.gz'),
                                   func_c=('{0}/visit_{1}/{3}/'
                                           '{0}_visit_{1}_{3}_S??????'
                                           '_corrcomped.nii.gz'))

""" Define the datasink """
datasink = pe.Node(nio.DataSink(base_directory=sink_bdir),
                   name="datasink")
datasink.inputs.parameterization = False


def get_substitutions(subject_id, visit, sMRI_stype, fMRI_stype, svol, fvol):
    """
    Replace output names of files with more meaningful ones
    """

    import re
    from os.path import join as opj

    scan_struct = re.search('_S[0-9]+', svol).group()[1:]
    scan_func = re.search('_S[0-9]+', fvol).group()[1:]
    func_root = '{0}_visit_{1}_{2}_{3}'.format(subject_id, visit,
                                               fMRI_stype, scan_func)
    struct_root = '{0}_visit_{1}_{2}_{3}'.format(subject_id, visit,
                                                 sMRI_stype, scan_struct)
    new_func_root = opj('visit_{}'.format(visit), fMRI_stype, func_root)
    new_struct_root = opj('visit_{}'.format(visit), sMRI_stype, struct_root)

    substitutions = [(struct_root + '_brain_out.', new_struct_root +
                     '_brain.'),
                     (struct_root + '_brain_mask',
                      new_struct_root + '_brain_mask'),
                     (struct_root + '_warped',
                      new_struct_root + '_MNI-r_2_MNI'),
                     (struct_root + '_fieldwarp',
                      new_struct_root + '_MNI-r_2_MNI_fieldwarp'),
                     (struct_root + '_brain_flirt',
                      new_struct_root + '_T1W_2_MNI-r'),
                     (func_root + '_realigned_flirt',
                      new_func_root + '_realigned_2_T1W'),
                     (func_root + '_realigned_warp',
                      new_func_root + '_realigned_2_MNI'),
                     (func_root + '_corrcomped_warp',
                      new_func_root + '_corrcomped_2_MNI')]

    return substitutions


name_substituter = pe.Node(Function(input_names=['subject_id',
                                                 'visit',
                                                 'sMRI_stype',
                                                 'fMRI_stype',
                                                 'svol',
                                                 'fvol'],
                                    output_names=['substitutions'],
                                    function=get_substitutions),
                           name='name_substituter')


def get_last_struct_vol(struct_fnames):
    """
    Retrieve the filename of the last structural scan
    """
    import re
    import numpy as np
    if type(struct_fnames) is str:
        extracted_fname = struct_fnames
    else:
        scanids = np.array([re.search('_S[0-9]+', sfn).group()[2:]
                            for sfn in struct_fnames]).astype(int)
        lastscan_ind = np.argmax(scanids)
        extracted_fname = struct_fnames[lastscan_ind]

    return extracted_fname


fname_selector = pe.Node(Function(input_names=['struct_fnames'],
                                  output_names=['extracted_fname'],
                                  function=get_last_struct_vol),
                         name='fname_selector')


""" Load Registration WorkFlows """
reg_struct_flow = regstruct.create_registration_flow('registration_structural')
reg_func_flow = regfunc.create_registration_flow('registration_functional')

""" Define entire pipeline """
regist_pipe = pe.Workflow(name="Registration_Pipeline")
regist_pipe.base_dir = wf_bdir
regist_pipe.connect([(infosource, datag, [('subject_id', 'subject_id'),
                                          ('visit', 'visit'),
                                          ('sMRI_stype', 'sMRI_stype'),
                                          ('fMRI_stype', 'fMRI_stype')]),
                     (infosource, reg_struct_flow, [('frac_thresh',
                                                     'inputspec.frac_thresh')]),
                     (infosource, datasink, [('subject_id', 'container')]),
                     (datag, name_substituter, [('func_r', 'fvol')]),
                     (fname_selector, name_substituter, [('extracted_fname',
                                                          'svol')]),
                     (infosource, name_substituter, [('subject_id',
                                                      'subject_id'),
                                                     ('visit', 'visit'),
                                                     ('sMRI_stype',
                                                      'sMRI_stype'),
                                                     ('fMRI_stype',
                                                      'fMRI_stype')]),
                     (datag, fname_selector, [('struct', 'struct_fnames')]),
                     (fname_selector, reg_struct_flow, [('extracted_fname',
                                                         'inputspec.struct')]),
                     (name_substituter, datasink, [('substitutions',
                                                    'substitutions')]),
                     (datag, reg_func_flow, [('func_r', 'inputspec.func_r')]),
                     (datag, reg_func_flow, [('func_c', 'inputspec.func_c')]),
                     (reg_struct_flow, reg_func_flow, [('outputspec.brain_extr_file',
                                                        'inputspec.struct_brain'),
                                                       ('outputspec.warp_coeff_file',
                                                        'inputspec.fieldcoeff_file')]),
                     (reg_func_flow, datasink, [('outputspec.func_rInMNI',
                                                 '@func_rInMNI'),
                                                ('outputspec.func_cInMNI',
                                                 '@func_cInMNI'),
                                                ('outputspec.func2struct_rigid',
                                                 '@func2struct_rigid'),
                                                ('outputspec.funcInT1W',
                                                 '@funcInT1W')]),
                     (reg_struct_flow, datasink, [('outputspec.warped_file',
                                                   '@warped_file'),
                                                  ('outputspec.warp_coeff_file',
                                                   '@warped_coeff'),
                                                  ('outputspec.rigid_file',
                                                   '@rigid_file'),
                                                  ('outputspec.rigid_aff_file',
                                                   '@rigid_affine'),
                                                  ('outputspec.brain_extr_file',
                                                   '@brain_extr_file'),
                                                  ('outputspec.brain_mask_file',
                                                   '@brain_mask_file')])
                   ])

if __name__ == '__main__':
    regist_pipe.write_graph(graph2use='flat', format='svg')
    regist_pipe.write_graph(graph2use='colored', format='svg')
    if isinstance(use_multicore, bool) and use_multicore == False:
        regist_pipe.run()
    elif isinstance(use_multicore, int):
        regist_pipe.run('MultiProc', plugin_args={'n_procs': use_multicore})
    else:
        raise ValueError('variable \'use_multicore\' can only be an integer '
                         'or False')
