# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 17:14:23 2016

@author: ecastrow
"""

import nipype.pipeline.engine as pe
from nipype.interfaces import fsl
import nipype.interfaces.utility as util
from os.path import join as opj


def create_registration_flow(name='registration'):
    """
    Do registration of functional MR images to MNI standard space using T1W
    volumes as an intermediate reference.

    Uses FSL's FLIRT for rigid body registration (degrees of freedom = 6) and
    ApplyWarp for nonlinear transformation to standard space using T1W to MNI
    spline coefficients. FMRI images are warped to a downsampled version of
    the MNI atlas to match fMRI volumes' spatial resolution.

    Example
    -------
    >>> wf = create_registration_flow()
    >>> wf.inputs.inputspec.func_r = 'fvols_realigned.nii'
    >>> wf.inputs.inputspec.func_c = 'fvols_corrected.nii'
    >>> wf.inputs.inputspec.struct_brain = 'svol_betted.nii'
    >>> wf.inputs.inputspec.fieldcoeff_file = 'T1W_2_MNI_fieldwarp.nii'
    >>> wf.run()
    """
    # Node for inputs
    inputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                         'func_r',
                                                         'func_c',
                                                         'struct_brain',
                                                         'fieldcoeff_file']),
                        name='inputspec')

    # Node for outputs
    outputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                         'func_rInMNI',
                                                         'func_cInMNI',
                                                         'funcInT1W',
                                                         'func2struct_rigid']),
                         name='outputspec')

    # Rigid-body transformation of fMRI vols to SMRI (inter-modal corregistration)
    flt = fsl.FLIRT()
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.dof = 6
    rigid_func2struct = pe.MapNode(interface=flt, name='rigid_func2struct',
                                   iterfield=['in_file', 'reference'])

    # Warping nodes from fMRI to MNI (using nonlinear transf from sMRI to MNI)
    templates_dir = '/data1/standard_space'
    aw = fsl.ApplyWarp()
    aw.inputs.ref_file = opj(templates_dir, 'MNI152_T1_3mm.nii.gz')
    aw.inputs.mask_file = opj(templates_dir,
                              'MNI152_T1_3mm_brain_mask_dil_bin.nii.gz')
    warp_func2MNIr = pe.MapNode(interface=aw, name='warp_func2MNI_realigned',
                                iterfield=['in_file', 'field_file', 'premat'])
    warp_func2MNIc = pe.MapNode(interface=aw, name='warp_func2MNI_corrected',
                                iterfield=['in_file', 'field_file', 'premat'])

    # Define workflow and connections between nodes
    regist_func = pe.Workflow(name=name)
    regist_func.connect(inputnode, 'func_r', rigid_func2struct, 'in_file')
    regist_func.connect(inputnode, 'struct_brain', rigid_func2struct,
                        'reference')
    regist_func.connect(inputnode, 'func_r', warp_func2MNIr, 'in_file')
    regist_func.connect(inputnode, 'func_c', warp_func2MNIc, 'in_file')
    regist_func.connect(inputnode, 'fieldcoeff_file', warp_func2MNIr,
                        'field_file')
    regist_func.connect(inputnode, 'fieldcoeff_file', warp_func2MNIc,
                        'field_file')
    regist_func.connect(rigid_func2struct, 'out_matrix_file', warp_func2MNIr,
                        'premat')
    regist_func.connect(rigid_func2struct, 'out_matrix_file', warp_func2MNIc,
                        'premat')
    regist_func.connect(rigid_func2struct, 'out_matrix_file', outputnode,
                        'func2struct_rigid')
    regist_func.connect(rigid_func2struct, 'out_file', outputnode,
                        'funcInT1W')
    regist_func.connect(warp_func2MNIr, 'out_file', outputnode,
                        'func_rInMNI')
    regist_func.connect(warp_func2MNIc, 'out_file', outputnode,
                        'func_cInMNI')

    return regist_func
