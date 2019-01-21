# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:57:23 2016

@author: ecastrow
"""

import nipype.pipeline.engine as pe
from nipype.interfaces import fsl
from nipype.interfaces import freesurfer as fs
import nipype.interfaces.utility as util


def create_registration_flow(name='registration'):
    """
    Do registration of structural MRI images (T1-weighted) to MNI standard
    space

    Uses FSL's BET for brain extraction, FLIRT for affine registration
    (degrees of freedom = 12) and FNIRT for nonlinear transformation to
    standard space.

    Example
    -------
    >>> wf = create_registration_flow()
    >>> wf.inputs.inputspec.frac_thresh = 0.4
    >>> wf.inputs.inputspec.struct = 'svol.nii'
    >>> wf.run()
    """

    regist_flow = pe.Workflow(name=name)

    # Node for inputs
    inputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                            'struct',
                                                            'frac_thresh']),
                        name='inputspec')

    # Node for outputs
    outputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                        'brain_extr_file',
                                                        'brain_mask_file',
                                                        'rigid_file',
                                                        'rigid_aff_file',
                                                        'warped_file',
                                                        'warp_coeff_file']),
                         name='outputspec')

    # Define brain extraction node (reduce bias, discard neck voxels, apply
    # intensity threshold)
    btr = fsl.BET()
    btr.inputs.reduce_bias = True
    btr.inputs.threshold = True
    btr.inputs.mask = True
    brain_extractor = pe.MapNode(interface=btr, name='brain_extraction',
                                 iterfield=['in_file'])

    # Define node for affine registr (prior to warping image to standard space)
    flt = fsl.FLIRT()
    flt.inputs.reference = "/home/CHDI/opt/fsl/data/standard/"\
                           "MNI152_T1_1mm_brain.nii.gz"
    flt.inputs.output_type = "NIFTI_GZ"
    flt.inputs.dof = 12
    affine_regist = pe.MapNode(interface=flt, name='affine_registration',
                               iterfield=['in_file'])

    # Define node for nonlinear registration (warping) to standard space
    fnt = fsl.FNIRT()
    fnt.inputs.ref_file = "/home/CHDI/opt/fsl/data/standard/"\
                          "MNI152_T1_1mm.nii.gz"
    fnt.inputs.refmask_file = "/home/CHDI/opt/fsl/data/standard/"\
                              "MNI152_T1_1mm_brain_mask_dil.nii.gz"
    fnt.inputs.config_file = 'T1_2_MNI152_2mm'
    # Also, enable the generation of an image of spline coefficients
    fnt.inputs.fieldcoeff_file = True
    warping = pe.MapNode(interface=fnt, name='warping',
                         iterfield=['in_file', 'affine_file'])

    # Resampling nodes for brain-extracted T1W volumes (adjust for slight
    # variations of voxel resolution among subjects)
    mc = fs.MRIConvert()
    mc.inputs.vox_size = (1.1, 1.1016, 1.1016)
    resamp_brain = pe.MapNode(interface=mc, name='resampling_extracted_brain',
                              iterfield=['in_file'])

    # Connect constituent nodes of registration workflow
    regist_flow.connect(inputnode, 'struct', brain_extractor, 'in_file')
    regist_flow.connect(inputnode, 'frac_thresh', brain_extractor, 'frac')
    regist_flow.connect(inputnode, 'struct', warping, 'in_file')
    regist_flow.connect(brain_extractor, 'out_file', affine_regist, 'in_file')
    regist_flow.connect(brain_extractor, 'out_file', resamp_brain, 'in_file')
    regist_flow.connect(affine_regist, 'out_matrix_file', warping,
                        'affine_file')
    regist_flow.connect(resamp_brain, 'out_file', outputnode,
                        'brain_extr_file')
    regist_flow.connect(brain_extractor, 'mask_file', outputnode,
                        'brain_mask_file')
    regist_flow.connect(affine_regist, 'out_file', outputnode,
                        'rigid_file')
    regist_flow.connect(affine_regist, 'out_matrix_file', outputnode,
                        'rigid_aff_file')
    regist_flow.connect(warping, 'warped_file', outputnode,
                        'warped_file')
    regist_flow.connect(warping, 'fieldcoeff_file', outputnode,
                        'warp_coeff_file')
    return regist_flow


def brain_extraction_module(name='brain_extraction'):
    """
    Generate simple module for brain extraction for visits for which only
    structural images are available (e.g., Track study or PREDICT)

    Example
    -------
    >>> wf = brain_extraction_module()
    >>> wf.inputs.inputspec.frac_thresh = 0.4
    >>> wf.inputs.inputspec.struct = 'svol.nii'
    >>> wf.run()
    """

    brain_extraction_flow = pe.Workflow(name=name)

    # Node for inputs
    inputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                            'struct',
                                                            'frac_thresh']),
                        name='inputspec')

    # Node for outputs
    outputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                        'brain_extr_file',
                                                        'brain_mask_file']),
                         name='outputspec')

    # Define brain extraction node (reduce bias, discard neck voxels, apply
    # intensity threshold)
    btr = fsl.BET()
    #btr.inputs.reduce_bias = True  # <-- to be used for T1W images only
    btr.inputs.threshold = True
    btr.inputs.mask = True
    btr.inputs.robust = True        # <-- to be used for T2W images only
    brain_extractor = pe.MapNode(interface=btr, name='brain_extraction',
                                 iterfield=['in_file'])

    # Resampling nodes for brain-extracted T1W volumes (adjust for slight
    # variations of voxel resolution among subjects)
    mc_br = fs.MRIConvert()
    mc_br.inputs.vox_size = (1.1, 1.1016, 1.1016)
    resamp_brain = pe.MapNode(interface=mc_br, name='resampling_extracted_brain',
                              iterfield=['in_file'])
    mc_mk = fs.MRIConvert()
    mc_mk.inputs.vox_size = (1.1, 1.1016, 1.1016)
    mc_mk.inputs.resample_type = 'nearest'
    resamp_mask = pe.MapNode(interface=mc_mk, name='resampling_brain_mask',
                             iterfield=['in_file'])    # <-- for T2W images

    # Connect constituent nodes of registration workflow
    brain_extraction_flow.connect(inputnode, 'struct',
                                  brain_extractor, 'in_file')
#    brain_extraction_flow.connect(inputnode, 'frac_thresh',  # <-- for T1W images
#                                  brain_extractor, 'frac')
    brain_extraction_flow.connect(brain_extractor, 'out_file', resamp_brain,
                                  'in_file')
    brain_extraction_flow.connect(resamp_brain, 'out_file', outputnode,
                                  'brain_extr_file')
    brain_extraction_flow.connect(brain_extractor, 'mask_file', resamp_mask,
                                  'in_file')    # <-- for T2W images
    brain_extraction_flow.connect(resamp_mask, 'out_file', outputnode,
                                  'brain_mask_file')    # <-- for T2W images
#    brain_extraction_flow.connect(brain_extractor, 'mask_file', outputnode,
#                                  'brain_mask_file')   # <-- for T1W images
    return brain_extraction_flow
