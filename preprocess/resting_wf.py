# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import nipype.interfaces.fsl as fsl          # fsl
from nipype.algorithms.misc import TSNR
import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
from nipype.algorithms.rapidart import ArtifactDetect
from nipype.interfaces.fsl.utils import ImageMaths

# For unwarping
from dti_wf import my_robex, invert_contrast, my_ants_registration_syn
from nipype.workflows.dmri.fsl.artifacts import remove_bias
from nipype.interfaces.freesurfer.preprocess import MRIConvert
from nipype.interfaces.ants.resampling import WarpTimeSeriesImageMultiTransform
from nipype.interfaces.fsl.preprocess import FAST


def morph_open_close(vol_in, sphere_radius=7, suffix='smooth'):
    '''
    Performs and open and close morphological operation on input mask,
    using FSL's fslmaths, added py Pablo Polosecki
    '''
    import os

    def splitext2(path):
        for ext in ['.nii.gz']:
            if path.endswith(ext):
                path, ext = path[:-len(ext)], path[-len(ext):]
                break
        else:
            path, ext = os.path.splitext(path)
        return path, ext
    vol_p, vol_e = splitext2(vol_in)
    vol_out = ''.join([vol_p, '_' + suffix, vol_e])
    cmd = ('fslmaths -dt input {vol_in} -kernel sphere {sphere_radius} '
           '-ero -dilD -dilD -ero {vol_out}')
    op = os.system(cmd.format(vol_in=vol_in, sphere_radius=sphere_radius,
                              vol_out=vol_out))
    print(op)
    return vol_out


def concatetante_reg_files(file1, file2):
    '''
    For merging regressors
    '''
    import numpy as np
    import os.path as op

    in_mats = {f: np.loadtxt(f) for f in [file1, file2]}
    out_mat = np.hstack(tuple(in_mats.values()))
    out_dir = op.dirname(file1)
    out_fn = op.join(out_dir, 'merged_regressors.txt')
    np.savetxt(out_fn, out_mat)
    return out_fn


def remove_first_n_frames(in_vol_fn, n_frames=5):
    import nibabel as nib
    import os.path as op

    im_d = nib.load(in_vol_fn)
    in_4d = im_d.get_data()
    out_4d = in_4d[:, :, :, n_frames:]

    out_dir = op.dirname(in_vol_fn)
    in_fn = op.basename(in_vol_fn).split('.')[0]
    out_fn = op.join(out_dir, in_fn + '_start_excised.nii.gz')

    out_header = im_d.header
    out_affine = im_d.affine

    out_img = nib.Nifti1Image(out_4d, out_affine, header=out_header)
    nib.save(out_img, out_fn)
    return out_fn


def extract_noise_components(realigned_file, noise_mask_file, num_components):
    """Derive components most reflective of physiological noise
        num_components: the number of components to use. If <1, it means
        fraction of the variance than needs to be explained.
    """
    import os
    from nibabel import load
    import numpy as np
    import scipy as sp
    imgseries = load(realigned_file)
    components = None
    mask = load(noise_mask_file).get_data()
    voxel_timecourses = imgseries.get_data()[mask > 0]
    voxel_timecourses[np.isnan(np.sum(voxel_timecourses, axis=1)), :] = 0
    # remove mean and normalize by variance
    # voxel_timecourses.shape == [nvoxels, time]
    X = voxel_timecourses.T
    stdX = np.std(X, axis=0)
    stdX[stdX == 0] = 1.
    stdX[np.isnan(stdX)] = 1.
    stdX[np.isinf(stdX)] = 1.
    X = (X - np.mean(X, axis=0))/stdX
    u, s, _ = sp.linalg.svd(X, full_matrices=False)
    # var = np.square(s) / np.square(s).sum()
    if components is None:
        components = u[:, :num_components]
    else:
        components = np.hstack((components, u[:, :num_components]))
    components_file = os.path.join(os.getcwd(), 'noise_components.txt')
    np.savetxt(components_file, components, fmt="%.10f")
    return components_file


def select_volume(filename, which):
    """Return the middle index of a file
    """
    from nibabel import load
    import numpy as np
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filename).get_shape()[3]/2))
    else:
        raise Exception('unknown value for volume selection : %s' % which)
    return idx


def epi_brain_extract(in_file):
    '''
    input should be 3-D
    '''
    import nibabel as nib
    import numpy as np
    from os.path import (dirname as dn, basename as bn, join as opj)
    from os import getcwd
    from nilearn.masking import compute_epi_mask

    in_vol = nib.load(in_file)
    vol_arr = in_vol.get_data()
    # mean_val = np.nanmean(vol_arr)
    mask_img = compute_epi_mask(in_file)
    mask_arr = mask_img.get_data().astype(bool)

    brain_arr = np.zeros_like(vol_arr).astype(float)
    brain_arr[mask_arr] = vol_arr[mask_arr]

    out_header = in_vol.header
    out_affine = in_vol.affine

    out_img = nib.Nifti1Image(brain_arr, out_affine, header=out_header)
    base_dir = getcwd() # dn(in_vol)
    out_vol = opj(base_dir, bn(in_file).split('.')[0] + '_masked.nii.gz')
    out_mask = opj(base_dir, bn(in_file).split('.')[0] + '_mask.nii.gz')
    nib.save(out_img, out_vol)
    nib.save(mask_img, out_mask)
    return out_vol, out_mask




def create_t1_based_unwarp(name='unwarp'):
    """
    Unwarp an fMRI time series based on non-linear registration to T1.
        NOTE: AS IT STANDS THIS METHOD DID NOT PRODUCE ACCEPTABLE RESULTS
        IF BRAIN COVERAGE IS NOT COMPLETE ON THE EPI IMAGE.
        ALSO: NEED TO ADD AUTOMATIC READING OF EPI RESOLUTION TO GET

    """

    unwarpflow = pe.Workflow(name=name)
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['epi',
                                                                 'T1W']),
                        name='inputspec')
    outputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                               'unwarped_func',
                                                               'warp_files']),
                         name='outputspec')

    tmedian = pe.Node(interface=ImageMaths(), name='tmedian')
    tmedian.inputs.op_string = '-Tmedian'
    epi_brain_ext = pe.Node(interface=util.Function(function=epi_brain_extract,
                                                    input_names=['in_file'],
                                                    output_names=['out_vol',
                                                                  'out_mask']),
                            name='epi_brain_ext')

    fast_debias = pe.Node(interface=FAST(), name='FAST_debias')
    fast_debias.inputs.output_biascorrected = True

    robex = pe.Node(interface=util.Function(function=my_robex,
                                            input_names=['in_file'],
                                            output_names=['out_file',
                                                          'out_mask']),
                    name='robex')

    downsample_T1 = pe.Node(MRIConvert(), name='downsample_dti')
    downsample_T1.inputs.vox_size = (3.438, 3.438, 3.000)
    downsample_T1.inputs.out_type = 'niigz'

    contrast_invert = pe.Node(interface=util.Function(function=invert_contrast,
                                                      input_names=['in_t1_brain',
                                                                   'in_b0_brain'],
                                                      output_names=['out_fn']),
                               name='contrast_invert')

    ants_syn = pe.Node(interface=util.Function(function=my_ants_registration_syn,
                                               input_names=['in_T1W',
                                                            'in_epi'],
                                               output_names=['out_transforms']),
                       name='ants_syn')
    ants_warp = pe.Node(interface=WarpTimeSeriesImageMultiTransform(),
                        name='ants_warp')

    '''connections'''
    # unwarpflow.connect(inputnode, 'T1W', robex, 'in_file')
    unwarpflow.connect(inputnode, 'T1W', fast_debias, 'in_files')
    # unwarpflow.connect(robex, 'out_file', fast_debias, 'in_files')
    unwarpflow.connect(fast_debias, 'restored_image', robex, 'in_file')
    # unwarpflow.connect(fast_debias, 'restored_image', downsample_T1, 'in_file')
    unwarpflow.connect(robex, 'out_file', downsample_T1, 'in_file')
    unwarpflow.connect(downsample_T1, 'out_file', contrast_invert, 'in_t1_brain')
    unwarpflow.connect(inputnode, 'epi', tmedian, 'in_file')
    unwarpflow.connect(tmedian, 'out_file', epi_brain_ext, 'in_file')
    unwarpflow.connect(epi_brain_ext, 'out_vol', contrast_invert, 'in_b0_brain')
    unwarpflow.connect(contrast_invert, 'out_fn', ants_syn, 'in_T1W')
    unwarpflow.connect(epi_brain_ext, 'out_vol', ants_syn, 'in_epi')
    unwarpflow.connect(ants_syn, 'out_transforms', outputnode, 'out_transforms')

    unwarpflow.connect(inputnode, 'epi', ants_warp, 'input_image')
    unwarpflow.connect(contrast_invert, 'out_fn', ants_warp, 'reference_image')
    unwarpflow.connect(ants_syn, 'out_transforms', ants_warp, 'transformation_series')

    unwarpflow.connect(ants_syn, 'out_transforms', outputnode, 'warp_files')
    unwarpflow.connect(ants_warp, 'output_image', outputnode, 'unwarped_func')

    return unwarpflow

def create_realign_flow(name='realign'):
    """Realign a time series to the middle volume using spline interpolation

    Uses MCFLIRT to realign the time series and ApplyWarp to apply the rigid
    body transformations using spline interpolation (unknown order).

    Example
    -------

    >>> wf = create_realign_flow()
    >>> wf.inputs.inputspec.func = 'f3.nii'
    >>> wf.run() # doctest: +SKIP

    """
    realignflow = pe.Workflow(name=name)
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 ]),
                        name='inputspec')
    outputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                               'realigned_file',
                                                               'rms_files',
                                                               'par_file']),
                         name='outputspec')
    start_dropper = pe.Node(util.Function(input_names=['in_vol_fn',
                                                       'n_frames'],
                                          output_names=['out_fn'],
                                          function=remove_first_n_frames),
                            name='start_dropper')
    start_dropper.inputs.n_frames = 5

    realigner = pe.Node(fsl.MCFLIRT(save_mats=True, stats_imgs=True,
                                    save_rms=True, save_plots=True),
                        name='realigner')

    splitter = pe.Node(fsl.Split(dimension='t'), name='splitter')
    warper = pe.MapNode(fsl.ApplyWarp(interp='spline'),
                        iterfield=['in_file', 'premat'],
                        name='warper')
    joiner = pe.Node(fsl.Merge(dimension='t'), name='joiner')

    realignflow.connect(inputnode, 'func', start_dropper, 'in_vol_fn')
    realignflow.connect(start_dropper, 'out_fn', realigner, 'in_file')
    realignflow.connect(start_dropper, ('out_fn', select_volume, 'middle'),
                        realigner, 'ref_vol')
    realignflow.connect(realigner, 'out_file', splitter, 'in_file')
    realignflow.connect(realigner, 'mat_file', warper, 'premat')
    realignflow.connect(realigner, 'variance_img', warper, 'ref_file')
    realignflow.connect(splitter, 'out_files', warper, 'in_file')
    realignflow.connect(warper, 'out_file', joiner, 'in_files')
    realignflow.connect(joiner, 'merged_file', outputnode, 'realigned_file')
    realignflow.connect(realigner, 'rms_files', outputnode, 'rms_files')
    realignflow.connect(realigner, 'par_file', outputnode, 'par_file')
    return realignflow


def create_resting_preproc(name='restpreproc'):
    """Create a "resting" time series preprocessing workflow

    The noise removal is based on Behzadi et al. (2007)

    Parameters
    ----------

    name : name of workflow (default: restpreproc)

    Inputs::

        inputspec.func : functional run (filename or list of filenames)

    Outputs::

        outputspec.noise_mask_file : voxels used for PCA to derive noise components
        outputspec.filtered_file : bandpass filtered and noise-reduced time series

    Example
    -------

    >>> TR = 3.0
    >>> wf = create_resting_preproc()
    >>> wf.inputs.inputspec.func = 'f3.nii'
    >>> wf.inputs.inputspec.num_noise_components = 6
    >>> wf.inputs.inputspec.highpass_sigma = 100/(2*TR)
    >>> wf.inputs.inputspec.lowpass_sigma = 12.5/(2*TR)
    >>> wf.run() # doctest: +SKIP

    """

    restpreproc = pe.Workflow(name=name)

    # Define nodes
    inputnode = pe.Node(interface=util.IdentityInterface(fields=['func',
                                                                 'num_noise_components',
                                                                 'highpass_sigma',
                                                                 'lowpass_sigma'
                                                                 ]),
                        name='inputspec')
    outputnode = pe.Node(interface=util.IdentityInterface(fields=[
                                                              'noise_mask_file',
                                                              'filtered_file',
                                                              'motion_rms_files',
                                                              'motion_par_file',
                                                              'realigned_file',
                                                              'mask_file',
                                                              'outlier_files',
                                                              'intensity_files',
                                                              'outlier_plots']),
                         name='outputspec')
    slicetimer = pe.Node(fsl.SliceTimer(), name='slicetimer')
    realigner = create_realign_flow()

    art_detector = pe.Node(ArtifactDetect(), name='art_detector')
    art_detector.inputs.parameter_source = 'FSL'
    art_detector.inputs.mask_type = 'spm_global'
    art_detector.inputs.global_threshold = .5
    art_detector.inputs.norm_threshold = .6
    art_detector.inputs.use_differences = [True, True] ## [Movement, Intensity]
    art_detector.inputs.zintensity_threshold = 3
    art_detector.inputs.intersect_mask = True

    '''Mask smoother node, added by Pablo Polosecki to use EPI mask'''
    mask_smoother = pe.Node(util.Function(input_names=['vol_in'],
                                          output_names=['out_vol'],
                                          function=morph_open_close),
                            name='mask_smoother')
    tsnr = pe.Node(TSNR(regress_poly=2), name='tsnr')
    getthresh = pe.Node(interface=fsl.ImageStats(op_string='-k %s -p 98'),
                           name='getthreshold')
    threshold_stddev = pe.Node(fsl.Threshold(), name='threshold')

    ''' Mask conjunction, to limit noisy voxels to those inside brain mask'''
    conj_masker  = pe.Node(fsl.BinaryMaths(operation='mul'),
                           name='conj_masker')

    compcor = pe.Node(util.Function(input_names=['realigned_file',
                                                 'noise_mask_file',
                                                 'num_components'],
                                     output_names=['noise_components'],
                                     function=extract_noise_components),
                       name='compcorr')
 #   cat_regressors = pe.Node(util.Function(input_names=['file1',
 #                                                       'file2'],
 #                                          output_names=['out_fn'],
 #                                          function=concatetante_reg_files),
 #                            name='cat_regressors')
    remove_noise = pe.Node(fsl.FilterRegressor(filter_all=True),
                           name='remove_noise')
    bandpass_filter = pe.Node(fsl.TemporalFilter(),
                              name='bandpass_filter')

    # Define connections
    restpreproc.connect(inputnode, 'func', slicetimer, 'in_file')
    restpreproc.connect(slicetimer, 'slice_time_corrected_file',
                        realigner, 'inputspec.func')
    restpreproc.connect(realigner, 'outputspec.realigned_file',
                        tsnr, 'in_file')
    restpreproc.connect(tsnr, 'stddev_file', threshold_stddev, 'in_file')
    restpreproc.connect(tsnr, 'stddev_file', getthresh, 'in_file')
    restpreproc.connect(mask_smoother, 'out_vol', getthresh, 'mask_file')
    restpreproc.connect(getthresh, 'out_stat', threshold_stddev, 'thresh')
    restpreproc.connect(realigner, 'outputspec.realigned_file',
                        compcor, 'realigned_file')
    restpreproc.connect(inputnode, 'num_noise_components',
                        compcor, 'num_components')
    restpreproc.connect(tsnr, 'detrended_file',
                        remove_noise, 'in_file')
    # Combiinng compcorr with motion regressors:
    #restpreproc.connect(compcor, 'noise_components',
    #                    cat_regressors, 'file1')
    #restpreproc.connect(realigner, 'outputspec.par_file',
    #                    cat_regressors, 'file2')
    #restpreproc.connect(cat_regressors, 'out_fn',
    #                    remove_noise, 'design_file')
    restpreproc.connect(compcor, 'noise_components',
                         remove_noise, 'design_file')
    restpreproc.connect(inputnode, 'highpass_sigma',
                        bandpass_filter, 'highpass_sigma')
    restpreproc.connect(inputnode, 'lowpass_sigma',
                        bandpass_filter, 'lowpass_sigma')
    restpreproc.connect(remove_noise, 'out_file', bandpass_filter, 'in_file')
    restpreproc.connect(conj_masker, 'out_file',
                        outputnode, 'noise_mask_file')
    restpreproc.connect(bandpass_filter, 'out_file',
                        outputnode, 'filtered_file')
    restpreproc.connect(realigner, 'outputspec.rms_files',
                        outputnode, 'motion_rms_files')
    restpreproc.connect(realigner, 'outputspec.par_file',
                        outputnode, 'motion_par_file')
    restpreproc.connect(realigner, 'outputspec.realigned_file',
                        outputnode, 'realigned_file')
    restpreproc.connect(realigner, 'outputspec.realigned_file',
                        art_detector, 'realigned_files')
    restpreproc.connect(realigner, 'outputspec.par_file',
                        art_detector, 'realignment_parameters')
    restpreproc.connect(art_detector, 'mask_files',
                        mask_smoother, 'vol_in')
    restpreproc.connect(mask_smoother, 'out_vol',
                        outputnode, 'mask_file')
    restpreproc.connect(art_detector, 'outlier_files',
                        outputnode, 'outlier_files')
    restpreproc.connect(art_detector, 'intensity_files',
                        outputnode, 'intensity_files')
    #restpreproc.connect(art_detector, 'plot_files',
    #                    outputnode, 'outlier_plots')
    restpreproc.connect(mask_smoother, 'out_vol',
                        conj_masker, 'in_file')
    restpreproc.connect(threshold_stddev, 'out_file',
                        conj_masker, 'operand_file')
    restpreproc.connect(conj_masker, 'out_file',
                        compcor, 'noise_mask_file')
    return restpreproc