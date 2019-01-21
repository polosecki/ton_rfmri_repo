# -*- coding: utf-8 -*-
"""

TOOLS FOR USING NIPYPE IN FILE CONVERSION WORKFLOWS

Created on Wed Feb 10 11:09:09 2016
@author: pipolose
"""

import nipype.pipeline.engine as pe
from nipype.interfaces.utility import Function
from nipype.interfaces.freesurfer import MRIConvert
from nipype.interfaces.fsl.utils import Reorient2Std
from nipype.interfaces.dcm2nii import Dcm2nii
from nipype.interfaces.fsl import Merge

from nipype.interfaces.base import TraitedSpec, OutputMultiPath, File
import os


this_dir = os.path.dirname(os.path.abspath(__file__))
env = os.environ
pypath = 'PYTHONPATH'
if pypath in env.keys():
    env[pypath] += ':{}'.format(this_dir)
else:
    env[pypath] = ':{}'.format(this_dir)

'''
Useful function definitions
------------------------
'''


# Function version of the subject_dict to use in nipype node:
def subj_object_provider(subj_name, subj_dict):
    """
    Convenience function version of the subject_dict, to be turned into a
    nipype node.

    Parameters
    ----------
    subj_name : str
        subject ID
    subj_dict : dict
        Dictionary of explore.hf subject objects

    Returns
    -------
    subj_obj : explore.hf subject object
        Corresponding subject object
    """
    subj_obj = subj_dict[subj_name]
    return subj_obj


# Provide path to dcm files:
def dcm_provider(subj_obj, visit=None, subtype=None):
    """
    Convenience function that provides path to dcm files to be used in Nipype
    Node

    Parameters
    ----------
    subj_obj : explore.hf subject object
        subject object
    visit: int, default: None
        Visit number. If None, all visits are used.
    subtype: str
        Scan subtype description, as it appears on mri.csv (for TON). For THD
        it only takes 'DTI', 'T1W', 'T1W Repeat' and 'T2W'

    Returns
    -------
    dcm_list : list
        List of absolute paths to dcm files, one per scan.
    """
    import numpy as np
    try:
        dcm_list = filter(bool, subj_obj.get_scan_dicom(visit=visit,
                                                        subtype=subtype))
    except KeyError:
        dcm_list = []
    dcm_list = [w.replace(' ', '_') for w in dcm_list if w is not '']

    DTI_subtrings = ['DIRECTION/', 'ep2d_diff_48dir_p2_7b0/', 'HIGH/', 'DTI_5',
                     'DTI_4']
    # 'DTI_4', 'DTI_5',
    if subtype == 'DTI':
        dcm_list = [dcm for dcm in dcm_list
                    if np.any([substr in dcm for substr in DTI_subtrings])]

    return dcm_list


def rename_vol(input_vol, subject_id, subj_obj, visit=None, subtype=None):
    """
    Convenience function that makes a copy of a volume file with a nice name,
    to be used as a Nipype node.

    Parameters
    ----------
    input_vol :str
        Absolute path to the input volume
    subject_id : str
        self-explanatory
    subj_obj: explore.hf subject object
        subject object
    visit: int, default: None
        Visit number. If None, all visits are assumed. CURRENTLY MUST PROVIDE
        A VISIT NUMBER FOR NAME TO BE OK.
    subtype: ste, default: None
        Scan subtype as in mri.csv. If None, all subtypes are assumed.
        CURRENTLY MUST PROVIDE A SUBTYPE STR FOR NAME TO BE OK.

    Returns
    -------
    out_file : str
        Absolute path to nicely named volume.
    """
    from os.path import join as opj
    from os.path import dirname as opd
    from os.path import isfile as isf
    from os.path import splitext
    from os import remove
    import pandas as pd
    import shutil

    def splitext2(path):
        for ext in ['.nii.gz']:
            if path.endswith(ext):
                path, ext = path[:-len(ext)], path[-len(ext):]
                break
        else:
            path, ext = splitext(path)
        return ext

    if 'DTI' in subtype:
        if hasattr(subtype, 'extend'):
            subtype.extend('Generic')
        else:
            subtype = [subtype, 'Generic']

    idx = pd.IndexSlice
    mri_df = subj_obj.mri
    if not mri_df.index.is_lexsorted():
        mri_df = mri_df.sort_index(in_place=True)

    if visit is None:
        if subtype is None:
            used_df = mri_df
        else:
            used_df = mri_df.loc[idx[subtype, :], :]
    else:
        if subtype is None:
            used_df = mri_df.loc[idx[:, 'Visit ' + str(visit)], :]
        else:
            used_df = mri_df.loc[idx[subtype, 'Visit ' + str(visit)], :]

    scanid = ['S' + scid for scid in
               used_df[used_df['scanstatus'] == 1]['scanid'].values.astype(str)]
    scanid = [s for s in scanid if s in input_vol][0]

    fext = splitext2(input_vol)
    dn = opd(input_vol)
    if hasattr(subtype, 'extend'):
        used_subtype = subtype[0]
    else:
        used_subtype = subtype
    used_subtype = used_subtype.replace(' ', '_')
    fn = '_'.join([subject_id, 'visit', str(visit), used_subtype, scanid])
    out_file = opj(dn, fn + fext)
    if isf(out_file):
        remove(out_file)
    # rename(input_vol,out_file)
    shutil.copyfile(input_vol, out_file)
    return out_file

class DTIMRIConvertOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True),
                               desc='converted output file')
    out_bvals = OutputMultiPath(File(exists=True),
                                desc='converted output file')
    out_gradients = OutputMultiPath(File(exists=True),
                                    desc='converted output file')


class DTIMRIConvert(MRIConvert):
    '''
    MRIConvert for DTI:
        This is a wrapper for FreeSurfer 6's mri_convert, which outputs a
        b-value file, and a gradient direction vector file.
        This is supposed to be used only for DWI files.
        It will produce errors because otherwise because some declared
        outputs will be non-existent.
    Inputs:
        Same as nipype.interfaces.freesurfer.MRIConvert
    Outputs:,
        out_file: Full path to converted file
        out_bvals: Full path to text file with b-values
        out_gradients: Full path to text file with gradient directions

    '''
    output_spec = DTIMRIConvertOutputSpec
    _cmd = ('/home/CHDI/opt/freesurfer-Linux-centos6'
            '_x86_64-stable-pub-v6.0.0/bin/mri_convert.bin')

    def _list_outputs(self):
        outputs = self.output_spec().get()
        of = self._get_outfilename()
        of_parts = of.split('.')
        if of_parts[-1] == 'gz':
            root_fn = '.'.join(of_parts[:-2])
        else:
            root_fn = '.'.join(of_parts[:-1])
        outputs['out_file'] = of
        outputs['out_bvals'] = '.'.join([root_fn, 'bvals'])
        outputs['out_gradients'] = '.'.join([root_fn, 'voxel_space', 'bvecs'])
        return outputs

def cat_n_transpose_b0_in_vects(in_bval, in_bvec, dti_fn, b0_idx=0):
    '''
    Utility function for adding b-0 values and transposing b-files
    from a strange Philips scanner site in TRACK-ON (3823)
    in_bval: bval file (potentially in list form) returned by dcm2nii
    in_bvec: bvec file (potentially in list form) returned by dcm2nii
    dti_fn: file name of dti_file to use for ouput
    '''
    import os.path as op
    import numpy as np
    from os import getcwd

    if isinstance(in_bval, list):
        in_bval = in_bval[0]
    if isinstance(in_bvec, list):
        in_bvec = in_bvec[0]

    fn_base = op.basename(dti_fn).split('.')[0]

    # Load (and transpose!!)
    bvec_arr = np.loadtxt(in_bvec).T
    out_bvec = np.zeros((bvec_arr.shape[0] + 1,
                         bvec_arr.shape[1]))
    out_bvec[:] = np.nan
    out_bvec[b0_idx, :] = 0
    out_bvec[np.where(np.isnan(out_bvec))] = bvec_arr.flatten()

    bval_arr = np.loadtxt(in_bval)
    out_bval = np.zeros((bval_arr.shape[0] + 1,))
    out_bval[:] = np.nan
    out_bval[b0_idx] = 0
    out_bval[np.isnan(out_bval)] = bval_arr

    out_bvec_fn = op.join(getcwd(), fn_base + '.bvec')
    np.savetxt(out_bvec_fn, out_bvec, fmt='%.8f')

    out_bval_fn = op.join(getcwd(), fn_base + '.bval')

    np.savetxt(out_bval_fn, out_bval, fmt='%.6f')
    return out_bvec_fn, out_bval_fn


def grab_dcm_dir(in_dcm):
    import os.path as op
    return op.dirname(in_dcm)

def weird_convert_dti_dcm(in_dcm):

    import os
    import numpy as np
    import re

    subjid = re.search('R[0-9X]+', in_dcm).group()
    year = re.search('_201[1234]', in_dcm).group()[1:]
    visit_dict = {'2012': 1, '2013': 2, '2014': 3, '2011': 4}
    visit = visit_dict[year]
    scanid = re.search('S[0-9]+', in_dcm).group()
    ton_dir = '/data1/cooked/TONf'
    test_fn = os.path.join(ton_dir, subjid, 'visit_{}'.format(visit), 'DTI',
                           '_'.join([subjid, 'visit', str(visit), 'DTI',
                                     scanid])) + '.bvals'
    if os.path.exists(test_fn):
        assert np.all(np.loadtxt(test_fn) != 0)
    converter = Dcm2nii()
    converter.inputs.source_names = in_dcm
    converter.inputs.gzip_output = True
    converter.inputs.output_dir = os.getcwd()

    converter.run()

    merger = Merge()
    merger.inputs.in_files = converter.output_files
    merger.inputs.dimension = 't'
    merged_result = merger.run()
    fn_base = os.path.basename(in_dcm).split('.')[0]

    merged_file = os.path.join(os.getcwd(), fn_base + '.nii.gz')
    os.rename(merged_result.outputs.merged_file, merged_file)

    in_bval = converter.bvals[0]
    in_bvec = converter.bvecs[0]
    b0_idx = 0
    assert np.all(np.loadtxt(in_bval) != 0)

    # Load (and transpose!!)
    bvec_arr = np.loadtxt(in_bvec).T
    out_bvec = np.zeros((bvec_arr.shape[0] + 1,
                         bvec_arr.shape[1]))
    out_bvec[:] = np.nan
    out_bvec[b0_idx, :] = 0
    out_bvec[np.where(np.isnan(out_bvec))] = bvec_arr.flatten()

    bval_arr = np.loadtxt(in_bval)
    out_bval = np.zeros((bval_arr.shape[0] + 1,))
    out_bval[:] = np.nan
    out_bval[b0_idx] = 0
    out_bval[np.isnan(out_bval)] = bval_arr

    out_bvec_fn = os.path.join(os.getcwd(), fn_base + '.bvecs')
    np.savetxt(out_bvec_fn, out_bvec, fmt='%.8f')

    out_bval_fn = os.path.join(os.getcwd(), fn_base + '.bvals')

    np.savetxt(out_bval_fn, out_bval, fmt='%.6f')
    return merged_file, out_bvec_fn, out_bval_fn


'''
Conversion Nodes and Workflow definitions
---------------------
'''

#: Define object generating node:
#: Requires the following input by user:
#: sub_obj_prov_node.inputs.subj_dict = subjects_dict
sub_obj_prov_node = pe.Node(Function(input_names=['subj_name', 'subj_dict'],
                                     output_names=['subj_obj'],
                                     function=subj_object_provider),
                            name='sub_obj_prov_node')


#: dcm provider node:
#: Requires the following input by user:
#: dcm_prov_node.inputs.subtype = 'T1W'
dcm_prov_node = pe.Node(Function(input_names=['subj_obj',
                                              'visit',
                                              'subtype'],
                                 output_names=['dcm_list'],
                                 function=dcm_provider),
                        name='dcm_prov_node')

#: MapNode (i.e., takes list as an input type) for Freesurfer's MRIConvert
dcm_2_nii = pe.MapNode(MRIConvert(), name='dcm_2_nii', iterfield=['in_file'])



#: MapNode (i.e., takes list as an input type) for FSL's reorient2std
reorient = pe.MapNode(Reorient2Std(), name='reorient', iterfield=['in_file'])

#: Conversion WorkFlow that connects MRIConvert to reorient2std
conversion_wf = pe.Workflow(name='conversion_wf')
conversion_wf.connect([(dcm_prov_node, dcm_2_nii, [('dcm_list', 'in_file')]),
                       (dcm_2_nii, reorient, [('out_file', 'in_file')]),
                       ])

'''
Specialized DTI conversion node and worflow
'''
#: MapNode (i.e., takes list as an input type) for Freesurfer's MRIConvert
dti_dcm_2_nii = pe.MapNode(DTIMRIConvert(),
                           name='dti_dcm_2_nii',
                           iterfield=['in_file'])

dti_conversion_wf = pe.Workflow(name='dti_conversion_wf')
dti_conversion_wf.connect([(dcm_prov_node, dti_dcm_2_nii, [('dcm_list',
                                                            'in_file')])
                           ])



'''
Even more specialized DTI conversion node and Workflow
for unusual TRACK-ON Philips site
'''
weird_dti_dcm_2_nii = pe.MapNode(Function(function=weird_convert_dti_dcm,
                                          input_names=['in_dcm'],
                                          output_names=['out_nii',
                                                        'out_bvec',
                                                        'out_bval']),
                                name='weird_dti_dcm_2_nii',
                                iterfield=['in_dcm'])
weird_dti_conversion_wf = pe.Workflow(name='weird_dti_conversion_wf')
weird_dti_conversion_wf.connect([(dcm_prov_node, weird_dti_dcm_2_nii,
                                  [('dcm_list', 'in_dcm')])
                                ])


#: Rename Node: useful for renaming the output of conversion WF nicely
ren_node = pe.MapNode(Function(input_names=['input_vol',
                                            'subject_id',
                                            'visit',
                                            'subtype',
                                            'subj_obj'],
                               output_names=['out_file'],
                               function=rename_vol),
                      name='ren_node',
                      iterfield=['input_vol'])
'''
FIELDMAP conversion node
'''

fm_conversion_wf = pe.Workflow(name='fm_conversion_wf')
fm_conversion_wf.connect([(dcm_prov_node, dcm_2_nii, [('dcm_list', 'in_file')])
                          ])
