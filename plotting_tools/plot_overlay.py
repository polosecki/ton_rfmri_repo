# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 19:41:09 2016

@author: pipolose
"""

import numpy as np
from nilearn import plotting
import os
import nibabel as nib
from os.path import join as opj
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap as LSCM
import subprocess
import matplotlib as mpl
from collections import defaultdict
mpl.rcParams['pdf.fonttype'] = 42


in_dict = {}
in_dict['do_save'] = False
in_dict['formats_used'] = ['pdf', 'png']
in_dict['out_dir'] = '/home/CHDI/polo/figures'
in_dict['threshold'] = 0

mymap = LSCM.from_list(name='mymap',
                       colors=['cyan','blue','grey','red','yellow'])


def set_fig_bg_contrast(sh, vmin=4000, vmax=9000):
    '''
    sh: nilearm slicer figure object
    '''
    for vname, vh in sh.axes.items():
        vax = vh.ax
        ims = vax.get_images()
        ims[1].set_clim(vmin=vmin, vmax=vmax)
    return


class NiiThresholder():
    '''
    Class that takes an input nifti file name and provides a temporary
    thresholded nii file, with a minimum cluster size
    '''
    def __init__(self, in_nii_fn=None, cluster_type='pos', thres=1,
                 clus_vox_size=50):
        '''
        Set object parameters
        '''
        self.in_nii_fn = in_nii_fn
        self.cluster_type = cluster_type
        if in_nii_fn:
            self.__set_out_fn(in_nii_fn)
        else:
            self.out_fn = None
        self.thres = thres
        self.clus_vox_size = clus_vox_size
        return

    def __set_out_fn(self, in_nii_fn):
        '''
        Helper function
        '''
        self.out_fn = opj(os.path.dirname(in_nii_fn), 'temp.nii.gz')
        return

    def clean_out_fn(self):
        if os.path.exists(self.out_fn):
            os.remove(self.out_fn)
        return

    def __run_fsl_cmds(self, arg_dict={}):
        fdict = defaultdict(lambda: None)
        for k, v in arg_dict.items():
            fdict[k] = v
        key_list = ['in_nii_fn','thres','out_fn','clus_vox_size',
                    'cluster_type']
        for k in key_list:
            if fdict[k] is None:
                try:
                    fdict[k] = getattr(self, k)
                except AttributeError:
                    pass


        clus_arg_dict = {'pos': '', 'neg': '--min'}
        sign_dict = {'pos': 1, 'neg': -1}
        size_nii_fn = opj(os.path.dirname(fdict['out_fn']), 'size_vol.nii.gz')
        cmd = ('cluster -i {in_nii_fn} --thresh={thresh} --connectivity=6 '
               '--osize={out_fn} '
               '--minextent={clus_vox_size} {clus_type_arg}')
        full_cmd = cmd.format(in_nii_fn=fdict['in_nii_fn'],
                      thresh=sign_dict[fdict['cluster_type']] * fdict['thres'],
                      out_fn=size_nii_fn,
                      clus_vox_size=fdict['clus_vox_size'],
                      clus_type_arg=clus_arg_dict[fdict['cluster_type']])
        op = subprocess.check_output(full_cmd, shell=True)
        print(op)

        cmd = ('fslmaths {in_nii_fn} -mas {size_nii_fn} {out_fn}')
        full_cmd = cmd.format(in_nii_fn=fdict['in_nii_fn'],
                              size_nii_fn=size_nii_fn,
                              out_fn=fdict['out_fn'])
        op = subprocess.check_output(full_cmd, shell=True)
        print(op)

        os.remove(size_nii_fn)
        return

    def make_thresholded_image(self, **kwargs):
        '''
        main method: creates thresholded nifti and provides its filename,
        accepts update of object parameters
        '''
        obj_dir = dir(self)
        for name, val in kwargs:
            if name in obj_dir:
                setattr(self, name, val)
        if self.in_nii_fn:
            self.__set_out_fn(self.in_nii_fn)
        else:
            raise ValueError('in_nii_fn is not defined')

        if self.cluster_type is not 'both':
            self.__run_fsl_cmds()
        else:
            '''
            Run for pos and neg, then sum the results,
            remove intermediate files
            '''
            name_parts = os.path.basename(self.out_fn).split('.')
            out_fns = []
            for ct_idx, used_clus_type in enumerate(['pos', 'neg']):
                out_fn = name_parts[0] + '_{}.'.format(ct_idx) +\
                    '.'.join(name_parts[1:])
                out_fns.append(out_fn)
                self.__run_fsl_cmds({'out_fn': out_fn,
                                     'cluster_type': used_clus_type})
            vols = [nib.load(fn) for fn in out_fns]
            out_arr = np.zeros(vols[0].shape)
            for vol in vols:
                out_arr += vol.get_data()
            out_img = nib.Nifti1Image(out_arr, vol[0].affine,
                                      header=vol[0].header)
            nib.save(out_img, self.out_fn)
            for fn in out_fns:
                os.remove(fn)
        return self.out_fn


def plot_n_save_3plane(in_dict):
    '''
    in_dict fields:
        in_dir
        in_fn
        cut_coords
        do_save (boolean)
        formats_used (a list-like object)
        area (brain area, for naming the output file)
        cmap (optional): a matplotlib colorbar object
        symmetric_cbar (optional): whether to treat cmap as symetrical

    '''
    if 'symmetric_cbar' not in in_dict.keys():
        in_dict['symmetric_cbar'] = 'auto'
    if 'cmap' not in in_dict.keys():
        in_dict['cmap'] = mymap
        if in_dict['symmetric_cbar'] == 'auto':
            in_dict['cmap'] = cm.autumn
    if 'vmax' not in in_dict.keys():
        in_dict['vmax'] = None
    if 'draw_cross' not in in_dict.keys():
        in_dict['draw_cross'] = True
    if 'cluster_dict' not in in_dict.keys():
        cluster_dict = {}
    else:
        cluster_dict = in_dict['cluster_dict']
        cluster_dict['in_nii_fn'] = opj(in_dict['in_dir'], in_dict['in_fn'])

    if cluster_dict:
        thresholder = NiiThresholder(**cluster_dict)
        used_fn = thresholder.make_thresholded_image()
    else:
        used_fn = opj(in_dict['in_dir'], in_dict['in_fn'])

    pv = plotting.plot_stat_map(used_fn,
                                black_bg=True,
                                bg_img=(os.getenv('FSLDIR') +
                                        '/data/standard/MNI152_T1_1mm_brain.nii.gz'),
                                threshold=in_dict['threshold'],
                                cut_coords=in_dict['cut_coords'],
                                cmap = in_dict['cmap'],
                                symmetric_cbar=in_dict['symmetric_cbar'],
                                draw_cross=in_dict['draw_cross'],
                                vmax=in_dict['vmax'])

    set_fig_bg_contrast(pv)

    plt.show()

    if in_dict['do_save']:
        for fmt in in_dict['formats_used']:
            pv.savefig(opj(in_dict['out_dir'], in_dict['in_fn'].split('.')[0]
                       + '_' + in_dict['area'] + '.' + fmt))
    try:
        thresholder.clean_out_fn()
    except NameError:
        pass

    return pv
