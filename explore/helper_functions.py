# -*- coding: utf-8 -*-
"""
Helper functions to organize CHDI imaging data
Created on Fri Jan 15 11:07:53 2016
@author: Pablo Polosecki
Python Version: Python 3.5.1 |Anaconda 2.4.1 (64-bit)
"""

import glob as gl
import pandas as pd
import numpy as np
import os

from functools import partial


def linear_pred(m,b,x):
    y = m * x + b
    return y

def scan_year(visit, studyid='TON'):
    """
    Retrieve the year in which a scan was collected.

    Parameters
    ----------
    visit : str or int
        Visit number
    studyid: str, optional
        Specifies the study from which files will be retrieved. Valid
        values are 'THD' and 'TON'.

    Returns
    -------
    sc_year : int
        Actual scan year
    """
    if type(visit) is str:
        visit = int(visit[-1:])
    if studyid == 'TON':
        years = [2012, 2013, 2014]
    else:
        years = [2008, 2009, 2010, 2011]
    sc_year = years[visit-1]
    return sc_year

# Define root directories of every type of scan for each study (TON or THD)
# For TON: mritype = (0: unknown; 1: sMRI; 2:fMRI; 3: DTI)
# For THD: mritype = (0,4: DTI; 1,2,3: sMRI)
rootdir_per_scan_type = dict(TON={0: '',
                                  3: ('/data1/chdi_disks/Disk1/IBM_SOW3/'
                                      'TRACK/Imaging_data/TrackOn_DTI/DTI'),
                                  1: ('/data1/chdi_disks/Disk1/IBM_SOW3/'
                                      'TRACK/Imaging_data/TrackOn_sMRI'),
                                  2: ('/data1/chdi_disks/Disk2/'
                                      'IBM SOW3-part2/TrackOn/fMRI')},
                             THD={0: ('/data1/chdi_disks/Disk1/IBM_SOW3/'
                                      'TRACK/Imaging_data/TrackHD/DTI'),
                                  4: ('/data1/chdi_disks/Disk1/IBM_SOW3/'
                                          'TRACK/Imaging_data/TrackHD/DTI'),
                                  1: ('/data1/chdi_disks/Disk1/IBM_SOW3/'
                                      'TRACK/Imaging_data/TrackHD/sMRI'),
                                  2: ('/data1/chdi_disks/Disk1/IBM_SOW3/'
                                      'TRACK/Imaging_data/TrackHD/sMRI'),
                                  3: ('/data1/chdi_disks/Disk1/IBM_SOW3/'
                                      'TRACK/Imaging_data/TrackHD/sMRI')})
#rootdir_per_scan_type = dict(TON={2: ('/data1/chdi_disks/Disk4/TRACKON')})

class Subject:
    """
    Subject class that integrates all information about a subject (name,
    visits, elegibility data, imaging folders, analyses that have been
    performed) into a single object

    Parameters
    ----------
    subjid : str
        Subject ID
    general_df : `pandas.core.frame.DataFrame`
        Dataframe loaded from general_ton.csv
    mri_df : `pandas.core.frame.DataFrame`
        Dataframe loaded from mri.csv
    """

    def get_general_info(self, general_df, studyid='TON'):
        """
        Retrieve general information about the subjects from general_ton.csv

        Parameters
        ----------
        general_df: `pandas.core.frame.DataFrame` or `dict`
            Single dataframe with general csv file info or dictionary of
            dataframes from general.csv and general_ton.csv
        studyid: str, optional
            Specifies the study from which files will be retrieved. Valid
            values are 'THD' and 'TON'
        """
        if isinstance(general_df, dict):
            gen_df = general_df[studyid]
        else:
            gen_df = general_df
        if gen_df[gen_df.subjid == self.subjid].shape[0] != 1:
            raise ValueError(('The subject ID you requested ({}) is not'
                              'unique in the general database').
                             format(self.subjid))
        sg = gen_df[gen_df.subjid == self.subjid].iloc[0]
        self.studyid = studyid
        # Generalized assignment of Subject group attribute (absent on THD)
        if studyid == 'TON':
            self.group = ['control', 'preHD'][sg.group-1]
            self.inclusion_criteria = {'CAG_repeats': sg.ic4,
                                       'disease_burden': sg.ic5,
                                       'motorscores': sg.ic6,
                                       'good_age': sg.ic7}
            self.eligibility_criteria = sg[['ec1', 'ec2', 'ec3', 'ec4', 'ec5',
                                            'ec6', 'ec7', 'ec8', 'ec9', 'ec10',
                                            'ec11', 'ec12']].to_dict()
            self.eligible = sg.eligible
            hand_attr = 'handed'

        else:
            self.group = ['control', 'preHD', 'earlyHD'][sg.incl02]
            self.inclusion_criteria = {'CAG_repeats': sg.incl02c,
                                       'disease_burden': sg.incl02e,
                                       'motorscores': sg.incl02g,
                                       'good_age': not(sg.excl02 | sg.excl03)}
            self.exclusion_criteria = sg[['excl01', 'excl04', 'excl05',
                                          'excl06', 'excl07', 'excl08',
                                          'excl09', 'excl10', 'excl11',
                                          'excl12', 'excl13']].to_dict()
            hand_attr = 'handness'

        sg.fillna(value={hand_attr: 4}, inplace=True)
        sg.fillna(value={'ethnic': 7}, inplace=True)
        sg.fillna(value={'sex': np.nan}, inplace=True)
        sg.fillna(value={'age': np.nan}, inplace=True)
        ethnicity_dict = {1: 'caucassian', 11: 'african_black',
                          12: 'african_north', 13: 'asian_west',
                          14: 'asian_east', 15: 'mixed',
                          2: 'american_black', 3: 'american_latin',
                          6: 'other', 7: 'unknown'}
        self.demographics = {'age': sg.age,
                             'sex': sg.sex,
                             'ethnicity': ethnicity_dict[sg.ethnic],
                             'handness': ['right', 'left', 'mixed', 'unknown']\
                             [int(getattr(sg, hand_attr)) - 1]}

    def get_mri_info(self, mri_df):
        """
        Retrieve scan-related information from mri.csv

        Parameters
        ----------
        mri_df : `pandas.core.frame.DataFrame`
            Dataframe loaded from mri.csv

        """
        temp = mri_df[mri_df.subjid == self.subjid].copy(deep=True)
        if self.studyid == 'TON':
            # For TON the dictionary is defined by `subtype` as reported in the
            # document (augmented to include extra DTI scans -- blank on csv)
            mri_type_dict = {1: 'T1W', 2: 'T1W Repeat', 3: 'T2W',
                             4: 'Resting State', 5: 'WM Task', 6: 'Motor Task',
                             7: 'Practice', 8: 'Field Map', 9: 'Generic',
                             10: 'NODDI', 11: 'CEST/MTR', 12: 'DTI'}
            temp.fillna(value={'subytpe': 12}, inplace=True)
            temp['subytpe'] = temp['subytpe'].astype(int)
        else:
            # For THD the dictionary is defined by inspection of `mritype` on
            # the mri.csv spreadsheet. As consistent as possible with TON
            mri_type_dict = {0: 'DTI', 1: 'T1W', 2: 'T1W Repeat', 3: 'T2W',
                             4: 'DTI'}
            temp['subytpe'] = temp['mritype'].astype(int)

        temp.replace({'subytpe': mri_type_dict}, inplace=True)
        temp.set_index(['subytpe', 'visit'], inplace=True)
        temp.index.set_names('subtype', level=0, inplace=True)
        if not temp.index.is_lexsorted():
            temp = temp.sort_index()
        self.mri = temp
        return

    def get_subject_info(self, subj_df):
        """
        Retrieve general information of a participant that was not compiled at
        a specific visit, but rather once or in anually updated manner (from
        subject.csv)

        Parameters
        ----------
        subj_df : `pandas.core.frame.DataFrame`
            Dataframe loaded from subject.csv
        """
        ss = subj_df[subj_df.subjid == self.subjid]
        siteid = np.unique(ss.siteid.tolist())
        if len(siteid) != 1:
            raise ValueError(('Subject ID {} has different `site ids` for',
                              'TRACK and TrackOn').format(self.subjid))
        self.siteid = siteid[0]

    def __make_CAP_score_function(self, vd):
        """
        Estimates the visit_day to CAP score visit transformation, given the
        track_on_visit estimates of dbscore

        Parameters
        ----------
        vd : dict of `pandas.core.frame.DataFrame`
            Dictionary of visit_ton dataframes

        Returns
        -------
        CAP_dy_func: function
            Function that takes the day of a TRACK/TON visit and returns a CAP
        """
        tk = [vk for vk in sorted(vd.keys())
              if (('ton' in vk) and (self.subjid in vd[vk].subjid.values))]
#       tk = [df for if self.subjid in ] fix HERE
        if len(tk) >=2:
            dy_in = np.array([vd[vk][vd[vk]['subjid'] == self.subjid].visdy.iloc[0]
                              for vk in tk])
            db_in = np.array([vd[vk][vd[vk]['subjid'] == self.subjid].dbscore.iloc[0]
                              for vk in tk])

        try:
            ok_idx = ~np.isnan(db_in)
            x = dy_in[ok_idx]
            m, b = np.linalg.lstsq(np.stack((x, np.ones(x.shape)),
                                                axis=1), db_in[ok_idx])[0]
        except:
            m = b = np.nan
        CAP_dy_func = partial(linear_pred, m, b)
        return CAP_dy_func

    def get_pheno_vars(self, pheno_df):
        """
        Produces a datafram with phenotypic variables per visit

        Parameters
        ----------
        pheno_df : Pandas datafram
            Dataframe with phenotypic variables (e.g., one provided by SOW4)

        Returns
        -------
        CAP_dy_func: function
            Function that takes the day of a TRACK/TON visit and returns a CAP
        """
        df = pheno_df[pheno_df['subjid'] == self.subjid].copy(deep=True)
        vd_to_vis = {dv: dk for dk, dv in self.visdy.items()}
        df['visit'] = df['visdy'].replace(vd_to_vis)
        df.set_index('visit', inplace=True)
        self.pheno_df = df

    def get_cog_motor_performance(self, visit_dict):
        '''
        '''
        cog_motor_tasks = ['sdmt', 'stroop', 'paced_tap', 'indirect_circle_trace',
                           'map_search', 'cancelation', 'spot_change',
                           'mental_rotation', 'count_backwards', 'grip_var']
        field_list = cog_motor_tasks + ['visdy', 'visit']
        visits_used = self.visdy.keys()
        visits_used.sort(key=lambda x: x[::-1])
        all_vis_dfs = []
        for v_idx, visit in enumerate(visits_used):
            visit_df = visit_dict[visit]
            fields_in_dict = [fn for fn in field_list if fn in visit_df.columns]
            nan_fields = [fn for fn in field_list if fn not in visit_df.columns]
            vis_dict = visit_df[visit_df['subjid'] == self.subjid].iloc[0][
                fields_in_dict].to_dict()
            for field in nan_fields:
                vis_dict[field] = np.nan
            vis_dict['visit'] = visit
            vis_dict['v_idx'] = v_idx
            all_vis_dfs.append(vis_dict)
        out_df = pd.DataFrame(all_vis_dfs).set_index('v_idx')
        self.cog_motor_performance = out_df

    def __init__(self, subjid=None, general_df=None, mri_df=None,
                 subj_df=None, visit_df_dict=None, pheno_df=None,
                 studyid='TON'):
        # Subject.all_subjects.append(self)
        if subjid is not None:
            self.subjid = subjid
            if general_df is not None:
                self.get_general_info(general_df, studyid)
            if mri_df is not None:
                self.get_mri_info(mri_df)
            if subj_df is not None:
                self.get_subject_info(subj_df)
            if visit_df_dict is not None:
                self.CAP_from_visdy = self.__make_CAP_score_function(visit_df_dict)
                self.visdy = dict()
                self.CAP = dict()
                for vk, df in visit_df_dict.iteritems():
                    if self.subjid in df['subjid'].values:
                        vd = df[df['subjid'] == self.subjid]['visdy'].iloc[0]
                        self.visdy[vk] = vd
                        self.CAP[vk] = self.CAP_from_visdy(vd)
                self.get_cog_motor_performance(visit_df_dict)
            if pheno_df is not None:
                self.get_pheno_vars(pheno_df)
            #Continue here: make get_pheno_vars function, duplicate visdy col,
            #rename it and apply inverse dictionary

    def get_scan_dicom(self, mri_df=None, visit=None, subtype=None):
        """
        Retrieve list of dicom filenames (single dicom filename for each
        directory) where valid scans of the evaluated subject are located

        Parameters
        ----------
        mri_df : `pandas.core.frame.DataFrame`, optional
            Dataframe loaded from mri.csv
        visit : int, optional
            Integer value that specifies the visit number
        subtype : str, optional
            String that defines the type of image being queried (e.g., "T1W").
            For more infoirmation, please refer to
            "TRACK-IDS-2015-10-R1-DataDictionary(1).pdf", section 4.15 (MRI)

        Returns
        -------
        dcm_list : list
            list of single dicom filenames from directories where valid
            scans are located
        """
        if 'DTI' in subtype:
            if hasattr(subtype, 'extend'):
                subtype.extend('Generic')
            else:
                subtype = [subtype, 'Generic']

        if mri_df is None:
            mri_df = self.mri

        if visit is not None:
            visit_str = 'Visit ' + str(visit)
        else:
            visit_str = None

        idx = pd.IndexSlice
        if not mri_df.index.is_lexsorted():
            mri_df = mri_df.sort_index()

        used_df = mri_df.loc[idx[subtype, visit_str], :]

        dcm_list = []
        #from IPython.terminal.debugger import TerminalPdb; TerminalPdb().set_trace()

        for (scandy, mritype, subjid, scandesc,
             scanid, scanstatus, this_vst) in zip(
             used_df['scandy'], used_df['mritype'],
             used_df['subjid'], used_df['scandesc'], used_df['scanid'],
             used_df['scanstatus'],
             used_df.index.get_level_values('visit')):
                try:
                    scandesc = scandesc.replace(' ', '_')
                except:
                    scandesc = 'NO_SCAN_DESCRIPTION'
                dirlist = gl.glob('/'.join([rootdir_per_scan_type[self.studyid][mritype],
                                            subjid, scandesc,
                                            str(scan_year(this_vst,
                                                self.studyid)) + '*',
                                            'S' + str(scanid)]))
                cond = dirlist and scanstatus == 1
                if cond:
                    dcm_long_list = gl.glob('/'.join([dirlist[0], '*.dcm']))
                    cond = cond and dcm_long_list
                if cond:
                    dcm_list.append(dcm_long_list[0])
                else:
                    dcm_list.append('')
        return dcm_list

    def get_valid_visits(self, mri_df=None, subtype='T1W'):
        """
        Retrieve list of visits associated to a given subject where valid scans
        of a specific imaging subtype have been acquired. The output is an
        array of valid subject/visit pairs.

        Parameters
        ----------
        mri_df : `pandas.core.frame.DataFrame`, optional
            Dataframe loaded from mri.csv
        subtype : str, optional
            String that defines the type of image being queried. Default value
            is 'T1W'. For more infoirmation, please refer to
            "TRACK-IDS-2015-10-R1-DataDictionary(1).pdf", section 4.15 (MRI)

        Returns
        -------
        subj_vst_array : ndarray
            Aggregate array of valid subject/visit pairs
        """

        subjects_list = []
        visits_list = []

        if mri_df is None:
            mri_df = self.mri
        if subtype in mri_df.index.get_level_values('subtype'):
            used_df = mri_df.xs((subtype,), level=[0])
            for (scandy, mritype, subjid, scandesc,
                 scanid, scanstatus, this_vst) in zip(
                     used_df['scandy'], used_df['mritype'], used_df['subjid'],
                     used_df['scandesc'], used_df['scanid'], used_df['scanstatus'],
                     used_df.index.get_level_values('visit')):
                scandesc = scandesc.replace(' ', '_')
                dirlist = gl.glob('/'.join([rootdir_per_scan_type[self.studyid][mritype],
                                            subjid, scandesc,
                                            str(scan_year(this_vst,
                                                self.studyid)) + '*',
                                            'S' + str(scanid)]))
                cond = dirlist and scanstatus == 1
                if cond:
                    dcm_long_list = gl.glob('/'.join([dirlist[0], '*.dcm']))
                    cond = cond and len(dcm_long_list) > 0
                if cond:
                    subjects_list.append(subjid)
                    vst_nmb = int(this_vst.split(' ')[-1])
                    visits_list.append(vst_nmb)
            # correct for redundant list of visits
            inds = np.unique(visits_list, return_index=True)[1]
            visits_list = np.array(visits_list)[inds].tolist()
            subjects_list = np.array(subjects_list)[inds].tolist()

        subj_vst_array = np.array([subjects_list, visits_list])
        return subj_vst_array

def make_Track_ON_subjects(datadir, load_full_visit_forms=False):
    """
    Create list and dict of subjects from TRACK-ON study,
    together with relevant data frames.

    Parameters
    ----------
    datadir: str
        Specifies directory with csv files.
    load_full_visit_forms: bool, optional
        Specifies if Visit_X.csv files should be loaded. Generally,
        NOT recommmded since it makes things SLOW. Defaults to False.

    Returns
    -------
    subject_list: list
        Contains subject id strings.
    subjects_dict: dict
        Keys: subject id, values: Subject object.
    gton: `pandas.core.frame.DataFrame`
        Includes contents from general_ton.csv.
    mri_TON: `pandas.core.frame.DataFrame`
        Contains rows of mri.csv belonguing to the TRACK-ON study only.
    """
    cvs_names = [file for file in os.listdir(datadir) if file.endswith('.csv')]
    visit_ton_forms = ['visit1_ton', 'visit2_ton', 'visit3_ton']
    visit_track_forms = ['visit1', 'visit2', 'visit3', 'visit4']
    used_csv_list = ['general_ton', 'mri', 'subject']
    THD_cog_motor_dict = {'sdmt_correct': 'sdmt',
                          'swr_correct': 'stroop',
                          'ptap_3hz_alltrials_self_intertapinterval_stddev': 'paced_tap',
                          'circle_indirect_alltrials_annulus_length': 'indirect_circle_trace',
                          'msearch_totalcorrect_1minute': 'map_search',
                          #                         'cancelation': ''],
                          'spot_setsize5_k': 'spot_change',
                          'mrot_all_percentcor': 'mental_rotation',
                          'gripvarright': 'grip_var'}
                          #'count_backwards': ''}
    TON_cog_motor_dict = {'sdmt_correct': 'sdmt',
                          'stroop_correct': 'stroop',
                          'ptap_3hz_all_self_iti_sd': 'paced_tap',
                          'circle_ind_all_annulus_l': 'indirect_circle_trace',
                          'msearch_totcorr_1min': 'map_search',
                          'cancel_digit_totalcorrect_90s': 'cancelation',
                          'spot_setsize5_k': 'spot_change',
                          'mrot_all_percentcor':'mental_rotation',
                          'circle_cnt_direct_totalnumber': 'count_backwards',
                          'lhx_gf_rx_cvf': 'grip_var'}

#    if load_full_visit_forms:
#        used_csv_list.extend(visit_ton_forms)
#        used_csv_list.extend(visit_track_forms)
    # Make a dictionary of dataframes, one for each csv file:
    df_dict = {cvs_n.split('.')[0]: pd.read_csv(os.path.join(datadir, cvs_n),
               sep='\t') for cvs_n in cvs_names
               if cvs_n.split('.')[0] in used_csv_list}
    pheno_fn = os.path.join(datadir, 'track_pheno_data.csv')
    if os.path.isfile(pheno_fn):
        df_dict['track_pheno_data'] = pd.read_csv(pheno_fn, sep=',')
    if not load_full_visit_forms:
        visit_df_dict = {}
        for v_t in visit_track_forms:
            csv_used = os.path.join(datadir, v_t + '.csv')
            if v_t == 'visit1':
                used_cols = ['subjid', 'studyid', 'visit',
                             'visdy','caglarger__value']
            else:
                used_cols = ['subjid', 'studyid', 'visit', 'visdy']

            used_cols.extend(THD_cog_motor_dict.keys())
            with open(csv_used,'r') as f:
                head=f.readline()
            cols_in_file = head.split('\t')
            ok_cols = [col for col in used_cols if col in cols_in_file]
            visit_df_dict[v_t] = pd.read_csv(csv_used, sep='\t',
                                             usecols=ok_cols)
            visit_df_dict[v_t].rename(columns=THD_cog_motor_dict,
                                            inplace=True)
        for visit_ton in visit_ton_forms:
            csv_ton_used = os.path.join(datadir, visit_ton + '.csv')
            used_cols = ['subjid', 'studyid', 'visdy', 'dbscore']
            used_cols.extend(TON_cog_motor_dict.keys())
            with open(csv_ton_used,'r') as f:
                head=f.readline()
            cols_in_file = head.split('\t')
            ok_cols = [col for col in used_cols if col in cols_in_file]

            visit_df_dict[visit_ton] = pd.read_csv(csv_ton_used,
                                                   sep='\t',
                                                   usecols=ok_cols)
            visit_df_dict[visit_ton].rename(columns=TON_cog_motor_dict,
                                            inplace=True)
    else:
        long_visit_list = visit_track_forms.extend(visit_ton_forms)

        visit_df_dict = {cvs_n.split('.')[0]: pd.read_csv(os.path.join(datadir, cvs_n),
               sep='\t') for cvs_n in cvs_names if cvs_n.split('.')[0]
               in long_visit_list}
    gton = df_dict['general_ton']
    mri = df_dict['mri']
    subj_df = df_dict['subject']
    if 'track_pheno_data' in df_dict.keys():
        pheno_df = df_dict['track_pheno_data']
    else:
        pheno_df = None
#    visits_ton = {key: ton_df for key, ton_df in df_dict.iteritems()
#                  if key in visit_ton_forms}
    mri_TON = mri[mri.studyid == 'TON']  # dframe with TRACK-ON scans only
    subjects_dict = dict()
    subject_list = list()
    for subj_ix, subj_name in enumerate(gton['subjid']):
        subjects_dict[subj_name] = Subject(subj_name, gton, mri_TON, subj_df,
                                           visit_df_dict, pheno_df)
        subject_list.append(subj_name)
    return subject_list, subjects_dict, gton, mri_TON, visit_df_dict


def make_Track_subjects_subset(datadir, mristr=3., studyid='TON'):
    """
    Create list and dict of subjects from a given TRACK study (Track HD or
    Track On) and a specific field strength. It also outputs relevant data
    frames.

    Parameters
    ----------
    datadir: str
        Specifies directory with csv files.
    mristr: float, optional
        Specifies the field strength of the files to be retrieved.
    studyid: str, optional
        Specifies the study from which files will be retrieved. Valid values
        are 'THD' and 'TON'.

    Returns
    -------
    subject_list: list
        Contains subject id strings.
    subjects_dict: dict
        Keys: subject id, values: Subject object.
    gen_tk: `pandas.core.frame.DataFrame`
        Includes contents from appropriate Track study general csv file.
    mri_tk: `pandas.core.frame.DataFrame`
        Contains rows of mri.csv associated to the specified Track study only.
    """
    csv_names = [file for file in os.listdir(datadir) if file.endswith('.csv')]
    used_csv_list = ['general_ton', 'general', 'mri', 'subject']
    # Make a dictionary of dataframes, one for each csv file:
    df_dict = {cvs_n.split('.')[0]: pd.read_csv(os.path.join(datadir, cvs_n),
               sep='\t') for cvs_n in csv_names if cvs_n.split('.')[0]
               in used_csv_list}
    gen_tk = {key: df_dict[key] for key in used_csv_list[:2]}
    gen_tk['TON'] = gen_tk.pop('general_ton')
    gen_tk['THD'] = gen_tk.pop('general')
    mri_tk = df_dict['mri']
    # Retrieve info only from defined study of interest and field of strength
    mri_tk = mri_tk[mri_tk.studyid == studyid]
    mri_tk = mri_tk[mri_tk.mristr == mristr]
    subj_df = df_dict['subject']
    subjects_dict = dict()
    subject_list = list()
    subjects_ids = np.unique(mri_tk['subjid'])
    for subj_name in subjects_ids:
        subjects_dict[subj_name] = Subject(subj_name, gen_tk, mri_tk,
                                           subj_df, studyid=studyid)
        subject_list.append(subj_name)
    return subject_list, subjects_dict, gen_tk, mri_tk


def make_Track_ON_subjs_n_visits(datadir, subtype='T1W'):
    """
    Retrieve list of visits for which valid scans of an imaging subtype exist.
    This search is performed for all subjects listed in `mri.csv`. The output
    is an array of valid subject/visit pairs.

    Parameters
    ----------
    datadir: str
        Specifies directory with csv files.
    subtype : str, optional
        String that defines the type of image being queried. Default value
        is 'T1W'. For more information, please refer to
        "TRACK-IDS-2015-10-R1-DataDictionary(1).pdf", section 4.15 (MRI)

    Returns
    -------
    subj_vst_array : ndarray
        Aggregate array of valid subject/visit pairs
    """

    csv_names = [file for file in os.listdir(datadir) if file.endswith('.csv')]
    used_csv_list = ['general_ton', 'mri', 'subject']
    # Make a dictionary of dataframes, one for each csv file:
    df_dict = {csv_n.split('.')[0]: pd.read_csv(os.path.join(datadir, csv_n),
               sep='\t') for csv_n in csv_names if csv_n.split('.')[0]
               in used_csv_list}
    gton = df_dict['general_ton']
    mri = df_dict['mri']
    subj_df = df_dict['subject']
    mri_TON = mri[mri.studyid == 'TON']  # dframe with TRACK-ON scans only
    subj_visit_array = np.array([])
    subjects_ids = np.unique(mri_TON['subjid'])
    for subj_name in subjects_ids:
        subj_obj = Subject(subj_name, gton, mri_TON, subj_df)
        if subj_visit_array.size == 0:
            subj_visit_array = subj_obj.get_valid_visits(subtype=subtype)
        else:
            new_subj_visit = subj_obj.get_valid_visits(subtype=subtype)
            if new_subj_visit.size > 0:
                subj_visit_array = np.concatenate((subj_visit_array,
                                                   new_subj_visit), axis=1)
    return subj_visit_array
