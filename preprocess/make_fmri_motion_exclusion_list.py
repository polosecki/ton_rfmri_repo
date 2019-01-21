from glob import glob
from os import path as op
import pandas as pd

'''
THIS SCRIPT LOOKS AT INTENSITY AND RMS DISPLACEMENT FILES IN EVERY TRACK-ON
VISIT, IDENTIFIES OUTLIERS AND SAVES THEM TO CSV FILE
'''

do_save = False
ton_dir = '/data1/cooked/TONf'
visit_list = [1, 2, 3]
subtype = 'Resting_State'
outlier_frames_thres = 10
outlier_run_csv_fn = ('/data2/polo/half_baked_data/TON_rsfMRI/'
                      'outlier_runs.csv')
subj_list = [op.split(dd)[-1] for dd in glob(ton_dir + '/R*')]
subj_list.sort()


ol_numbers = []
subjs = []
visits = []

for s_ix in subj_list:
    print(s_ix)
    for v_ix in visit_list:
        s_v_dir = op.join(ton_dir, s_ix, 'visit_{}'.format(v_ix), subtype)
        if op.isdir(s_v_dir):
            file_template = '_'.join([s_ix, 'visit_{}'.format(v_ix), subtype,
                                      'S??????', '{}']) + '.{}'
            outlier_files = glob(op.join(s_v_dir,
                                         file_template.format('outliers',
                                                              'csv')))
            fn_idx = 0
            ol_fn = outlier_files[fn_idx]
            this_out = pd.read_csv(ol_fn)
            ol_n = (this_out['intensity'] | this_out['motion']).sum()
            ol_numbers.append(ol_n)
            subjs.append(s_ix)
            visits.append(v_ix)

ol_df = pd.DataFrame.from_dict({'subjid': subjs,
                                'visit': visits,
                                'n_outliers': ol_numbers})

outlier_runs = ol_df[ol_df.n_outliers > 10]

if do_save:
    exclusion_list = outlier_runs.to_csv(outlier_run_csv_fn)
