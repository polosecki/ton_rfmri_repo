close all; clear variables

csv_fname= '/data2/polo/code/MLtool/TON_resting_classification/TON_info_for_MLtools.csv';
TON_cooked = '/data1/cooked/TONf';
subtype = 'Resting_State';
vol_suffix = 'realigned_2_MNI';
strict_anatomical_mask = '/data1/standard_space/MNI152_T1_3mm_strict_mask.nii.gz';
st = readtable(csv_fname);

subj_list = st.subject_IDs;
visits = [st.first_visit st.last_visit];
visits = mat2cell(visits,ones(size(visits,1),1),[size(visits,2)]);

template_vol_fname = '/data1/standard_space/MNI152_T1_3mm.nii.gz';
tv = MRIread(template_vol_fname);

sink_v = zeros(size(tv.vol));


% s_idx = 1;
% v_idx = 1;
vol_present = ones(length(subj_list),length(visits{1}));

for s_idx =1:length(subj_list)
    disp(s_idx)
    if visits{s_idx}(1) ~= visits{s_idx}(2)
        for v_idx=1:length(visits{s_idx})
            fdir = fullfile(TON_cooked,subj_list{s_idx},['visit_' num2str(visits{s_idx}(v_idx))],subtype);
            fs = dir(fullfile(fdir,'*.nii.gz'));
            dl = {fs.name}';
            di = strfind(dl, vol_suffix);
            used_file_idx = find(cell2mat(cellfun(@length,di,'UniformOutput',false)));
            if length(used_file_idx) ~=1
                vol_present(s_idx,v_idx) = 0;
                continue
            end
            fname = fullfile(fdir,dl{used_file_idx});
            in_v = MRIread(fname);
            sink_v = sink_v + mean(in_v.vol,4);
        end
    else
        for v_idx=1:length(visits{s_idx})
            vol_present(s_idx,v_idx) = 0;
        end
    end
end

mask_v = sink_v > mean(sink_v(:));

mask_dir = './masks';
if ~exist(mask_dir,'dir')
    mkdir(mask_dir)
end

out_vol_fname = fullfile(mask_dir,[subtype '_TON_mask.nii.gz']);
out_mri = tv;
out_mri.vol = logical(mask_v);
MRIwrite(out_mri,out_vol_fname,'int');

%Make smooth mask by mohpological opening and closing:
smoothed_mask_fname = fullfile(mask_dir,[subtype '_TON_mask_smoothed.nii.gz']);
cmd_str = 'fslmaths -dt input %s -kernel sphere 12 -ero -dilD -dilD -ero %s';
system(sprintf(cmd_str,out_vol_fname,smoothed_mask_fname));

%Make strcit mask by multiypling mask with a strict anatomical one:
strict_anat_mask = MRIread(strict_anatomical_mask);
out_mri = tv;
out_mri.vol = logical(strict_anat_mask.vol .* mask_v);
out_vol_fname = fullfile(mask_dir,[subtype '_TON_mask_strict.nii.gz']);
MRIwrite(out_mri,out_vol_fname,'int');


%Save mean volume:
out_vol_fname = [subtype '_TON_mean_func.nii.gz'];
out_mri = tv;
out_mri.vol = sink_v./nnz(vol_present);
MRIwrite(out_mri,out_vol_fname);