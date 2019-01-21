close all; clear variables

do_save = false;
csv_fname= '/data2/polo/code/MLtool/TON_resting_classification/TON_info_for_MLtools.csv';
TON_cooked = '/data1/cooked/TONf';
subtype = 'Resting_State';
vol_suffix = 'realigned_2_MNI';

st = readtable(csv_fname);

subj_list = st.subject_IDs;
visits = [st.first_visit st.last_visit];
visits = mat2cell(visits,ones(size(visits,1),1),[size(visits,2)]);

template_vol_fname = '/data2/polo/code/MLtool/TON_resting_classification/masks/Resting_State_TON_mean_func.nii.gz';
tv = MRIread(template_vol_fname);
t_vol = tv.vol;

vol_present = ones(length(subj_list),length(visits{1}));
corr_coefs = nan(size(vol_present));

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
            in_vol = squeeze(mean(in_v.vol,4));
            cc=corrcoef(in_vol(:),t_vol(:));
            corr_coefs(s_idx,v_idx) = cc(1,2);
        end
    else
        for v_idx=1:length(visits{s_idx})
            vol_present(s_idx,v_idx) = 0;
        end
    end
        
end

log_corr=log10((1-corr_coefs));
hist(log_corr(:),30);


ml_corr=nanmean(log_corr(:));
sl_corr=nanstd(log_corr(:));

dev=abs(log_corr-ml_corr)/sl_corr>3;
[ix,jx] = find(dev); 

subjects_not_found = {subj_list{(vol_present(:,1) & vol_present(:,2)) == 0}}';
if do_save
    save('mask_outliers.mat','dev','vol_present','log_corr');
    dlmwrite('subjects_not_found.csv', subjects_not_found,'')
end


%%Separated by subject class
h1=log_corr(st.group==1,:);
h2=log_corr(st.group==-1,:);
bins=linspace(-1.9,-0.8,15);

figure;
subplot(2,1,1)
hist(h1(:),bins);
title('Correlation with mean functional by subject class TOP:pre-HD, BOTTON:controls')
ylabel('Counts')
ylim([0 60])
subplot(2,1,2)
hist(h2(:),bins);
ylabel('Counts')
xlabel('Deviation from perfect correlation (log units)')
ylim([0 60])


