%%Take log of matrix:
    
%clear variables

mfile_dir = '/data2/polo/code/MLtool/TON_resting_classification/proc_data/non_smooth_non_subsamp/thres07';
%mfile=fullfile(mfile_dir,'TON_Resting_State_bct_degrees.mat');
%mfile_dir = '/data2/polo/code/MLtool/TON_resting_classification/proc_data/non_smooth_non_subsamp/clustering_coef';
%mfile=fullfile(mfile_dir,'TON_Resting_State_bct_degrees.mat');
%feat_used='bct_degrees';
%mfile='/data2/polo/code/MLtool/TON_resting_classification/proc_data/smoothed_subsamp/TON_Resting_State_bct_degrees_log.mat';
mfile='/data2/polo/code/TON_rsfmri_paper/VBM_controls/TON_log_deg_maps_local_gm_corrected_combat.mat'
load(mfile);

do_log=false;
using_log = ~isempty(strfind(mfile,'log_'));
if do_log
    zin=data(:,1:end-1);
    zin = log10(zin+1);
%    zin = zin./repmat(mean(zin,2),[1 size(zin,2)]);
    data = [zin data(:,end)];
    ofile=fullfile(mfile_dir,['TON_Resting_State_' feat_used '_log.mat']);
    if exist('fmri_subjects','var')
        save(ofile,'data', 'fmri_subjects');
    else
        save(ofile,'data');
    end
    using_log=true;
end
%% Smooth data: (no subsample)
do_smooth = true;
if do_smooth
    data= [out_data data(:,end)];
    md = '/data2/polo/code/MLtool/TON_resting_classification/proc_data/smoothed_no_subsamp/thres07';
    %md = '/data2/polo/code/MLtool/TON_resting_classification/proc_data/smoothed_non_subsamp/pagerank';
    if ~exist(md,'dir')
        mkdir(md)
    end
    if using_log
        ofile=fullfile(md, ['TON_Resting_State_' feat_used  '_log.mat']);
    else
        ofile=fullfile(md, ['TON_Resting_State_' feat_used '.mat']);
    end
    if exist('fmri_subjects','var')
        save(ofile,'data', 'fmri_subjects');
    else
        save(ofile,'data');
    end
else
    md=mfile_dir;
end

%% Normalize data
to_norm= data(:,1:end-1);
norm_types = {'max', 'mean', 'median'};
%norm_types = {'ranked'};
for nti=1:length(norm_types)
    switch norm_types{nti}
        case 'max'
            norms=max(to_norm,[],2);
        case 'mean'
            norms=mean(to_norm,2);
        case 'median'
            x=median(to_norm,2);
            norms=max(x,log10(2)*ones(size(x)));
        case 'ranked'
            [~, sorted_idx] = sort(to_norm, 2);
            norms=ones(size(to_norm,1),1);
    end
    normed_data = to_norm./repmat(norms,[1 size(to_norm,2)]);
    data = [normed_data data(:,end)];
    if strcmp(norm_types{nti},'ranked')
        data = [sorted_idx data(:,end)];
    end
    if using_log
        ofile=fullfile(md, ['TON_Resting_State_' feat_used '_log_norm_' norm_types{nti} '.mat']);
    else
        ofile=fullfile(md, ['TON_Resting_State_' feat_used '_norm_' norm_types{nti} '.mat']);
    end
    if exist('fmri_subjects','var')
        save(ofile,'data', 'fmri_subjects');
    else
        save(ofile,'data');
    end    
end

%% Make longitudinal degree index changes
mfile='/data2/polo/code/MLtool/TON_resting_classification/proc_data/non_smooth/TON_Resting_State_bct_degrees.mat';
load(mfile);
%[out_data] = smooth_data_vector(data(:,1:end-1),dd,fd,false,1.5);
%out_data = log10(data(:,1:end-1)+1);
out_data = data(:,1:end-1);
mo = mean(out_data, 2);
out_data = out_data./ repmat(mo, [1 size(out_data,2)]); 

odd_idx = 1:2:size(out_data,1); even_idx = 2:2:size(out_data,1);
odd_run = out_data(odd_idx,:); even_run =  out_data(even_idx,:);

change_data = (even_run-odd_run)./(even_run+odd_run+1);
%[change_data] = smooth_data_vector(change_data,dd,fd,false,.5);

y=data(odd_idx,end); mm=nan(size(change_data,2),1); pp=mm;
for cc=1:size(change_data,2)
    [tt,p0]=corrcoef(y,change_data(:,cc));
    mm(cc)=tt(1,2);
    pp(cc)=p0(1,2);
end
[p,h,stats]=signrank(mm); %median correlation of labels and longituinal tests is HIGHLY SIGNIFICANT
% z=15!!!

data = [change_data y];
ofile='/data2/polo/code/MLtool/TON_resting_classification/proc_data/smoothed_no_subsamp/TON_Resting_State_bct_degrees_longitudinal.mat';
save(ofile,'data');

% Weighted correlation votes: %This suggests weighted degree changes can
% provide a longitudinal classifier
%center=mean(change_data,1);
%centered_change=(2*[change_data>repmat(center,[size(change_data,1) 1])]-1).*repmat(mm',[size(change_data,1) 1]);
change_center = mean([mean(change_data(y==1,:)); mean(change_data(y==-1,:))]);
centered_change=([change_data-repmat(change_center,[size(change_data,1) 1])]).*repmat(mm',[size(change_data,1) 1]);
sum_centered = sum(centered_change,2);
[a,b]=corrcoef(sum_centered,y);

pre_HD=sum_centered(y==1);
controls=sum_centered(y==-1);
%%
figure
hist(mm,200);
lh=line([0 0 ], ylim, 'color', 'k')
lh.LineWidth =1;
lh=line([median(mm) median(mm) ], ylim, 'color', 'g')
lh.LineWidth =1;
xlabel('Correlations with class labels')
ylabel('Voxel counts')

figure;
h1=hist(pre_HD); hold on
pre_HD=sum_centered(y==1);
controls=sum_centered(y==-1);
figure;
hist(pre_HD); hold on
h = findobj(gca,'Type','patch');
h.FaceColor = [0 .5 .5];
hist(controls);
xlabel('Sum of degree change index, weighted by the correlation with labels')
legend({'PreHD','Controls'})


%% CORRELATION FEATURES:
%clear variables

mfile_dir = '/data2/polo/code/MLtool/TON_resting_classification/proc_data/subsamp/post_rigid_fMRI_2_T1_may_2016';
mfile=fullfile(mfile_dir,'TON_Resting_State_all_corr_smooth.mat');
%mfile='/data2/polo/code/MLtool/TON_resting_classification/proc_data/smoothed_subsamp/TON_Resting_State_bct_degrees_log.mat';
load(mfile);
zin=data(:,1:end-1);
%zin = atanh(zin);
zin = zin./repmat(mean(zin,2),[1 size(zin,2)]);
data = [zin data(:,end)];
ofile=fullfile(mfile_dir,'TON_Resting_State_all_corr_smooth_norm_mean.mat');
save(ofile,'data');