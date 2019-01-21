
%Data description for TRACK-ON data

%% Define Regressors to be used for voxelwise correlation maps
%reg_used = {'sdmt','cancelation', 'grip_var', 'paced_tap', 'map_search', 'stroop', 'spot_change', 'cancelation', 'mental_rotation', 'indirect_circle_trace', 'grip_var', 'count_backwards'};
reg_used = {'PC_0', 'PC_1', 'PC_2', 'PC_3', 'PC_4', 'PC_5', 'PC_6', 'PC_7', 'PC_8', 'PC_9'};
constant_control_regressors={'age', 'sex', 'CAP', 'CAG'};
control_var_used={};

%% Path to functional brain masl
func_mask_path = ['/data2/polo/code/MLtool/'...
  'TON_resting_classification/masks/Resting_State_TON_mask_strict.nii.gz'];

%% Load csv with  subject data:

csv_fname= 'TON_info_for_MLtools.csv';
st = readtable(csv_fname);
hh =load('mask_outliers.mat');
non_outliers = ~any(hh.dev,2) & all(hh.vol_present,2);
st=st(non_outliers,:);

in_csv_fn = '/data2/polo/half_baked_data/slopes/abs/raw_slopes_ok_subjs_abs.csv';
tab_data = readtable(in_csv_fn);
acceptable_behavior_subjects = tab_data.subjid;
filter_by_behavior_files = true;

behavior_file = '/data2/polo/half_baked_data/slopes/abs/deCAPed_pc_vals_abs.csv';
behav = readtable(behavior_file);

dd = struct('dataset','TON',...
            'task','Resting_State',... % task name
            'mypath','/data1/cooked/TONf',...            
             'subjects',[],... %PP defined below
            'labels',[],...
            'maskfile',func_mask_path,...
            'nonzeros',0,... % PP Taken from mask
            'corr_thresh',0.7,...
            'vol_suffix','corrcomped_2_MNI.nii.gz',...
            'Fisher',false);

visits = [st.first_visit st.last_visit];


if filter_by_behavior_files
    subj_is_acceptable = ismember(st.subject_IDs, tab_data.subjid) &...
        diff(visits,1,2)~=0;
else
    subj_is_acceptable = diff(visits,1,2)~=0;
end

subj_subset = find(subj_is_acceptable);



dd.subject_IDs = st.subject_IDs(subj_subset) ;
dd.subjects = dd.subject_IDs;

tab_acceptable_subj_idx =nan(size(dd.subject_IDs));
behav_subj_idx = nan(size(dd.subject_IDs));
for i=1:length(dd.subject_IDs)
    z= find(strcmp(dd.subject_IDs{i},tab_data.subjid));
    if ~isempty(z)
        tab_acceptable_subj_idx(i) = z;
    end
    z= find(strcmp(dd.subject_IDs{i},behav.subjid));
    if ~isempty(z)
        behav_subj_idx(i) = z;
    end
end

visits = mat2cell(visits,ones(size(visits,1),1),[size(visits,2)]);
dd.visits = visits(subj_subset); dd.runs = dd.visits;

% Read volume dimensions:
some_vol = get_fname(dd,dd.subject_IDs{1},dd.visits{1}(1));
some_vol = MRIread(some_vol);
dims = size(some_vol.vol);
dd.dimX = dims(1); dd.dimY = dims(2); dd.dimZ = dims(3); dd.TRs = dims(4);
dd.mySize = prod(dims(1:3));

dd.voxel_size = [some_vol.xsize some_vol.ysize some_vol.zsize];
clear('some_vol');



dd.labels =  st.group(subj_subset);  





% Make control regressors:
for i=1:length(constant_control_regressors)
    name = constant_control_regressors{i};
dd.ctrl_regressors.(name) = ...
    [table2array(tab_data(tab_acceptable_subj_idx,{name})) table2array(tab_data(tab_acceptable_subj_idx,{name}))]; % work here
end

for i=1:length(reg_used)
    name = reg_used{i};
    z = nan(length(dd.labels),2);
    z_idx = ~isnan(behav_subj_idx);
    for j=1:size(z,2)
        z(z_idx,j) = table2array(behav(behav_subj_idx(z_idx), name));
    end
    reg.(name)= z;
end


dd.reg = reg;


[mask,inds] = get_mask_inds(dd);  % get indices of nonzero entries;
dd.nonzeros = nnz(mask);
