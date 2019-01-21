% main script used for making degree maps

% set path
set_all_path;
load_data = true;
%%%%% Step 1: initialization  - load data descriptor from a file
TON_resting_state_dd; % load data descriptor dd
TON_resting_state_cd; % load classifier descriptor cd
do_regression = true; %To look at correlation with cognitive slopes
do_happy_sad = true; % makes analysis on preHD subjects only
regressor_name = 'PC_0'%'sdmt'%'PC_7';% 'sdmt';%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Specify feature types
%%%%%%%%%%%%%%%%%%%%%%%%%%%%se%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fd = struct('iind',[],'jind',[],'Seed',1,'precomputed',0,...
    'feature_type','bct_degrees',...
    'n_features', dd.nonzeros,...
    'longitudinal',false,...
    'use_log_scale',true,...
    'log_constant',1,...
    'normalize_data','median',...%
    'use_subsample_mask', []);

fmri_subjects = dd.subjects;

    % the name of the file to save the data in the standard ML (SML) format
    % where the rows are samples, columns are features (variables), and the
    % last column is the class label
    [out_fnm,err] = sprintf('%s_%s_%s.mat',dd.dataset,dd.task,fd.feature_type);
    if isfield(fd,'use_log_scale') && fd.use_log_scale
        out_fnm = [out_fnm(1:end-4) '_log' out_fnm(end-3:end)];
        disp('log_scale');
    end
    if isfield(fd,'normalize_data') % obs: this is applied before data substraction in longitudinal measure
        switch fd.normalize_data
            case 'mean'
                out_fnm = [out_fnm(1:end-4) '_norm_mean' out_fnm(end-3:end)];
            case 'max'
                out_fnm = [out_fnm(1:end-4) '_norm_max' out_fnm(end-3:end)];
            case 'median'
                out_fnm = [out_fnm(1:end-4) '_norm_median' out_fnm(end-3:end)];
        end
    end
    if isfield(fd,'longitudinal') && fd.longitudinal
        out_fnm = [out_fnm(1:end-4) '_longitudinal' out_fnm(end-3:end)];
    end    
    %%%%% Step 2: feature extraction
    %extract features and create a dataset in standard ML (SML) format
    
    if load_data
        
        out_fnm = './TON_log_deg_maps.mat'
        disp(['out filename has been forced to: ' out_fnm])
        load(out_fnm)
    else
        data = feature_extractor(dd,fd,out_fnm);
        save(out_fnm,'data','fmri_subjects');
    end
    if ~do_regression
        dd.reg = [];
    end
    
    if do_happy_sad
        data = make_happy_sad_data_and_regressors(data,dd,reg,regressor_name);
        %NOTE: THIS LABELS ARE placeholders, SO THE PAIRWISE COMPARISON IS A NO-GO IN HERE
        if do_regression
            data(:,end)=1;
            reg_fields = fieldnames(dd.reg);
            % Remove healthy subjects from regression values
            for ridx=1:length(reg_fields)
                dd.reg.(reg_fields{ridx}) = dd.reg.(reg_fields{ridx})(dd.labels==1,:);
            end
        else
            error('Use happy sad analysis only for correlation (i.e., regression)')
        end
        out_fnm = [out_fnm(1:end-4) '_happy_sad' out_fnm(end-3:end)];
        save(out_fnm,'data', 'fmri_subjects');
    end
    
    process_data(out_fnm,dd,fd);
    %end
    
