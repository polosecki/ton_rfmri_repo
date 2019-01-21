function data = feature_extractor(dd,fd,out_fnm)
% exttract features from the data, and combine them with the labels
% save in the standard ML format (rows - samples, columns - features, last
% column - class label)

% fd - 'feature descriptor - structure containing all info about type of the features
% dd - 'data descriptor' - structure contaning description of the data

% dd.subjects - array of subject ID's (first column) and labels (second column)
% dd.runs - the number of runs

t=cputime; i=0; missing_subjects = []; dat = [];
fnm = ['current_' out_fnm];


X=[]; dat = [];
ok_subj = ones(size(dd.subject_IDs));

for j=1:length(dd.subjects)
    runs_per_subject = 0;
    for run = 1:length(dd.runs{j})
        if isfield(fd,'longitudinal') && fd.longitudinal && length(dd.runs{j})~=2
            warning(['Number of runs in subject ' dd.subject_IDs{j} ' is not 2'...
                'but required feture is longitudinal.'])
            break
        end
            dat = make_features(dd,fd,dd.subject_IDs{j},dd.runs{j}(run)); % create row of features for j-th subject
        if isempty(dat) %could not read file of that subject and and that run
            continue;
        else
            if isfield(fd,'use_log_scale') && fd.use_log_scale
                dat = log10(dat + fd.log_constant);
                disp('data was log_scaled');
                
            end
            if isfield(fd,'use_subsample_mask') && ~isempty(fd.use_subsample_mask)
                [dat] = smooth_data_vector(dat,dd,fd);
                disp('data was decimated');
            end
            if isfield(fd,'normalize_data') % obs: this is applied before data substraction in longitudinal measure
                switch fd.normalize_data
                    case 'mean'
                        nf=mean(dat);
                    case 'max'
                        nf=max(dat);
                    otherwise
                        nf=1;
                end
                dat=dat./nf;
            end
            i = i+1;
            runs_per_subject = runs_per_subject + 1;
            X(i,:) = dat;
            y(i) = dd.labels(j);
            s_ID(i) = dd.subject_IDs(j);
            s_name(i,:) = dd.subjects(j);
            s_run(i) = run;
        end
    end
    if isfield(fd,'longitudinal') && fd.longitudinal && runs_per_subject==2
        X(end-1,:) = diff(X(end-1:end,:),1);
        X(end,:) = [];
        y(end)=[];
        s_ID(end) = [];
        s_name(end)=[];
        s_run(end)=[];
        i=i-1;
    end
    save(fnm,'X','y', 's_ID', 's_name', 's_run','dd','fd');
    if ~runs_per_subject % no runs found for this subject
        % delete this subject from the list
        missing_subjects = [missing_subjects dd.subject_IDs(j)];
        ok_subj(j) = -1;
        save('missing_subjects.mat','missing_subjects');
        
    end
end

% if any subjects must be deleted because their data could not be read (marked as -1), do it now:
inds = find(ok_subj ~= -1);
if length(inds) < length(dd.subject_IDs) % some subjects deleted
    warning('For some subjects there is incomplete data')
    dd.subject_IDs = dd.subject_IDs(inds);
    dd.labels = dd.labels(inds);
end


display 'feature extraction: elapsed time'; cputime-t

data = [X y'];




