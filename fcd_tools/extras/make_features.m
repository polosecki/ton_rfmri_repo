function [X] = make_features(dd,fd,subj,run)

% fd - stucture containing feature description
% subj - subject index in the array dd.subjects
% run - run ID


% if fd.precomputed  % precomputed features for given subject/run
%     X = get_precomputed_features(dd,fd,subj,run);
%     return;
% end


%%%%
switch dd.dataset
    case 'TON'
        data = read_data_TON_rsfmri(dd, subj, run);
    otherwise
        error('Specify a data loadling function')
end

if isempty(data) % couldn't read the file for some reasons
    X=[];
    return;
end

z=sum(mean(data,1)==0);
if z>0% find(~data)
    fprintf('zeros in data on %d voxels',z);
    %    return;
end


% faster way of computing correlation matrix
% the standard corrcoef is slower and does not handle 0 time series
% properly (returns NaNs)
disp('corr_matrix');
C = my_corrcoef(data);


%save(sprintf('Cmat_%d_%d.mat',subj,fonc), 'corr_mat');

clear data;

[nRows,nCols] = size(C);
C(1:(nRows+1):nRows*nCols) = 0;

display('feature construction')
% construct features from the corelation matrix - use feature_type
switch lower(fd.feature_type)
    case 'bct_degrees'
        X = full(degrees_und(abs(C) > dd.corr_thresh));
        
    otherwise
        
end


clear C;
clear abs(C);
