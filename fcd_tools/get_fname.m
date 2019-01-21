function [in_file] = get_fname(dd, subject_id, visit) 
% Find the filename of the volume correspoding to a certain subject and
% visit, given the data descriptor dd
%

chdi_home = dd.mypath;
subtype = dd.task;
vol_suffix  = dd.vol_suffix;

wd = fullfile(chdi_home,subject_id,['visit_' num2str(visit)],subtype);

fs = dir(fullfile(wd,'*.nii.gz'));
dl = {fs.name}';
di = strfind(dl, vol_suffix);
used_file_idx = find(cell2mat(cellfun(@length,di,'UniformOutput',false)));
if length(used_file_idx)>1
    error(['More than one ' vol_suffix ' vol found in ' wd])
elseif isempty(used_file_idx)
    warning(['No ' vol_suffix ' vol found in ' wd])
    in_file = ['none.nii'];
    return
end

in_file = fullfile(wd,dl{used_file_idx});