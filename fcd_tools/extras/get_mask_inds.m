function [mask,inds] = get_mask_inds(dd)
% get the indices of nonzero mask voxels

chdi_home = dd.mypath;
subject_id pp
wd = fullfile(chdi_home,subject_id,['visit_' num2str(visit)],subtype);

fs = dir(fullfile(wd,'*.nii.gz'));
dl = {fs.name}';
di = strfind(dl, 'corrcomped');
used_file_idx = find(cell2mat(cellfun(@length,di,'UniformOutput',false)));
if length(used_file_idx)>1
    error(['More than one corrcomped vol found in ' wd])
elseif isempty(used_file_idx)
    error(['No corrcomped vol found in ' wd])
end

in_file = fullfile(wd,dl{used_file_idx})

mask = analyze75read(sprintf('%s%s',dd.mypath,dd.maskfile));
mask = permute ( mask,[2 1 3] );
mask = reshape(mask,1,[]);
  
inds = find(mask);% % % %   plot the data
 
end