function [mask,inds] = get_mask_inds(dd)

in_file = dd.maskfile;
mri = MRIread(in_file);
mask = logical(reshape(mri.vol, [1 dd.mySize]));
inds = find(mask);
end