function [out_data] = smooth_data_vector(data,dd,fd,do_subsample,fwhm)
%smoothes and subsamples data vector by converting to 3d and back to 2d

if nargin==3
    do_subsample=true;
end
if nargin<5
fwhm = 1.5;
end
%dv = data(1,1:end-1);
out_data = zeros(size(data));
[mask,inds] = get_mask_inds(dd);
edv = double(mask);

mMRI=MRIread(dd.maskfile);

for d_idx=1:size(data,1)
    dv = data(d_idx,:);
    edv(inds) = dv;
    dvol = reshape(edv,[dd.dimX dd.dimY dd.dimZ]);
    
    
    sigma = fwhm/(2*sqrt(2*log(2)));
    sdvol = imgaussfilt3(dvol, sigma);
    
    % h1=permute(shiftdim(dvol,-1),[2 3 1 4]);
    % figure;
    % montage(h1,colormap)
    %
    % h2=permute(shiftdim(sdvol,-1),[2 3 1 4]);
    % figure;
    % montage(h2,colormap)
    
    if do_subsample
        mMRI.vol = sdvol;
        temp_fname = sprintf('./tt_%d.nii',round(1e6*rand));
        MRIwrite(mMRI,temp_fname);
        
        temp_fname2 = sprintf('./tt_%d.nii.gz',round(1e6*rand));
        cmd_str='fslmaths %s -subsamp2 %s';
        system(sprintf(cmd_str,temp_fname,temp_fname2),'-echo');
        rv = MRIread(temp_fname2);
        delete(temp_fname,temp_fname2);
        
        mri = MRIread(fd.use_subsample_mask);
        mask = reshape(mri.vol, [1 prod(mri.volsize)]);
        flat_rv = reshape(rv.vol, [1 prod(mri.volsize)]);
        inds = find(mask);
        %out_data = zeros(size(mask));
        out_data(d_idx,:) = flat_rv(inds);
    else
        hh = reshape(sdvol,[1 dd.mySize]);
        out_data(d_idx,:) = hh(inds);
    end
end