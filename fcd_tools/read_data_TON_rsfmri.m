function [data,inds] = read_data_TON_rsfmri(dd, subject_id, visit)
% READ DATA FROM A GIVEN SUBJECT AND RUN (RUN=VISIT)
% RETURNS DATA MATRIX IF VOLUME FOUND, OTHERWISE, RETURNS EMPTY
%
% INPUTS
%subject_id: subject dir name (string)
%

[in_file] = get_fname(dd, subject_id, visit);

data=[];
inds = [];

mri = MRIread(in_file);
if isempty(mri)
    warning(['Volume not found. Subj: ' subject_id ' visit: ' num2str(visit)])
    pause
    return;
else
    ds = size(mri.vol);
    if ds(4)~=dd.TRs
        warning(['Wrong number of TRs Subj: ' subject_id ' visit: ' num2str(visit) ' TRs: ' num2str(ds(4))])
    pause
    end
    nvox_diff = [ds(1)-dd.dimX ds(2)-dd.dimY ds(3)-dd.dimZ];
    vox_size_diff = [mri.xsize-dd.voxel_size(1)
                     mri.ysize-dd.voxel_size(2)
                     mri.zsize-dd.voxel_size(3)];
    disp('Size discrepancy:')
    disp(vox_size_diff)
    if any(nvox_diff)        
        warning(['Wrong vol size. Subj: ' subject_id ' visit: ' num2str(visit)])
        disp(nvox_diff)
%        pause
        return
    end
    data=reshape(mri.vol, [prod(ds(1:3)), ds(4)])'; 
    if max(abs(data)) == 0 
        display(sprintf('all-zero map in %s',infile));
        return; 
    end

    [~, inds] = get_mask_inds(dd);  % get indices of nonzero entries;
   
%   plot the data
%   M1=reshape(mask,[53 63 46]);
%   for i=1:46
%         subplot(7,7,i); pcolor( (M1(:,:,i)).^10 ); shading flat; %pause(0.05); 
%         % pcolor( (M1(:,:,i)).^10 ); shading interp; pause(0.05); 
%   end
 
%    data = data.*repmat(mask,1,TRs);
 
%     M1=reshape(data(:,1),[53 63 46]);
%     for i=1:46
%         subplot(7,7,i); pcolor( (M1(:,:,i)).^10 ); shading flat; %pause(0.05); 
%         % pcolor( (M1(:,:,i)).^10 ); shading interp; pause(0.05); 
%     end
%     
    
    data = data(:,inds);  % rows- samples, columns - voxels
    
end

