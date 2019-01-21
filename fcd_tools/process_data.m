function process_data(out_fnm,dd,fd)

TON_resting_state_cd;

classifiers = {}

% ttest of the features

alpha=0.05;
results_file = 'ttest_allmaps_FDR05.txt';
mapid = fd.feature_type;
if isfield(fd,'use_subsample_mask') && ~isempty(fd.use_subsample_mask)
mask_fd=fd.use_subsample_mask;
else
mask_fd=[];
end
mask_finalZ=[];
if isfield(dd,'reg') && ~isempty(dd.reg)
    [tmap,zmap]=make_statistical_maps_corr_func(out_fnm,alpha,mapid,mask_fd, mask_finalZ,dd);
else
    [tmap,zmap]=make_statistical_maps_func(out_fnm,results_file,alpha,mapid,mask_fd, mask_finalZ,dd);
end

end

