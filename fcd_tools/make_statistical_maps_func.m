function [tmap,zmap]=make_statistical_maps_func(inf,resf,alpha,mapid,fd_mask, mask_finalZ,dd)


statistical_test = 'wilcoxon'; %'ttest';
[~,a,~]=fileparts(inf)
out_dir = fullfile('.','figures',a);

if ~exist(out_dir,'dir')
    mkdir(out_dir)
end
if isfield(dd,'reg') && ~isempty(dd.reg)
    statistical_test='correlation';
end



if ~isempty(fd_mask)
    dd.maskfile = fd_mask;
    h=MRIread(dd.maskfile);
    dims = size(h.vol);
    dd.dimX = dims(1); dd.dimY = dims(2); dd.dimZ = dims(3);
    dd.mySize = prod(dims(1:3));
    dd.voxel_size = [h.xsize h.ysize h.zsize];
    clear h
end

if  ~isempty(mask_finalZ)
    warning('zmap mask use not currently implemented')
    pause
end

is_long_data = ~isempty(strfind(inf,'longitudinal'));
is_log_scale = ~isempty(strfind(inf,'log'));
if is_long_data
    suffix_idx = 2;
    disp('Longitudinal!')
else
    suffix_idx = 1;
end
suffixes={'','_longitudinal'};
suffix_used = suffixes{suffix_idx};

if  is_log_scale
    suffix_used = [suffix_used '_log'];
end


h=load(inf);
data = h.data; clear h

[n,m_max]=size(data);
[mask,inds] = get_mask_inds(dd);

%RECOVER FEATURE MATRIX
if m_max > length(inds)+1
    m = data(:,1:(end-1)); %DON'T MAP TO BRAIN MASK, BUT REMOVE LABELS
elseif m_max == length(inds)+1
    m=zeros(n,length(mask)); %MAKE FEATURE MATRIX WITH ZEROS IN MASKED IDX
    m(:,inds) = data(:,1:(end-1)); %DON'T INCLUDE LABELS
else % non-mappable  features, e.g. pairwise correlations
    error('You have less features than brain voxels, bad')
end


m= m';
y = data(:,end);

dimX = dd.dimX; dimY = dd.dimY; dimZ = dd.dimZ;

% GET INDICES OF EACH SUBJECT CLASS
ind_s = find(y==1);
ind_n= find(y==-1);

mp = m;
HuntP=nanmean(mp(:,ind_s)')'; % MEAN CLASS 1
nrmP = nanmean(mp(:,ind_n)')'; % MEAN CLASS 2
dfrP=HuntP-nrmP; %DIFFERENCE
tmap=ones(size(nrmP));
zmap=zeros(size(nrmP));

isIn=nanmean(abs(mp),2)>0;
for i=1:size(m,1),
    if isIn(i)
        switch statistical_test
            case 'ttest'
                [h,p,ci,stats] = ttest2(mp(i,ind_n),mp(i,ind_s));
                tmap(i)=(p);
                zmap(i)=dfrP(i)/stats.sd; % stats.df;
            case 'wilcoxon'
                [p,h,stats] = ranksum(mp(i,ind_n),mp(i,ind_s));
                tmap(i)=(p);
                zmap(i)=sign(dfrP(i))*abs(stats.zval); % stats.df;
            case 'correlation'
                [rho,p] = corr(mp(i,ind_s)',dd.reg(ind_s));
                tmap(i)=p;
                zmap(i)=rho;
        end
        if mod(i,1000)==0
            
            disp([num2str(i),' ',num2str(p)]);
            
        end
        
        
    end;
end

figure;
hist(tmap,100)
line([.05 .05], ylim,'color','k')
title('Distribution of pvals')
xlabel('Pvals, (-log10)')
set(gca','XScale','log')

%%%%%%%% FDR CALCULATION

%Holm-Bonferroni Correction:
HBp = zeros(size(tmap));
HBp(isIn) = frmrHolmBonferoni(tmap(isIn));

VX=find(isIn);
[h, crit_p, adj_p]=fdr_bh(tmap(isIn),alpha);
%[h, crit_p]=fdr_bky(tmap(isIn),alpha);
disp(['Voxels passed FDR: ' num2str(nnz(h))])
fdr_tmap = ones(size(tmap));
fdr_zmap = zeros(size(zmap));
if any(h)
    fdr_tmap(VX(h)) = tmap(VX(h));
    fdr_zmap(VX(h)) = zmap(VX(h));
end

% Actual number of voxels considered
% we may use a mask instead
PX=tmap(VX);
PVX=[PX VX];
% This has p-vals in first col,
% original voxel index in second
sortedPVX=sortrows(PVX,1);

x=1:length(PX);

fh=figure;
loglog(x,sortedPVX(:,1));
hold on
loglog(x,0.05*x/length(PX));

title(sprintf('P values: map %s',mapid),'fontsize',12);
outfig1 = sprintf('pvals_%s.fig',mapid);
outeps1 = sprintf('pvals_%s.eps',mapid);

saveas(fh,fullfile(out_dir,outfig1),'fig');
saveas(fh,fullfile(out_dir,outeps1),'psc2');

% Now finalZ has z-vals over confident voxels
disp(['Survive FDR: ',num2str(nnz(h)),', Total: ',num2str(nnz(isIn))]);


template = MRIread(dd.maskfile)

tmap(tmap==0)=eps(0);
pmap1 = -log10(tmap).*sign(zmap);


fdr_pmap = -log10(fdr_tmap).*sign(zmap);

pmri = template;
pmri.vol = reshape(pmap1, pmri.volsize);
pmri_fname = sprintf('%s%s_pmap.nii.gz',mapid,suffix_used);
MRIwrite(pmri,fullfile(out_dir,pmri_fname));

zmri = template;
zmri.vol = reshape(zmap, zmri.volsize);
zmri_fname = sprintf('%s%s_zmap.nii.gz',mapid,suffix_used);
MRIwrite(zmri, fullfile(out_dir,zmri_fname));

HBpmri = template;
HBpmri.vol = reshape(HBp, HBpmri.volsize);
HBPmri_fname = sprintf('%s%s_pmap_HBonfCorrected.nii.gz',mapid,suffix_used);
MRIwrite(HBpmri, fullfile(out_dir,HBPmri_fname));

FDRpmri = template;
FDRpmri.vol = reshape(fdr_pmap, FDRpmri.volsize);
FDRpmri_fname = sprintf('%s%s_pmap_FDRmasked.nii.gz',mapid,suffix_used);
MRIwrite(FDRpmri, fullfile(out_dir,FDRpmri_fname));

FDRzmri = template;
FDRzmri.vol = reshape(fdr_zmap, FDRzmri.volsize);
FDRzmri_fname = sprintf('%s%s_zmap_FDRmasked.nii.gz',mapid,suffix_used);
MRIwrite(FDRzmri, fullfile(out_dir,FDRzmri_fname));
