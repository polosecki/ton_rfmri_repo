function  inter = plot_stability(cd,fid)
% cd - classifier descriptor

alg = cd.alg; K=cd.top_K; method = cd.method;
lambda1 = cd.lambda1; lambda2 = cd.lambda2;
 
sel_type = cd.sel_type; thresh = cd.thresh; 

infname = cd.fnm;     

 s_lambda1 = ''; s_lambda2='';
    
    if cd.lambda1
        s_lambda1=sprintf('_lam1_%.3f',lambda1);
    end
    if cd.lambda2
        s_lambda2=sprintf('_lam2_%.3f_',lambda2);
    end
    
outfig_all = sprintf('Figures/%s_%s%s%strain_%s.fig',alg,method,s_lambda1,s_lambda2,sel_type);
outeps_all = sprintf('Figures/%s_%s%s%strain_%s.eps',alg,method,s_lambda1,s_lambda2,sel_type); 


plot_sym = {'bo-';'b--';'bs-';'m-';'k--';'r--';'g--';'ro-';'gv-'};

folds = cd.folds;



outfig = sprintf('stability_frac_%s_%s.fig',sel_type,infname{1});  
outeps = sprintf('stability_frac_%s_%s.eps',sel_type,infname{1}); 

% outfig1 = sprintf('stability_frac_300%s%d.fig',sel_type,K(length(K)));  
% outeps1 = sprintf('stability_frac_300%s%d.eps',sel_type,K(length(K))); 
% 
% outfig2 = sprintf('stability_frac_1000%s%d.fig',sel_type,K(length(K)));  
% outeps2 = sprintf('stability_frac_1000%s%d.eps',sel_type,K(length(K))); 



inter = zeros(length(infname),length(K));
for i = 1:size(infname,1)
    infname{i} 
   
    fnm = infname{i};
    fnm(strfind(fnm,'_')) = '-';   

    for tf = 1:size(K,2)
      top_features = K(tf);
 
       fold=1;   
       if thresh
          rank_fnm = sprintf('ranking/%s%.2f_fold%d_out_%d_%s',sel_type,thresh,fold,folds,infname{i});
        else
          rank_fnm = sprintf('ranking/%s_fold%d_out_%d_%s',sel_type,fold,folds,infname{i});
       end
       load(rank_fnm);
       
       common_features =  ranked_features(1:top_features);
       clear ranked_features;
      
       for fold=2:folds       
           
           if thresh
                rank_fnm = sprintf('ranking/%s%.2f_fold%d_out_%d_%s',sel_type,thresh,fold,folds,infname{i});
           else
                rank_fnm = sprintf('ranking/%s_fold%d_out_%d_%s',sel_type,fold,folds,infname{i});
           end
           load(rank_fnm);
       
           common_features = intersect(common_features,ranked_features(1:top_features)); 
           clear ranked_features;
       end
       inter(i,tf) = size(common_features,2);
    end
    
    figure(fid);
    plot(K, inter(i,:)./K, plot_sym{i});
    hold on;
    ylabel('% features in common','FontSize',12); xlabel('top voxels','FontSize',12); 
    [s,e]=sprintf('Common features over %d CV folds:  \n %s', folds, fnm); 
         
    title(s,'fontsize',12);  

    

end
 
  legend(cd.legend);
  saveas(gcf,outfig,'fig'); 
     saveas(gcf,outeps,'psc2');  
 
% xlim([0 300]);  
% saveas(gcf,outfig1,'fig'); 
% saveas(gcf,outeps1,'psc2');  
% 
% xlim([0 1000]);  
% saveas(gcf,outfig2,'fig'); 
% saveas(gcf,outeps2,'psc2'); 





%%%%%%%

 