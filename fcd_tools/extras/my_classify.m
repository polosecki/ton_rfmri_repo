function [err,fpos_err,fneg_err] = my_classify(cd)
%TODO:
% 1-Truncate K vector so that it doesn't exceed number of features.

% cd - classifier descriptor
runit=1;
alg = cd.alg; K=cd.top_K; alg = cd.alg; method = cd.method;
lambda1 = cd.lambda1; lambda2 = cd.lambda2;

if strcmp(alg,'weka')
    method = cd.weka_classifier;
end
 
sel_type = cd.sel_type; thresh = cd.thresh; 

infname = cd.fnm;    

 s_lambda1 = sprintf('%.2f_',lambda1); 
 s_lambda2= sprintf('%.2f_',lambda2);
% s_lambda2='';
%     
%     if lambda1
%         s_lambda1=sprintf('_lam1_%.3f',lambda1);
%     end
%     if lambda2
%         s_lambda2=sprintf('_lam2_%.3f_',lambda2);
%     end



for i = 1:size(infname,1)
    infname{i}
    load(infname{i});
    [n,m_max]=size(data);
    neg_samples = nnz(data(:,end) < 0); %Number of samples in class -1
    
    in_fnm = infname{i};
    in_fnm(strfind(in_fnm,'_')) = '-'; in_fnm(strfind(in_fnm,'/')) = '-';
    
    if isfield(cd,'weka_classifier')
        outfig_all = sprintf('Figures/%s_%s_%s%s%s_all.fig',in_fnm,cd.weka_classifier,method,s_lambda1,s_lambda2);
        outeps_all = sprintf('Figures/%s_%s_%s%s%s_all.eps',in_fnm,cd.weka_classifier,method,s_lambda1,s_lambda2);
    else
        outfig_all = sprintf('Figures/%s_%s%s%s_%s_all.fig',alg,method,s_lambda1,s_lambda2,in_fnm);
        outeps_all = sprintf('Figures/%s_%s%s%s_%s_all.eps',alg,method,s_lambda1,s_lambda2,in_fnm);
    end
    
    
    
    if nnz(isnan(data))
        display 'NaN in the data';
        exit;
    end

    majority_class_err = ones(1,length(K))*min(neg_samples,n-neg_samples)/n;
    if i==1;
        figure(1000); plot( K, majority_class_err, 'g--'); hold on;
    end
    
   

    if isfield(cd,'weka_classifier')
        [outfname,err]=sprintf('results/%s_%s%s%s%s',cd.weka_classifier,method,s_lambda1,s_lambda2,in_fnm);
    else
        [outfname,err]=sprintf('results/%s_%s%s%s%s',alg,method,s_lambda1,s_lambda2,in_fnm);
    end

 
    outfig0 = sprintf('Figures/map_%s%d_%s.fig',sel_type,K(length(K)),in_fnm);
    outeps0 = sprintf('Figures/map_%s%d_%s.eps',sel_type,K(length(K)),in_fnm);
    
    if isfield(cd,'weka_classifier')
            outfig1 = sprintf('Figures/%s_%s%s%s_%s%d_%s',cd.weka_classifier,method,s_lambda1,s_lambda2,sel_type,K(length(K)),in_fnm);
            outfig1(1,(end-2):end)='fig';
    else
    outfig1 = sprintf('Figures/%s_%s%s%s_%s%d_%s.fig',alg,method,s_lambda1,s_lambda2,sel_type,K(length(K)),in_fnm);
    %outfig1(1,(end-2):end)='fig';
    end
    
    outeps1 = outfig1; 
    outeps1(1,(end-2):end)='eps';
 
    
    lambdas = [lambda1 lambda2];
    %%%%%%%%%%%% run classifier
 
    if m_max < K(1)-1 % less features than we wanted to select
        K = [m_max-1];    %SHO 
    end
        
 
    if K==1 & strcmp(alg,'MRF')
            alg = 'GNB';
    end

    KK=length(K);
    avg_err = zeros(1,KK);
    fpos_err = zeros(1,KK);
    fneg_err = zeros(1,KK);
    
    try
        for tf = 1:KK
            top_features = K(tf);
            
            if thresh
                current_infname = sprintf('%s%.2f_top_%d_%s',sel_type,thresh,top_features,in_fnm);
            else
                current_infname = sprintf('%s_top_%d_%s',sel_type,top_features,in_fnm);
      
            end
            
            k = cd.folds;
            
            % result=evaluate_classifier(infname,outfname,k,m_max,classifier,method,lambdas,normalize, sel_type, top_features, thresh)
            if runit
                 
                result(tf)=evaluate_classifier(infname{i},outfname,k,m_max,alg,method,lambdas,0,sel_type,top_features,thresh,cd);
                save(outfname,'result');
            else
                load(outfname);
            end
            
            avg_err(tf) = result(tf).total_err;
            fpos_err(tf) = result(tf).total_fpos_err;
            fneg_err(tf) = result(tf).total_fneg_err;
            
        end
        
        figure(i);
        %semilogx(K, fpos_err, 'b*-', K, fneg_err, 'ro-', K, avg_err,  'ks-', K, majority_class_err, 'gv-');
        plot(K, fpos_err, 'b*-', K, fneg_err, 'ro-', K, avg_err,  'ks-', K, majority_class_err, 'gv-');
        
        
        [val,ind] = min(avg_err); 
        best_K = K(ind);
        
        if cd.start_ranking_from > 1 && ~strcmpi(alg,'en') && ~strcmpi(alg,'lasso')
            best_K = best_K + cd.start_ranking_from  - 1;
        end
        % save best result in the file
        if isfield(cd,'kyle_file')
            if cd.kyle_file == 1
                fid = fopen('combine_results.txt','a');
                fprintf(fid,'%s  %s(%s), K=%d: TP(sensitivity)=%.2f, TN(specificity)=%.2f, Acc=%.2f, Baseline Acc=%.3f \n',infname{i},alg,method,best_K,1-fneg_err(ind),1-fpos_err(ind),1-avg_err(ind),1-majority_class_err(ind));
                fclose(fid);
            end
            if cd.kyle_file == 2
                fid = fopen('en_test_results.txt','a');
                fprintf(fid,'%s  %s(%s), K=%d: TP(sensitivity)=%.2f, TN(specificity)=%.2f, Acc=%.2f, Baseline Acc=%.2f \n',infname{i},alg,method,best_K,1-fneg_err(ind),1-fpos_err(ind),1-avg_err(ind),1-majority_class_err(ind));
                fclose(fid);
            end
        elseif isfield(cd,'save_file')
            fid = fopen(cd.save_file,'a');
                fprintf(fid,'%s  %s(%s)K=%d: TP(sensitivity)=%.2f, TN(specificity)=%.2f, Acc=%.2f, Baseline Acc=%.3f \n',infname{i},alg,method,best_K,1-fneg_err(ind),1-fpos_err(ind),1-avg_err(ind),1-majority_class_err(ind));
                fclose(fid);
            
        else
            fid = fopen('current_results.txt','a');
            fprintf(fid,'%s  %s(%s), K=%d: TP(sensitivity)=%.2f, TN(specificity)=%.2f, Acc=%.2f, Baseline Acc=%.2f \n',infname{i},alg,method,best_K,1-fneg_err(ind),1-fpos_err(ind),1-avg_err(ind),1-majority_class_err(ind));
            fclose(fid);
        end
    catch ERROR
        disp(ERROR)
    
        if isfield(cd,'kyle_file')
            if cd.kyle_file == 1
                fid = fopen('combine_results.txt','a');
                fprintf(fid,'%s  %s(%s): Error \n',infname{i},alg,method);
                fclose(fid);
            end
            if cd.kyle_file == 2
                fid = fopen('en_test_results.txt','a');
                fprintf(fid,'%s  %s(%s): Error \n',infname{i},alg,method);
                fclose(fid);
            end
        elseif isfield(cd,'save_file')
            fid = fopen(cd.save_file,'a');
                fprintf(fid,'%s  %s(%s): Error \n',infname{i},alg,method);
                fclose(fid);
        else
            fid = fopen('current_results.txt','a');
            fprintf(fid,'%s  %s(%s): Error \n',infname{i},alg,method);
            fclose(fid);
        end
    end
    xlabel(sprintf('K top voxels (%s)',sel_type),'fontsize',16);
    ylim([0 1]);
    %legend('no varsel', 'varsel(tau/rho=100)');
    legend('false positive', 'false negative', 'total error','baseline');

    if isfield(cd,'weka_classifier')
      [s,e]=sprintf('%s on  %s',cd.weka_classifier,in_fnm);   
    else
    [s,e]=sprintf('%s on %s',alg,in_fnm);    
    end        
                
    title(s,'fontsize',12);  
    saveas(gcf,outfig1,'fig'); 
    saveas(gcf,outeps1,'psc2');
    
%     %%% all features
%      figure(1000);
%      plot(K, avg_err,  cd.plot(i)); hold on;
% %    
%      xlabel(sprintf('K top voxels (%s)',sel_type),'fontsize',16);
%      ylim([0 1]);
%      %legend('no varsel', 'varsel(tau/rho=100)');
%      legend(cd.legend);
%     
%     if isfield(cd,'weka_classifier')
%       [s,e]=sprintf('%s on  all features',cd.weka_classifier);   
%     else
%     [s,e]=sprintf('%s on  all features',alg);    
%     end
%                 
%     title(s,'fontsize',12);  
%     saveas(gcf,outfig_all,'fig'); 
%     saveas(gcf,outeps_all,'psc2');

end
 


 


%%%%%%%

 