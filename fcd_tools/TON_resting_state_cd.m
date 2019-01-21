%This script  creates a classifier descriptor


cd = struct('alg','','method','','lambda1',0,'lambda2',0,'folds',0,'top_K',0);

%cd. alg = 'GNB'; % 'GNB' or 'MRF'
%cd.method = 'no_varsel_MRF'; % for MRF, need to select specific method for learning inverse covariance matrix 
% other options: 
%method = 'our_covsel';  % does not work with new matlab
%method='ALM';
%method = 'sinco';
%method = 'projected_gradient'; % need to set prec parameter
%method = 'varsel_MRF';

%method = 'both_mix_vox';

%lambdas = [0.0001 0.001 0.01  0.1 1 ];

cd.alg = 'GNB';
% cd.lambda1=0.01; % parameter of MRF or Lasso
% cd.lambda2=0.01; % parameter of EN
% cd.method = 'no_varsel_MRF';

cd.sel_type = 'mi';%'ttest';  % type of ranking for feature selection
cd.thresh = 0;  

% cd.CV_type = '10-out';%'LOO' % leave one out; '10-fold' 
%       %k = n/2;  % when there are 2 runs per subject, MUST do at least
%       %leave TWO out CV (combine both runs)
%       %k = n; % when there is 1 run per subject, can  do leave-one-out CV (if the number of folds is n
      
cd.top_K = [ 3 5 7 8 9 10 12 15 20 30 40 50 70 100 200 300 500 1000 5000 10000 20000 30000];

cd.fnm ={'Neurospin_bct_strength_filter_0.mat'
    };
cd.legend ={'corr_subset_0'};
cd.plot = ['bx-'];
cd.preproc = 0;
cd.start_ranking_from = 1; % weird stuff for me, PP

