cd = struct('alg','','method','lambda','lambda2');

cd. alg = 'SVM'; % 'GNB' or 'MRF'
cd.method = 'no_varsel_MRF'; % for MRF, need to select specific method for learning inverse covariance matrix 
% other options: 
%method = 'our_covsel';  % does not work with new matlab
%method='ALM';
%method = 'sinco';
%method = 'projected_gradient'; % need to set prec parameter
%method = 'varsel_MRF';
%method = 'no_varsel_MRF';
%method = 'both_mix_vox';

%lambdas = [0.0001 0.001 0.01  0.1 1 ];

cd.lambda1=0.01; % parameter of MRF or Lasso
cd.lambda2=0.01; % parameter of EN

cd.sel_type = 'ttest';  % type of ranking for feature selection
cd.thresh = 0;