function X = my_standardize(X)
% STANDARDIZE  Standardize the observations of a data matrix.
%    X = STANDARDIZE(X) centers and scales the observations of a data
%    matrix such that each variable (column) has unit variance.
%
% Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk

[n p] = size(X);
X = center(X);
%X = X./(ones(n,1)*std(X,1));
zero_inds = find(max(abs(X))==0);
X = X./(ones(n,1)*std(X));
X(:,zero_inds)=0;

