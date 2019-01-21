function [data] = make_happy_sad_data_and_regressors(data,dd,reg,regressor_name)

hd_runs=zeros(size(dd.labels,1)*2,1);
hd_subjs=(dd.labels==1);
hd_runs(1:2:end-1)=hd_subjs; hd_runs(2:2:end)=hd_subjs;
hd_runs = logical(hd_runs);


ctrl_reg = dd.ctrl_regressors;
temp_x = [ones(length(dd.labels),1)  ctrl_reg.age(:,1) ctrl_reg.sex(:,1) ctrl_reg.CAP(:,1) ctrl_reg.CAG(:,1)];
X=nan(size(temp_x,1)*2, size(temp_x,2));
X(1:2:end-1,:)=temp_x;
X(2:2:end,:)=temp_x;
% Notice that fit is made with healthy subjects only:
b=X(~hd_runs,1:3)\data(~hd_runs,:);

%Notice that correction is applied to HD subjects:
data(hd_runs,:) = data(hd_runs,:)-X(hd_runs,1:3)*b;

%Correct for effect of CAP/CAG in HD runs 
b=X(hd_runs,[1 4 5])\data(hd_runs,:);
data(hd_runs,:) = data(hd_runs,:)-X(hd_runs,[1 4 5])*b;

%Filter out the healthy patients from the data:
data(~hd_runs,:)=[];


%The following labeling of happy sad runs expects that non HD
%subjects have been removed from data
used_regressor = reg.(regressor_name);
sad_runs = -ones(size(data,1),1); % irrelevant starting value
sad_subjs= [used_regressor(dd.labels==1,1) < -.5];
sad_runs(1:2:end-1)=sad_subjs; sad_runs(2:2:end)=sad_subjs;
if any(sad_runs==-1)
    error('Some labels have been misproduced')
end
sad_runs(sad_runs==0)=1;

data(:,end) = sad_runs;
