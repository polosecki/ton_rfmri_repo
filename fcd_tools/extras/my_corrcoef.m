function [C] = my_corrcoef(data)

data = my_standardize(data);
N=size(data,1);
C =   data'*data/(N-1); % normalize by 1/(N-1)

end

