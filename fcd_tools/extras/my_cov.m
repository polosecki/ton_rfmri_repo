function [C] = my_cov(data)

data = center(data);
N=size(data,1);
C = data'*data/(N-1);

end

