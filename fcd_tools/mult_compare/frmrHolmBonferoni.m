function adjustedPVals = frmrHolmBonferoni(pVals)

%See http://en.wikipedia.org/wiki/Holm-Bonferroni_method
%Holm, S. (1979). 
%"A simple sequentially rejective multiple test procedure". 
%Scandinavian Journal of Statistics 6 (2): 65-70. JSTOR 4615733

adjustedPVals = nan(size(pVals));

[pVals, sortIndex] = sort(pVals(:));
pVals( pVals == 0 ) = eps(0);
weights = fliplr(1:length(pVals))';
weightedPVals = weights.*pVals;

for i = 1:length(weightedPVals)
    adjustedPVals(sortIndex(i)) = min( max(weightedPVals(1:i)), 1);
end