function BCA=CacluateBCA(Y,YPred,isBCA)
if ~exist('Y','var')
    Y=[1 1 1 1 2 2];
    YPred=[ 1 1 1 1 1 1];
end
if ~exist('isBCA','var')
    isBCA=1;
end
if isBCA
    CM = confusionmat(Y,YPred);
    Sensitivity = diag(CM)./sum(CM,2);
    BCA = nanmean(Sensitivity);
else
    BCA = mean(Y==YPred); % Accuracy
end
end