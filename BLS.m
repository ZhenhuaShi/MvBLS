function [trainBCA,testBCA,Lambda2B]=BLS(trainX,trainY,testX,testY,Lambda2s)
if ~exist('trainX', 'var')
    clear;
    rng default
    [N,N2,D1,D2,D3,NC]=deal(178,100,7,6,5,3);
    trainX={rand(N,D1), rand(N,D2), rand(N,D3)};
    trainY=datasample(1:NC,N,'replace',true)';
    tuneX={rand(N2,D1), rand(N2,D2), rand(N2,D3)};
    tuneY=datasample(1:NC,N2,'replace',true)';
    testX={rand(N2,D1), rand(N2,D2), rand(N2,D3)};
    testY=datasample(1:NC,N2,'replace',true)';
    testX={tuneX,testX};
    testY={tuneY,testY};
%     % multi-view to single-view
%     trainX=cell2mat(trainX);
%     testX=cellfun(@(u)cell2mat(u),testX,'UniformOutput',false);
    Lambda2s=0.1;
end
nFNodes=20;
nFGroups=20;
nMVENodes=500;
Scale=0.8;
if ~iscell(trainX)
    trainX=cellfun(@(u)trainX,cell(1),'UniformOutput',false);
    testX=cellfun(@(u)cellfun(@(v)u,cell(1),'UniformOutput',false),testX,'UniformOutput',false);
end
uniqueY=unique(trainY);
trainOY=double(bsxfun(@eq, trainY(:), uniqueY'));
trainOY(trainOY==0)=-1;
nV = length(trainX);
% trainFeatures
trainXBias = cellfun(@(X)[zscore(X')' .1*ones(size(X,1),1)], trainX, 'UniformOutput',false);
Zr = cellfun(@(XB)cellfun(@(z)mapminmax(XB*(2*rand(size(XB,2),nFNodes)-1)),cell(1,nFGroups),'UniformOutput',false), trainXBias, 'UniformOutput',false);
We = cellfun(@(zr,XB)cellfun(@(z)LASSOADMM(z,XB,1e-3,50,1)',zr,'UniformOutput',false),Zr,trainXBias,'UniformOutput',false);
[Z, PSZ] = cellfun(@(we,XB)cellfun(@(w)mapminmax(w'*XB',0,1),we,'UniformOutput',false),We,trainXBias,'UniformOutput',false);
trainMVZ = cellfun(@(z)cat(1,z{:})',Z,'UniformOutput',false);
if nV*nFNodes * nFGroups >= nMVENodes
    MVWh = orth(2*rand(nV*nFNodes * nFGroups+1,nMVENodes)-1);
else
    MVWh = orth(2*rand(nV*nFNodes * nFGroups+1,nMVENodes)'-1)';
end
trainMVZW = [cat(2,trainMVZ{:}) .1 * ones(size(cat(2,trainMVZ{:}),1),1)] *MVWh;
tmpPSH = Scale/max(abs(trainMVZW(:)));
trainMVH{1} = tansig(trainMVZW* tmpPSH);
trainT=[cat(2,trainMVZ{:}) cat(2,trainMVH{:})];
% testFeatures
testXBias = cellfun(@(mvX)cellfun(@(X)[zscore(X')' .1*ones(size(X,1),1)],mvX,'UniformOutput',false), testX, 'UniformOutput',false);
Zr = cellfun(@(mvX)cellfun(@(XB,we)cellfun(@(w)XB*w,we,'UniformOutput',false), mvX,We, 'UniformOutput',false), testXBias, 'UniformOutput',false);
Z = cellfun(@(u)cellfun(@(zr,PS)cellfun(@(z,p)mapminmax('apply',z',p),zr,PS,'UniformOutput',false),u,PSZ,'UniformOutput',false), Zr,'UniformOutput',false);
testMVZ = cellfun(@(u)cellfun(@(z)cat(1,z{:})',u,'UniformOutput',false), Z,'UniformOutput',false);
tmptestMVH = cellfun(@(u)tansig([cat(2,u{:}) .1 * ones(size(cat(2,u{:}),1),1)] * MVWh * tmpPSH),testMVZ, 'UniformOutput',false);
testMVH=cellfun(@(u)cellfun(@(v)u,cell(1),'UniformOutput',false),tmptestMVH, 'UniformOutput',false);
testT=cellfun(@(u,v)[cat(2,u{:}) cat(2,v{:})],testMVZ,testMVH,'UniformOutput',false);
% best Lambda2
for Lambda2=Lambda2s
    Wo = (trainT'  *  trainT+eye(size(trainT',1)) * Lambda2)\(trainT'  *  trainOY);
    yTrainPC = trainT * Wo;
    [~,trainYPred]= max(yTrainPC,[],2);
    tmp=CacluateBCA(trainY,uniqueY(trainYPred));
    yTestPC=cellfun(@(u)u * Wo,testT,'UniformOutput',false);
    [~,testYPred]= cellfun(@(u)max(u,[],2),yTestPC,'UniformOutput',false);
    tmpt=cellfun(@(u,v)CacluateBCA(v,uniqueY(u)),testYPred,testY,'UniformOutput',false);
    if ~exist('thre','var')||tmpt{1}>thre
        thre=tmpt{1};
        [trainBCA,testBCA]=deal(tmp,tmpt);
        Lambda2B=Lambda2;
    end
end
if length(testX)==1
    testBCA=testBCA{1};
end