function [trainAccs,testAccs,yTrainPC,yTestPC,Best]=MVBLS(trainMVX,trainY,testMVX,testY,params)
if ~exist('trainMVX', 'var')
    clear;
    rng default
    addpath(genpath('../../library'))
    [N,N2,D1,D2,D3,NC]=deal(178,100,7,6,5,3);
    trainMVX={rand(N,D1), rand(N,D2), rand(N,D3)};
    trainY=datasample(1:NC,N,'replace',true)';
    testMVX={rand(N2,D1), rand(N2,D2), rand(N2,D3)};
    testY=datasample(1:NC,N2,'replace',true)';
    params=[];
    params.nFNodess=[10,30]; % feature nodes  per window
    params.nFGroupss=[10,30]; % number of windows of feature nodes
    params.nENodess=[100,900]; % number of enhancement nodes
    params.nMVENodess=[100,900];
    [params.Algorithm,params.isADR]=deal('MVBLS',0);%deal('MVBLS3',2);
end
if ~iscell(trainMVX)
    trainMVX={trainMVX};
    testMVX={testMVX};
end
if ~exist('params','var')||~isfield(params,'nFNodess')
    params.nFNodess=30;%20;
end
if ~isfield(params,'nFGroupss')
    params.nFGroupss=30;%20;
end
if ~isfield(params,'Lambdas')
    params.Lambdas=1e-3;
end
if ~isfield(params,'Alphas')
    params.Alphas=1;
end
if ~isfield(params,'Lambda2s')% the l2 regularization parameter
    params.Lambda2s=0.1;
end
if ~isfield(params,'MaxIters')
    params.MaxIters=50;
end
if ~isfield(params,'Rhos')
    params.Rhos=1;
end
if ~isfield(params,'Scales')% the shrinkage scale of the enhancement nodes
    params.Scales=0.8;
end
if ~isfield(params,'Algorithm')
    params.Algorithm='MVBLS';%MVBLS%MVBLS2%MVBLS3
end
if ~isfield(params,'isADR')
    params.isADR=false;
end
if ~isfield(params,'nMVENodess')||ismember(params.Algorithm,{'MVBLS2'})
    params.nMVENodess=900;%500;
end
if ~isfield(params,'nENodess')||ismember(params.Algorithm,{'BLS','MVBLS'})
    params.nENodess=900;%500;
end

Alpha=params.Alphas(1);
Rho=params.Rhos(1);

uniqueY=unique(trainY);
trainOY=double(bsxfun(@eq, trainY(:), uniqueY'));
trainOY(trainOY==0)=-1;
nV = length(trainMVX);
Best=[];
yTrainPC=[]; yTestPC=[];
for nFNodes=params.nFNodess
    for nFGroups=params.nFGroupss
        for nENodes=params.nENodess
            for nMVENodes=params.nMVENodess
                for Lambda2=params.Lambda2s
                    for Lambda=params.Lambdas
                        for MaxIter=params.MaxIters
                            for Scale=params.Scales
                                if ~isfield(params,'optmodel')
                                    trainMVXBias = cellfun(@(X)[zscore(X')' .1*ones(size(X,1),1)], trainMVX, 'UniformOutput',false);
                                    Zr = cellfun(@(XB)cellfun(@(z)mapminmax(XB*(2*rand(size(XB,2),nFNodes)-1)),cell(1,nFGroups),'UniformOutput',false), trainMVXBias, 'UniformOutput',false);
                                    We = cellfun(@(zr,XB)cellfun(@(z)LASSOADMM(z,XB,Lambda,MaxIter,Rho,Alpha)',zr,'UniformOutput',false),Zr,trainMVXBias,'UniformOutput',false);
                                    [Z, PSZ] = cellfun(@(we,XB)cellfun(@(w)mapminmax(w'*XB',0,1),we,'UniformOutput',false),We,trainMVXBias,'UniformOutput',false);
                                    trainMVZ = cellfun(@(z)cat(1,z{:})',Z,'UniformOutput',false);
                                    if ismember(params.Algorithm,{'MVBLS2','MVBLS3'})
                                        if nFGroups*nFNodes >= nENodes
                                            Wh = cellfun(@(w)orth(2*rand(nFGroups*nFNodes+1,nENodes)-1),cell(1,nV),'UniformOutput',false);
                                        else
                                            Wh = cellfun(@(w)orth(2*rand(nFGroups*nFNodes+1,nENodes)'-1)',cell(1,nV),'UniformOutput',false);
                                        end
                                        trainMVZW = cellfun(@(z,w)[z .1 * ones(size(z,1),1)]*w,trainMVZ,Wh,'UniformOutput',false);
                                        PSH = cellfun(@(H)Scale/max(abs(H(:))),trainMVZW,'UniformOutput',false);
                                        trainMVH = cellfun(@(H,PS)tansig(H*PS),trainMVZW,PSH,'UniformOutput',false);
                                    end
                                    if ismember(params.Algorithm,{'BLS','MVBLS','MVBLS3'})
                                        if nV*nFNodes * nFGroups >= nMVENodes
                                            MVWh = orth(2*rand(nV*nFNodes * nFGroups+1,nMVENodes)-1);
                                        else
                                            MVWh = orth(2*rand(nV*nFNodes * nFGroups+1,nMVENodes)'-1)';
                                        end
                                        trainMVZW = [cat(2,trainMVZ{:}) .1 * ones(size(cat(2,trainMVZ{:}),1),1)] *MVWh;
                                        tmpPSH = Scale/max(abs(trainMVZW(:)));
                                        tmptrainMVH = tansig(trainMVZW* tmpPSH);
                                        if ismember(params.Algorithm,{'BLS','MVBLS'})
                                            trainMVH{1}=tmptrainMVH;
                                        else
                                            trainMVH{nV+1}=tmptrainMVH;
                                        end
                                    end
                                    if ~params.isADR
                                        trainT=[cat(2,trainMVZ{:}) cat(2,trainMVH{:})];
                                        Wo = (trainT'  *  trainT+eye(size(trainT',1)) * Lambda2)\(trainT'  *  trainOY);
                                        yTrainPC = trainT * Wo;
                                        [~,trainYPred]= max(yTrainPC,[],2);
                                        tmptrainAccs=mean(uniqueY(trainYPred)==trainY);
                                    end
                                else
                                    [tmptrainAccs]=deal(nan);
                                    PSZ=params.optmodel.PSZ;
                                    if ~params.isADR
                                        Wo=params.optmodel.Wo;
                                    else
                                        W=params.optmodel.W;
                                        b=params.optmodel.b;
                                    end
                                    We=params.optmodel.We;
                                    if ismember(params.Algorithm,{'MVBLS2','MVBLS3'})
                                        Wh=params.optmodel.Wh;
                                        PSH=params.optmodel.PSH;
                                    end
                                    if ismember(params.Algorithm,{'BLS','MVBLS','MVBLS3'})
                                        MVWh=params.optmodel.MVWh;
                                        tmpPSH=params.optmodel.tmpPSH;
                                    end
                                end
                                testMVXBias = cellfun(@(X)[zscore(X')' .1*ones(size(X,1),1)], testMVX, 'UniformOutput',false);
                                Zr = cellfun(@(XB,we)cellfun(@(w)XB*w,we,'UniformOutput',false), testMVXBias,We, 'UniformOutput',false);
                                Z = cellfun(@(zr,PS)cellfun(@(z,p)mapminmax('apply',z',p),zr,PS,'UniformOutput',false),Zr,PSZ,'UniformOutput',false);
                                testMVZ = cellfun(@(z)cat(1,z{:})',Z,'UniformOutput',false);
                                if ismember(params.Algorithm,{'MVBLS2','MVBLS3'})
                                    testMVH = cellfun(@(z,w,p)tansig([z .1*ones(size(z,1),1)]*w*p),testMVZ,Wh,PSH,'UniformOutput',false);
                                end
                                if ismember(params.Algorithm,{'BLS','MVBLS','MVBLS3'})
                                    tmptestMVH = tansig([cat(2,testMVZ{:}) .1 * ones(size(cat(2,testMVZ{:}),1),1)] * MVWh * tmpPSH);
                                    if ismember(params.Algorithm,{'BLS','MVBLS'})
                                        testMVH{1}=tmptestMVH;
                                    else
                                        testMVH{nV+1}=tmptestMVH;
                                    end
                                end
                                if ~params.isADR
                                    testT=[cat(2,testMVZ{:}) cat(2,testMVH{:})];
                                    yTestPC=testT * Wo;
                                    [~,testYPred]= max(yTestPC,[],2);
                                    tmptestAccs=mean(uniqueY(testYPred)==testY);
                                end
                                if ~exist('testAccs','var')||tmptestAccs>testAccs
                                    [trainAccs,testAccs]=deal(tmptrainAccs,tmptestAccs);
                                    if nargout >4
                                        Best.Algorithm=params.Algorithm;
                                        Best.isADR=params.isADR;
                                        Best.nFNodess=nFNodes;
                                        Best.nFGroupss=nFGroups;
                                        Best.nENodess=nENodes;
                                        Best.nMVENodess=nMVENodes;
                                        Best.Lambda2s=Lambda2;
                                        Best.optmodel.PSZ=PSZ;
                                        if ~params.isADR
                                            Best.optmodel.Wo=Wo;
                                        elseif ~isfield(params,'optmodel')
                                            Best.optmodel.W=tmpmodel.W;
                                            Best.optmodel.b=tmpmodel.b;
                                            Best.alpha=tmpmodel.alpha;
                                            Best.MaxIter=tmpmodel.MaxIter;
                                        end
                                        Best.optmodel.We=We;
                                        if ismember(params.Algorithm,{'MVBLS2','MVBLS3'})
                                            Best.optmodel.Wh=Wh;
                                            Best.optmodel.PSH=PSH;
                                        end
                                        if ismember(params.Algorithm,{'BLS','MVBLS','MVBLS3'})
                                            Best.optmodel.MVWh=MVWh;
                                            Best.optmodel.tmpPSH=tmpPSH;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end