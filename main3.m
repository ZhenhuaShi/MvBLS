clc; clear; close all; rng('default')
addpath(genpath('./DCCAE'))
diary ./DiaryFile
diary on;
dataFolder='./JHUChoiceZscoreF/';
dirs=dir([dataFolder '*.mat']);
dircell=struct2cell(dirs)' ;
DatasetNameList=dircell(:,1);
datasets = cellfun(@(u)strrep(u,'.mat',''),DatasetNameList,'UniformOutput',false);
S = length(datasets);
C = 4;
nRepeat=8;%3
useGPU=false;
methods = {'SVM','MVBLS','DCCAE'}; %{'SVM','Ridge','BLS','MvDA','DCCAE','MVBLS','MVBLS2','MVBLS3'};
nAlgs=length(methods);
params.nFNodess=15;%[10, 20] feature nodes  per window
params.nFGroupss=15; %[10, 20] number of windows of feature nodes
params.nENodess=300; %[100, 500] number of enhancement nodes
params.nMVENodess=300;%[100, 500]
params.Lambda2s=1;%10.^(-6:2:6)
params.Algorithm={'MVBLS'};%{'MVBLS2'};%{'MVBLS3'};
c=1e-4;
bs=16;
c2=1;
nEpoch=9;
[ACCTrain,ACCTest,ACCTrain2,ACCTest2,times]=deal(cellfun(@(t)nan(nAlgs,1),cell(S,nRepeat),'UniformOutput',false));
delete(gcp('nocreate'))
parpool(4);
% Display results in parallel computing
dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(data) fprintf('%d-%d ', data{1},data{2})); % print progress of parfor
%for s=2%41%1:S
parfor r=1:nRepeat
    dataDisp=cell(1,2);    dataDisp{1}=r; %warning off all;
    for s=1:S        
        dataDisp{2} = s;   dqWorker.send(dataDisp); % Display progress in parfor
        tmp=load([dataFolder datasets{s} '.mat']);
        X=tmp.X; Y=tmp.Y; Y2=tmp.Y2;
        N0=size(X{1},1);    N=round(N0*.8);
        X=cellfun(@(x)double(x(:,:)),X,'UniformOutput',false);
        MVX=X;    X=cat(2,X{:});   
        
        ids=randperm(N0);
        idsTrain=ids(1:N); XTrain=X(idsTrain,:); yTrain=Y(idsTrain); y2Train=Y2(idsTrain,:);
        idsTest=ids(N+1:end); XTest=X(idsTest,:); yTest=Y(idsTest); y2Test=Y2(idsTest,:);
        indTrain1 = sub2ind([N,C],(1:N)',y2Train(:,1));
        indTrain2 = sub2ind([N,C],(1:N)',y2Train(:,2));
        indTest1 = sub2ind([N0-N,C],(1:(N0-N))',y2Test(:,1));
        indTest2 = sub2ind([N0-N,C],(1:(N0-N))',y2Test(:,2));
        MVXTrain = cellfun(@(d)d(idsTrain,:), MVX, 'UniformOutput',false);
        MVXTest = cellfun(@(d)d(idsTest,:), MVX, 'UniformOutput',false);
        % SVM
        tic
        SVM = templateSVM('BoxConstraint',c,'KernelFunction','linear','Standardize',1);
        model = fitcecoc(XTrain,yTrain,'Learners',SVM);% error-correcting output codes (ECOC)
        [yTrainP, yTrainPC]= predict(model,XTrain);
        ACCTrain{s,r}(1) =  mean(yTrain==yTrainP);
        yTrainP = y2Train(:,1);
        id = yTrainPC(indTrain1)<yTrainPC(indTrain2);
        yTrainP(id)=y2Train(id,2);
        ACCTrain2{s,r}(1) =  mean(yTrain==yTrainP);
        [yTestP, yTestPC]= predict(model,XTest);
        ACCTest{s,r}(1) = mean(yTest==yTestP);
        yTestP = y2Test(:,1);
        id = yTestPC(indTest1)<yTestPC(indTest2);
        yTestP(id)=y2Test(id,2);
        ACCTest2{s,r}(1) =  mean(yTest==yTestP);
        times{s,r}(1)=toc;
        % MVBLS
        tic
        [ACCTrain{s,r}(2),ACCTest{s,r}(2),yTrainPC,yTestPC]=MVBLS(MVXTrain,yTrain,MVXTest,yTest,params);
        yTrainP = y2Train(:,1);
        id = yTrainPC(indTrain1)<yTrainPC(indTrain2);
        yTrainP(id)=y2Train(id,2);
        ACCTrain2{s,r}(2) =  mean(yTrain==yTrainP);
        yTestP = y2Test(:,1);
        id = yTestPC(indTest1)<yTestPC(indTest2);
        yTestP(id)=y2Test(id,2);
        ACCTest2{s,r}(2) =  mean(yTest==yTestP);
        times{s,r}(2)=toc;
        %% DCCAE
        tic
        K=10;
        % Regularizations for each view.
        rcov1=1e-4; rcov2=1e-4;
        % Hidden activation type.
        hiddentype='sigmoid';
        outputtype='sigmoid';
        dim1 = size(MVXTrain{1},2);
        dim2 = size(MVXTrain{2},2);
        % Architecture for view 1 feature extraction network.
        NN1=[1024 1024 1024 K];
        % Architecture for view 2 feature extraction network.
        NN2=[1024 1024 1024 K];
        % Architecture for view 1 reconstruction network.
        NN3=[1024 1024 1024 dim1];
        % Architecture for view 2 reconstruction network.
        NN4=[1024 1024 1024 dim2];
        % Weight decay parameter.
        l2penalty=1e-4;
        % Reconstruction error term weight.
        lambda=0.001;
        % Minibatchsize for the correlation term.
        cca_batchsize=bs;
        % Minibatchsize for reconstruction error term.
        rec_batchsize=bs;
        % Learning rate.
        eta0=0.01;
        % Rate in which learning rate decays over iterations.
        % 1 means constant learning rate.
        decay=1;
        % Momentum.
        momentum=0.99;
        % How many passes of the data you run SGD with.
        maxepoch=nEpoch;
        %                         % Pretraining is used to speedup autoencoder training.
        %                         pretrainnet='RBMPRETRAIN_K=10.mat';
        [F1opt,F2opt]=DCCAE_train( ...
            MVXTrain{1},MVXTrain{2},MVXTrain{1},MVXTrain{2},[],[],K,lambda,hiddentype,outputtype,...
            NN1,NN2,NN3,NN4,rcov1,rcov2,l2penalty, cca_batchsize,rec_batchsize,...
            eta0,decay,momentum,maxepoch,[],[],useGPU);%,[],pretrainnet);
        trainXproj=[gather(deepnetfwd(MVXTrain{1},F1opt)), gather(deepnetfwd(MVXTrain{2},F2opt))];
        testXproj=[gather(deepnetfwd(MVXTest{1},F1opt)), gather(deepnetfwd(MVXTest{2},F2opt))];
        SVM = templateSVM('BoxConstraint',c2,'KernelFunction','linear','Standardize',1);
        model = fitcecoc(trainXproj,yTrain,'Learners',SVM);% error-correcting output codes (ECOC)
        [yTrainP, yTrainPC]= predict(model,trainXproj);
        ACCTrain{s,r}(3) =  mean(yTrain==yTrainP);
        yTrainP = y2Train(:,1);
        id = yTrainPC(indTrain1)<yTrainPC(indTrain2);
        yTrainP(id)=y2Train(id,2);
        ACCTrain2{s,r}(3) =  mean(yTrain==yTrainP);
        [yTestP, yTestPC]= predict(model,testXproj);
        ACCTest{s,r}(3) = mean(yTest==yTestP);
        yTestP = y2Test(:,1);
        id = yTestPC(indTest1)<yTestPC(indTest2);
        yTestP(id)=y2Test(id,2);
        ACCTest2{s,r}(3) =  mean(yTest==yTestP);
        times{s,r}(3)=toc;
    end
end
save('RunTime_s45_r8.mat','methods','datasets','ACCTrain','ACCTest','ACCTrain2','ACCTest2','times')
diary off;