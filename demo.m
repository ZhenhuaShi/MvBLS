clc; clearvars; close all; rng(0);
nRepeats=8;
nn=1;
Lambdas=10.^(-6:2:6);
nVs=1:3;
LN0={'kNN','SVM','BLS','MvBLS'};
LN=cell(1,length(LN0)*length(nVs));
for i=1:length(nVs)
    LN(1+(i-1)*length(LN0):i*length(LN0))=strcat(LN0, ['-nV' num2str(nVs(i))]);
end
nAlgs=length(LN);

datasets={'goI08272012-01'}%{'Igo08282012-01'}

% Display results in parallel computing
dqWorker = parallel.pool.DataQueue; afterEach(dqWorker, @(data) fprintf('%d-%d ', data{1},data{2})); % print progress of parfor

[BCAtrain,BCAtune,BCAtest]=deal(cellfun(@(u)nan(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false));
[times,BestLambda]=deal(cellfun(@(u)nan(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false));
BestmIter=cellfun(@(u)ones(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
thres=cellfun(@(u)-inf(length(datasets),nAlgs),cell(nRepeats,1),'UniformOutput',false);
delete(gcp('nocreate'))
parpool(nRepeats);
parfor r=1:nRepeats
    dataDisp=cell(1,2);    dataDisp{1}=r;
    for s=1:length(datasets)
        dataDisp{2} = s;   send(dqWorker,dataDisp); % Display progress in parfor
        
        temp=load(['./' datasets{s} '.mat']);
        tX=temp.X; Y=temp.Y;
        tX=cellfun(@(x)double(x(:,:)),tX,'UniformOutput',false);
        N0=size(tX{1},1);
        N=round(N0*.8);
        idsTrain=datasample(1:N0,N,'replace',false);
        N1=round(N0*.1);
        idsTune=datasample(1:(N0-N),N1,'replace',false);
        id=0;
        for v=nVs
            if v>length(tX)
                X=cell2mat(tX);
            else
                X=tX{v};
            end
            X = zscore(X);
            XTrain=X(idsTrain,:); yTrain=Y(idsTrain);
            XTest=X; XTest(idsTrain,:)=[];
            yTest=Y; yTest(idsTrain)=[];
            XTune=XTest(idsTune,:); yTune=yTest(idsTune);
            XTest(idsTune,:)=[]; yTest(idsTune)=[];
            trainInd=idsTrain;
            testInd=1:N0;testInd(idsTrain)=[];
            tuneInd=testInd(idsTune);
            testInd(idsTune)=[];
            MXTrain=mean(XTrain);
            XTrain=XTrain-MXTrain; XTune=XTune-MXTrain; XTest=XTest-MXTrain;
            MvXTrain=cellfun(@(x)double(x(trainInd,:)),tX,'UniformOutput',false);
            MvXTune=cellfun(@(x)double(x(tuneInd,:)),tX,'UniformOutput',false);
            MvXTest=cellfun(@(x)double(x(testInd,:)),tX,'UniformOutput',false);
            MXTrain=cellfun(@(x)mean(x),MvXTrain,'UniformOutput',false);
            MvXTrain=cellfun(@(x,mx)x-mx,MvXTrain,MXTrain,'UniformOutput',false);
            MvXTune=cellfun(@(x,mx)x-mx,MvXTune,MXTrain,'UniformOutput',false);
            MvXTest=cellfun(@(x,mx)x-mx,MvXTest,MXTrain,'UniformOutput',false);
            
            %% kNN
            tic
            id=id+1;
            model = fitcknn(XTrain,yTrain,'NumNeighbors',nn,'Distance','euclidean','Standardize',1);
            BCAtrain{r}(s,id) = CacluateBCA(yTrain,predict(model,XTrain));
            BCAtune{r}(s,id) = CacluateBCA(yTune,predict(model,XTune));
            BCAtest{r}(s,id) = CacluateBCA(yTest,predict(model,XTest));
            times{r}(s,id)=toc;
            
            
            %% SVM
            tic
            id=id+1;
            for c=Lambdas
                SVM = templateSVM('BoxConstraint',c,'KernelFunction','linear','Standardize',1);
                model = fitcecoc(XTrain,yTrain,'Learners',SVM);% error-correcting output codes (ECOC)
                tmp=CacluateBCA(yTune,predict(model,XTune));
                if tmp>BCAtune{r}(s,id)||isnan(BCAtune{r}(s,id))
                    BCAtune{r}(s,id)=tmp;
                    BestLambda{r}(s,id)=c;
                    BCAtrain{r}(s,id) = CacluateBCA(yTrain,predict(model,XTrain));
                    BCAtest{r}(s,id) = CacluateBCA(yTest,predict(model,XTest));
                end
            end
            times{r}(s,id)=toc;
            
            
            %% BLS
            tic
            id=id+1;
            [BCAtrain{r}(s,id),testBCA,BestLambda{r}(s,id)]=BLS(XTrain,yTrain,{XTune,XTest},{yTune,yTest},Lambdas);
            BCAtune{r}(s,id)=testBCA{1};
            BCAtest{r}(s,id)=testBCA{2};
            times{r}(s,id)=toc;
            
            
            %% MvBLS
            id=id+1;
            if v>length(tX)
                tic
                [BCAtrain{r}(s,id),testBCA,BestLambda{r}(s,id)]=BLS(MvXTrain,yTrain,{MvXTune,MvXTest},{yTune,yTest},Lambdas);
                BCAtune{r}(s,id)=testBCA{1};
                BCAtest{r}(s,id)=testBCA{2};
                times{r}(s,id)=toc;
            else
                [BCAtrain{r}(s,id),BCAtune{r}(s,id),BCAtest{r}(s,id),BestLambda{r}(s,id)]=deal(BCAtrain{r}(s,id-1),BCAtune{r}(s,id-1),BCAtest{r}(s,id-1),BestLambda{r}(s,id-1));
            end
            
            
        end
    end
end
save('demo.mat','BCAtrain','BCAtune','BCAtest','times','BestLambda','datasets','nAlgs','LN','LN0','Lambdas','nRepeats','nVs');


%% Plot results
clear
load demo
totalHours=nansum(reshape(cat(1,times{:}),1,[]))/3600/8
close all;
lineStyles={'k--','k-','g--','g-','b--','b-','r--','r-','m--','m-','c--','c-'};
ids=1:length(LN0);
figure;
set(gcf,'DefaulttextFontName','times new roman','DefaultaxesFontName','times new roman','defaultaxesfontsize',12);
hold on;
for s=1:length(datasets)
    tmpt=cellfun(@(u)squeeze(u(s,:)),BCAtest,'UniformOutput',false);
    tmpt=nanmean(cat(3,tmpt{:}),3);
    tmpt=reshape(tmpt,length(LN0),length(nVs));
    for i=ids
        plot(tmpt(i,:),lineStyles{i},'linewidth',2);
    end
    set(gca,'XTick',nVs);
    set(gca,'yscale','log');
    xlabel('View'); ylabel('BCA'); box on; axis tight;
    title(datasets{s});
end
legend(LN0,'FontSize',12,'NumColumns',2);
ids=1:length(LN);
[tmp,ttmp]=deal(nan(length(datasets),length(LN),nRepeats));
for s=1:length(datasets)
    ttmp0=cellfun(@(u)squeeze(u(s,ids)),times,'UniformOutput',false);
    ttmp(s,ids,:)=cat(1,ttmp0{:})';
    for id=1:length(LN)
        try
            tmp(s,id,:)=cell2mat(cellfun(@(u,m)squeeze(u(s,id,find(m(s,id,:)==max(m(s,id,:)),1))),BCAtest,BCAtune,'UniformOutput',false));
        catch
        end
    end
end
A=[nanmean(nanmean(tmp(:,ids,:),1),3);
    nanstd(nanmean(tmp(:,ids,:),1),[],3);
    nanmean(nanmean(ttmp(:,ids,:),1),3);
    nanstd(nanmean(ttmp(:,ids,:),1),[],3)];
a=squeeze(nanmean(tmp(:,ids,:),3));
a=[a;nanmean(a,1)]; sa=sort(a,2);
b=a==sa(:,1);c=a==sa(:,2);
at=squeeze(nanmean(ttmp(:,ids,:),3));
al=nanmean(cat(3,BestLambda{:}),3); al=[al;nanmean(al,1)];