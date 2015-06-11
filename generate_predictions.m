% script function function uses transfer learning to predict EEG seizures

function [where,where2,where3] = generate_predictions(matFile)

% epoch length (in # of samples?)
epochLen = 50;
% Overlap between epochs (percentage)
overlap = 20;
% matFile = 'EEG_Mat/AH04_1.mat';

load(matFile, 'combFeat', 'recordingStart', 'recordingEnd', ...
    'seizureStart', 'seizureEnd', 'samplingRate');
recordingEnd = double(recordingEnd);
recordingStart = double(recordingStart);
seizureStart = double(seizureStart);
seizureEnd = double(seizureEnd);

% Total number of epochs in this recording
totalEpochs = ceil((etime(recordingEnd,recordingStart)/ (epochLen/10) - 1)* (1 / (1 - ( overlap / 100)))+1);
numSeizures = size(seizureStart,1);
S_start = zeros(numSeizures, 1);
S_end = zeros(numSeizures, 1);

% For each seizure
for ids = 1:numSeizures    
    % Starting epoch of the seizure.
    S_start(ids) = (etime(seizureStart(ids,:),recordingStart) / (epochLen/10)-1)* ...
        (1 / (1 - ( overlap / 100))) +1;
    % Ending epoch of the seizure.
    S_end(ids) = (etime(seizureEnd(ids,:),recordingStart) / (epochLen/10)-1)* ...
        (1 / (1 - ( overlap / 100))) +1;
end

totalEpochs = ceil(totalEpochs);
S_start = floor(S_start);
S_end = floor(S_end);

% read coreset and split into training and testing sets
coreSet = combFeat;
topSet = [1,11,13,14,15,18,19,20,21,26,27,33,35];

% define training set size
startWinSize = 40; %round(size(coreSet,1)/2);
S_start = S_start - startWinSize; S_end = S_end - startWinSize;

% put together transfer sets
[path, name, ~] = fileparts(matFile);
coreName = name;
matDir = dir([path '/*.mat']);
transSet = cell(size(matDir,1)-1,1); t = 1;

for idm = 1: size(matDir,1)
%     skip the current matFile, add all the others
    if strcmp([path '/' matDir(idm).name], matFile)
        continue
    else
        transSet{t} = [path '/' matDir(idm).name];
        t = t + 1;
    end
end

train = 1:startWinSize; tester = startWinSize+1 : size(coreSet,1);

% calculate parameters for training prime set (within training set)
split = 0.5;
idx = train(1:round(size(train,2)*split));

%% Version 1 : vanilla AR(1)
% create matrix of form b0 + b1*x1 + b2*x2 + ...
newfeats = coreSet(train,:);
X = newfeats(1:end-1,:); Xfull = [X(:,topSet), ones(size(X,1),1)];
Y = newfeats(2:end,:);

% find coefficients of regression
pX = pinv(Xfull)*Y; bestEst = pX'; % AR parameters
estfeats = zeros(size(tester,2),size(coreSet,2));
newfeats = coreSet;
for j = tester
    X = newfeats(j-1,:); Xfull = [X(:,topSet), ones(size(X,1),1)];
    y_nxt = Xfull*bestEst'; % full estimate
    estfeats(j-tester(1)+1,:) = y_nxt;    
end

%% Version 2: AR(1) + transfer learning
tPrime = coreSet(idx,:); valSet = coreSet(idx(end)+1:train(end),:); 
X = tPrime(1:end-1,:); Xfull = [X(:,topSet), ones(size(X,1),1)];
Y = tPrime(2:end,:);
% find coefficients of regression
pX = pinv(Xfull)*Y; wolearn = pX'; % AR parameters for training prime set
bestEst2 = tikhFunc2(wolearn,transSet,tPrime,valSet, topSet);

% predict next feature values and check the errors
estfeats2 = zeros(size(tester,2),size(coreSet,2));
newfeats = coreSet;
for j = tester    
    % with transfer learning + transformation
    X = newfeats(j-1,:); Xfull = [X(:,topSet), ones(size(X,1),1)];    
    y_nxt = Xfull*bestEst2'; % full estimate
    estfeats2(j-tester(1)+1,:) = y_nxt;    
end

%% Version 3: AR(1) + transfer learning + manifold alignment
bestEst3 = tikhFunc3(wolearn,transSet,tPrime,valSet, topSet);

% predict next feature values and check the errors
estfeats3 = zeros(size(tester,2),size(coreSet,2));
newfeats = coreSet;
for j = tester    
    % with transfer learning + transformation
    X = newfeats(j-1,:); Xfull = [X(:,topSet), ones(size(X,1),1)];    
    y_nxt = Xfull*bestEst3'; % full estimate
    estfeats3(j-tester(1)+1,:) = y_nxt;    
end

%%
% check the errors of the estimates of the features
T = zeros(size(tester,2),1);
T2 = zeros(size(tester,2),1);
T3 = zeros(size(tester,2),1);
for j = 1 : size(T,1)
    T(j) = (norm(coreSet(tester(j),:) - estfeats(j,:),'fro'));
    T2(j) = (norm(coreSet(tester(j),:) - estfeats2(j,:),'fro'));
    T3(j) = (norm(coreSet(tester(j),:) - estfeats3(j,:),'fro'));
end

% smoothen the errors by windowing - calculate forward differences
winSize = 30;
for j = 1 : size(T,1) - winSize
    temp = T(j:j+winSize-1);
    T(j) = trapz(temp);
    
    temp = T2(j:j+winSize-1);
    T2(j) = trapz(temp);
    
    temp = T3(j:j+winSize-1);
    T3(j) = trapz(temp);
end

% AR only
% disp('AR Analysis')
[~,~,~,where] = seizfinAnalysis(T,1,S_start,S_end,coreName);
where3 = where;
where2 = where3;

% AR + transfer learning
% disp('AR + TransferL Analysis')
[~,~,~,where2] = seizfinAnalysis(T2,2,S_start,S_end,coreName);

% AR + transformation learning
% disp('AR + TransformationL Analysis')
[~,~,~,where3] = seizfinAnalysis(T3,3,S_start,S_end,coreName);
% disp([where, where2, where3]);
% disp([finVal, finVal2, finVal3]);
end
%% 

function [bestEst] = tikhFunc3(partEst,dataprime,tPrime,valSet, topSet)
% with manifold alignment and transfer learning

bestEst = zeros(size(partEst,1),size(partEst,2),size(dataprime,1));
lamVal = zeros(size(dataprime,2),1); err = zeros(size(dataprime,2),1);
allSaver = cell(size(dataprime,1),1);
% iteratively run over all data sets finding the one with the least error,
% then add more data set parameters to reduce the error further
for a = 1 : size(dataprime,1)
    load(dataprime{a}, 'combFeatICA');
    tfeats = combFeatICA;
    
    % transform the transfer data
    tData = transformer(tfeats,tPrime);
    
    % create matrix of form b0 + b1*x1(t-1) + b2*x2(t-1) + ...
    X = tData(1:end-1,:); Xfull = [X(:,topSet), ones(size(X,1),1)];
    Y = tData(2:end,:); % normalized feature vector
    
    % find coefficients of regression
    pX = pinv(Xfull)*Y; hPar = pX'; allSaver{a} = hPar;       
    [bestEst(:,:,a), lamVal(a), err(a)] = errCalc(hPar,partEst,tPrime,valSet, topSet);
end

% take the parameter set estimate with the least error in estimation of the
% following validation set features
[~,idx] = min(err);
lambda = lamVal(idx)*size(tPrime,1)/(size(tPrime,1)+size(valSet,1));

x0 = [tPrime; valSet(1:end-1,:)]; x0 = [x0(:,topSet),ones(size(x0,1),1)];
x1 = [tPrime(2:end,:); valSet]; hPar = allSaver{idx}; g = size(hPar,2);
bestEst = (hPar'+pinv(x0'*x0+lambda*eye(g,g))*(x0'*(x1-x0*hPar')))';
end
%% 

function [bestEst] = tikhFunc2(partEst,dataprime,tPrime,valSet, topSet)
% with only transfer learning

bestEst = zeros(size(partEst,1),size(partEst,2),size(dataprime,1));
lamVal = zeros(size(dataprime,2),1); err = zeros(size(dataprime,2),1);
allSaver = cell(size(dataprime,1),1);
% iteratively run over all data sets finding the one with the least error,
% then add more data set parameters to reduce the error further
for a = 1 : size(dataprime,1)
    load(dataprime{a}, 'combFeatICA');
    tData = combFeatICA;
    
    % create matrix of form b0 + b1*x1(t-1) + b2*x2(t-1) + ...
    X = tData(1:end-1,:); Xfull = [X(:,topSet), ones(size(X,1),1)];
    Y = tData(2:end,:); % normalized feature vector
    
    % find coefficients of regression
    pX = pinv(Xfull)*Y; hPar = pX'; allSaver{a} = hPar;       
    [bestEst(:,:,a), lamVal(a), err(a)] = errCalc(hPar,partEst,tPrime,valSet, topSet);
end

% take the parameter set estimate with the least error in estimation of the
% following validation set features
[~,idx] = min(err);
lambda = lamVal(idx)*size(tPrime,1)/(size(tPrime,1)+size(valSet,1));

x0 = [tPrime; valSet(1:end-1,:)]; x0 = [x0(:,topSet),ones(size(x0,1),1)];
x1 = [tPrime(2:end,:); valSet]; hPar = allSaver{idx}; g = size(hPar,2);
bestEst = (hPar'+pinv(x0'*x0+lambda*eye(g,g))*(x0'*(x1-x0*hPar')))';
end
%%

function [bestEst,lamVal,minVal] = errCalc(hPar,partEst,tPrime,valSet, topSet)
bestEst = partEst; [m,n] = size(tPrime); minVal = Inf; [~,g] = size(hPar);
sizeTop = size(topSet,2);

x0 = [tPrime(1:end-1,topSet),ones(m-1,1)]; x1 = tPrime(2:end,:);
A = x0'*x0; B = x0'*(x1 - x0*hPar');

combSet = zeros(size(valSet,1),sizeTop+1);
combSet(:,1:sizeTop) = [tPrime(end,topSet); valSet(1:end-1,topSet)];
combSet(:,sizeTop+1) = ones(size(valSet,1),1);

for tikh = 0 : 0.01 : 1    
    estNow = (hPar'+pinv(A + tikh*eye(g,g))*B)';
    
    % use this estimate to predict in the validation set
    estfeats = zeros(size(valSet,1),n);
    for j = 1 : size(valSet,1)
        y_est = combSet(j,:)*estNow'; estfeats(j,:) = y_est;
    end
    
    % estfeats = combSet*estNow';
    errSet = norm(estfeats - valSet,'fro');
    if errSet < minVal
        bestEst = estNow; lamVal = tikh; minVal = errSet;
    end
end
end
%%

function tData = transformer(transfeats,tPrime)
rankC = rank(tPrime); rankTr = rank(transfeats);
[Uc,~,Vc] = svds(tPrime,rankC); 
rankC = size(Uc, 2);
[Ut,St,~] = svds(transfeats,rankTr);
rankTr = size(Ut, 2);
if rankTr > rankC
    Vzeros = zeros(size(Vc,1),rankTr - rankC);
    Uzeros = zeros(size(Uc,1),rankTr - rankC);
    Vprime = [Vc, Vzeros]; Uprime = Ut; 
    % Uprime = [Uc, Uzeros];
else
    Szeros = zeros(size(St,1),rankC - rankTr);
    St = [St; Szeros']; 
    Szeros = zeros(size(St,1),rankC - rankTr); St = [St, Szeros];
    Vprime = Vc; % Uprime = Uc;
    Uzeros = zeros(size(Ut,1),rankC - rankTr);
    Uprime = [Ut, Uzeros];
end
tData = (Uprime*St*Vprime');
end
%%

function [bestScore,finVal,topPt,where] = seizfinAnalysis(T,type,S_start,S_end,coreName)
% detect the seizure
winSize = 20; flagger = 0; sph = 120;
smWin = 1; where = []; % vars for old logic
prediction = [];
errors = zeros(size(T,1) - winSize - smWin,1);
winstd = errors;
step = floor(winSize/2);
for j = 1 : size(T,1) - winSize - smWin
    winErr = T(j:j+winSize-1); winMu = mean(winErr); winStd = std(winErr);
    smwinErr = T(j+winSize+smWin);

    errDev = smwinErr - winMu;
    errors(j) = errDev;
    winstd(j) = 3*winStd;
    if errDev > 3*winStd
        newErr = (j+winSize-1+smWin);
        if ~isempty(where)            
            if newErr - where(end) == step
                where = [where; newErr];
                flagger = flagger + 1;
                if flagger > 3
                    % predict seizure here
                    prediction = [prediction; where(end)];
                    where = [];
                    flagger = 0;
                end
            else
                where = [];
                flagger = 0;
            end
        else
            where = newErr; 
            flagger = flagger + 1;
        end
    end
end
% figure ;plot(errors, 'k'); hold on; plot(winstd, 'r-.');
% for i = prediction
%     line([i i], [min(errors) max(errors)], 'Color', 'b');
% end
% hold off; pause
where = prediction;

% quantifying the seizure prediction
if ~isempty(prediction)
    bestPt = prediction;
    % poisspar(bestPt >= 0) = [];
%     bestPt(bestPt >= S_start) = [];
    topPt = min(bestPt);
    bestScore = topPt/size(bestPt,1);
else
    bestScore = 0; topPt = 0; bestPt = [];
end

% delta discriminator
A = mean(T(S_start-12:S_end+12));
% B = mean(T([1:S_start-13,S_end+13:end]));
B = mean(T(1:S_start-13));
finVal = abs(A-B)/mean(T);

saveDir = 'Results/';
if ~isdir(saveDir)
    mkdir(saveDir)
end
fsize = 20; % font size
fig = figure('visible','off'); 
semilogy(1:size(T,1)-winSize,T(1:end-winSize),'k.-'); hold on;

for ids = 1:length(S_start)
    line([S_start(ids) S_start(ids)],[1 max(T)],'Color','r','LineWidth',2,'LineStyle','--');
    line([S_end(ids) S_end(ids)],[1 max(T)],'Color','r','LineWidth',2,'LineStyle','--');
end
if ~isempty(bestPt)
    for z = 1 : length(bestPt)        
        line([bestPt(z) bestPt(z)],[1 max(T)],'Color','b',...
            'LineWidth',2,'LineStyle','--');
    end
end
set(gca,'FontSize',fsize-2);
if type == 1
    title(['AR(1): Recording-',coreName],...
        'FontName','Arial','FontSize',fsize,'FontWeight','bold');
elseif type == 2
    title(['TL: Recording-',coreName],...
    'FontName','Arial','FontSize',fsize,'FontWeight','bold');
else
    title(['MA: Recording-',coreName],...
    'FontName','Arial','FontSize',fsize,'FontWeight','bold');
end

xlabel('Time in epochs (1 epoch = 5 seconds)','FontSize',fsize,...
    'FontWeight','bold','FontName','Arial'); 
ylabel('Error in prediction','FontSize',fsize,'FontWeight','bold',...
    'FontName','Arial');
if type == 1
    print(fig,'-dtiff','-r200',[saveDir coreName '_AR.tiff']);
elseif type == 2
    print(fig,'-dtiff','-r200',[saveDir coreName '_TL.tiff']);
else
    print(fig,'-dtiff','-r200',[saveDir coreName '_MA.tiff']);
end
% close all;

% fig1 = figure('visible','off'); plot(1:size(poisspar,2),poisspar,'r*-');
% title(['Patient-',int2str(pNum), ' Recording-',int2str(pRec)],'FontName',...
%     'Arial','FontSize',fsize,'FontWeight','bold'); 
% xlabel('Time (in epochs)','FontSize',fsize,'FontWeight','bold',...
%     'FontName','Arial'); 
% ylabel('Poisson parameter values','FontSize',fsize,'FontWeight','bold',...
%     'FontName','Arial');
% if type == 1
%     print(fig1,'-dtiff','-r200',[saveDir coreName '_ZZ_AR.tiff']);
% elseif type == 2
%     print(fig1,'-dtiff','-r200',[saveDir coreName '_ZZ_TL.tiff']);
% else
%     print(fig1,'-dtiff','-r200',[saveDir coreName '_ZZ_MA.tiff']);
% end
end

% % plot inflection points only
% inf_pts = [];
% for j = 3 : size(T,1) - winSize
%     if (T(j) > T(j-1) && T(j) > T(j-2) && T(j) > T(j+1) && T(j) > T(j+2)) ...
%             || (T(j) < T(j-1) && T(j) < T(j-2) && T(j) < T(j+1) && ...
%             T(j) < T(j+2))
%         inf_pts = [inf_pts, j];
%     end
% end

% for j = 1 : size(T,1) - winSize
%     winErr = T(1:j+winSize-1); % continuous window
%     newPar = poissfit(winErr);
%     if j > 10
%         meanPar = mean(poisspar);
%         stdPar = std(winErr);
%         if abs(newPar - meanPar) > 0.35*stdPar % 95% confidence
%             if flagger
%                 % Seizure prediction
%                 where = j + winSize - 1;
%                 break;
%             end
%             flagger = 1;
%         end
%     end    
%     poisspar = [poisspar, newPar];
% end


% if errDev > 0.25*winStd %&& abs(T(j)) > detect_error
%     where = [where; (j+winSize-1+smWin)];
%     flagger = flagger + 1;
%     if flagger > 10
%         oldArrT = mean(where(2:end-1) - where(1:end-2));
%         oldArrStd = std(where(2:end-1) - where(1:end-2));
%         newTime = j+winSize-1+smWin - where(end-1);
%         if oldArrT - newTime > 0.25*oldArrStd
%             % predict seizure here
%             twhere = where(end);
%             break;
%         end
%     end
% end