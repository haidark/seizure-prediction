function combine_features(matFile)

% matFile = 'EEG_Mat/AH04_1.mat';

winSize = 20;
keepmet = [1:7 11 14:20 22:30 32:35];
keepmet = setdiff(keepmet,[4 13 19]);

%        load the number of electrodes from the original mat file
load(matFile, 'electrodes', 'epochFeats', 'metricsMat', 'epochFeatsICA', 'metricsMatICA');
% original signal features
sigFeats = epochFeats;
sigFeatsICA = epochFeatsICA;
% graph metric features
gMets = metricsMat(:,keepmet);
gMetsICA = metricsMatICA(:,keepmet);
% all metrics combined
allMets = [gMets sigFeats];
allMetsICA = [gMetsICA sigFeatsICA];

numEpochs = size(epochFeats, 1);

%% Testing Data 
adjTen = zeros(numEpochs,electrodes,electrodes); 
degTen = zeros(numEpochs,electrodes,electrodes);
lapTen = zeros(numEpochs,electrodes,electrodes);

for k = 1 : numEpochs
    % calculate spectral metrics
    adjList = fopen([strrep(matFile, '.mat', '') '/adj_' int2str(k) '.txt'],'r');          

    Adj = zeros(electrodes,electrodes); 
    Deg = zeros(electrodes,electrodes); 
    l = 1; 
    
%     tline = fgets(adjList);
%     C1 = str2num(tline); 
%     Adj(l,C1(3:end)) = 1;
%     Deg(l,l) = C1(2); l = l + 1;
    while ~feof(adjList)
        tline = fgets(adjList);
        C1 = str2num(tline); 
        Adj(l,C1(3:end)) = 1;
        Deg(l,l) = C1(2); l = l + 1;
    end
    fclose(adjList);
    adjTen(k,:,:) = Adj; 
    degTen(k,:,:) = Deg;
    lapTen(k,:,:) = (Deg - Adj);
end

% average features over 20 epochs
for w = winSize
    
    varEig = zeros(numEpochs,1);
    varLap = zeros(numEpochs,1);
    metMean = zeros(numEpochs,size(allMets,2));
    
    for z = 1 : numEpochs
        beg = z-w+1;
        if beg < 1
            beg = 1;
        end
        
        adjCur = adjTen(beg:z,:,:);
        adjCur = reshape(sum(adjCur,1),electrodes,electrodes);

        degCur = degTen(beg:z,:,:);
        degCur = reshape(sum(degCur,1),electrodes,electrodes);

        lapCur = lapTen(beg:z,:,:);
        lapCur = reshape(sum(lapCur,1),electrodes,electrodes);

        invVal = adjCur*pinv(degCur,1);
        [E,~] = eig(invVal);
        E = E/sum(E);
        varEig(z) = var(E);

        [~,V] = eig(lapCur);
        V = diag(V); V = sort(V,'ascend');
        varLap(z) = V(2);

        metMean(z,:) = mean(allMets(beg:z,:));
    end
end

combFeat = real([metMean, varEig, varLap]); % combine graph and spectral features

[~,col] = find(isnan(metMean)); % remove NaN values from matrix
col = unique(col); 
combFeat(:,col) = 0;

% this is a questionable step...
combFeat = zscore(combFeat); % apply zscore to features matrix
save(matFile, 'combFeat', '-append');

%% Training Data
adjTen = zeros(numEpochs,electrodes,electrodes); 
degTen = zeros(numEpochs,electrodes,electrodes);
lapTen = zeros(numEpochs,electrodes,electrodes);

for k = 1 : numEpochs
    % calculate spectral metrics
    adjList = fopen([strrep(matFile, '.mat', '_ica') '/adj_' int2str(k) '.txt'],'r');          

    Adj = zeros(electrodes,electrodes); 
    Deg = zeros(electrodes,electrodes); 
    l = 1; 
    
%     tline = fgets(adjList);
%     C1 = str2num(tline); 
%     Adj(l,C1(3:end)) = 1;
%     Deg(l,l) = C1(2); l = l + 1;
    while ~feof(adjList)
        tline = fgets(adjList);
        C1 = str2num(tline); 
        Adj(l,C1(3:end)) = 1;
        Deg(l,l) = C1(2); l = l + 1;
    end
    fclose(adjList);
    adjTen(k,:,:) = Adj; 
    degTen(k,:,:) = Deg;
    lapTen(k,:,:) = (Deg - Adj);
end

% average features over 20 epochs
for w = winSize
    
    varEig = zeros(numEpochs,1);
    varLap = zeros(numEpochs,1);
    metMean = zeros(numEpochs,size(allMetsICA,2));
    
    for z = 1 : numEpochs
        beg = z-w+1;
        if beg < 1
            beg = 1;
        end
        
        adjCur = adjTen(beg:z,:,:);
        adjCur = reshape(sum(adjCur,1),electrodes,electrodes);

        degCur = degTen(beg:z,:,:);
        degCur = reshape(sum(degCur,1),electrodes,electrodes);

        lapCur = lapTen(beg:z,:,:);
        lapCur = reshape(sum(lapCur,1),electrodes,electrodes);

        invVal = adjCur*pinv(degCur,1);
        [E,~] = eig(invVal);
        E = E/sum(E);
        varEig(z) = var(E);

        [~,V] = eig(lapCur);
        V = diag(V); V = sort(V,'ascend');
        varLap(z) = V(2);

        metMean(z,:) = mean(allMetsICA(beg:z,:));
    end
end

combFeatICA = real([metMean, varEig, varLap]); % combine graph and spectral features

[~,col] = find(isnan(metMean)); % remove NaN values from matrix
col = unique(col); 
combFeatICA(:,col) = 0;

% this is a questionable step...
combFeatICA = zscore(combFeatICA); % apply zscore to features matrix
save(matFile, 'combFeatICA', '-append');
end