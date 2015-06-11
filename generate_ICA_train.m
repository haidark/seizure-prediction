% find seizure independent component
function generate_ICA_train(matFile)

%     matFile = 'EEG_Mat/AH04_1.mat';
    load(matFile, 'seizure_data');

    % Modified Gram-Schmidt process - returns indices of vectors
    bestix = modgramsc(seizure_data);

    % Run ICA with number of components determined from modified GS
    [Out1, mixMat, ~] = fastica(seizure_data','numOfIC',bestix+1);
    icaData = Out1';

    % find and store all the norms
    sqMat = mixMat; colnorm = zeros(size(sqMat,2),1);
    for z = 1 : size(sqMat,2)
        colnorm(z) = norm(sqMat(:,z));
    end
    [~,idx1] = sort(colnorm);
    mixMat(idx1(1:2),:) = 0; % remove the bottom 2 based on column norm

    % reconstitute signal
    icaData = icaData*mixMat';

    %         % Method 2: find seizure IC and remove it from analysis
    %         m = size(icaData,1); delta = (winSize - 1)/2;
    %         Y = zeros(m,size(icaData,2));
    %         for cntr = 1 : m
    %             left = max(1, cntr - delta);
    %             right = min(m, cntr + delta);
    %             Y(cntr,:) = mean(abs(icaData(left:right,:)));
    %         end
    %         
    %         meanY = mean(Y, 1);
    %         for icas = 1 : size(icaData, 2)
    %             Y(:,icas) = Y(:,icas) - meanY(icas);
    %         end
    save(matFile, 'icaData', '-append');
end
%% 

function totes = find_best_seizure_IC(icaData,S_start,S_end)
% create an indicator vector size of seizure data, set 1s in seizure region
% minVal = Inf; idx = 0;
totes = zeros(size(icaData,2),1);
mu = zeros(size(icaData,2),1);
% determine seizure independent component
for i = 1 : size(icaData,2)
    % only in seizure region
    icaData(:,i) = icaData(:,i)/norm(icaData(:,i));
    icaSeiz = icaData(S_start:S_end,i);
    icaNonSeiz = icaData([1:S_start-1, S_end+1:end],1);
    mu(i,1) = mean(icaSeiz);
    stddev = sum((icaSeiz - mu(i,1)).^2)/size(icaSeiz,1);
    sqNonSeiz = sum(icaNonSeiz.^2);
    totes(i,1) = 1 - (sqNonSeiz + stddev);
%     if totes < minVal
%         minVal = totes;
%         idx = i;
%     end
end

end
%% 

function [marker] = modgramsc(seizure_data)

marker = size(seizure_data,2);
for j = 1 : size(seizure_data,2)
    % center the data vectors
    seizure_data(:,j) = seizure_data(:,j) - mean(seizure_data(:,j));
end

% Create functions for each vector
seizure_func = zeros(size(seizure_data,1),size(seizure_data,2));
for j = 1 : size(seizure_func,2)
    seizure_func(:,j) = exp(1i*seizure_data(:,j)/(2*pi));
    
    % center the functionals
    seizure_func(:,j) = seizure_func(:,j) - mean(seizure_func(:,j));
end

normStoreData = zeros(size(seizure_data,2),1);
normStoreFunc = zeros(size(seizure_data,2),1);
for k = 1 : size(seizure_data,2)
    normStoreData(k) = norm(seizure_data(:,k));
    normStoreFunc(k) = norm(seizure_func(:,k));
    
    % normalize data vector
    seizure_data(:,k) = seizure_data(:,k)/normStoreData(k);
    
    % orthogonalize functional to data vector
    C1= ctranspose(seizure_data(:,k));
    C1 = C1 * seizure_func(:,k);
    seizure_func(:,k) = seizure_func(:,k) - seizure_data(:,k)*C1;
    
    % normalize functional
    seizure_func(:,k) = seizure_func(:,k)/normStoreFunc(k);
    
    % for loop
    for z = k + 1 : size(seizure_data,2)
        C1 = ctranspose(seizure_data(:,k));
        C1 = C1 * seizure_data(:,z);
        C2 = ctranspose(seizure_func(:,k));
        C2 = C2 * seizure_data(:,z);
        seizure_data(:,z) = seizure_data(:,z) - seizure_data(:,k)*C1 - ...
            seizure_func(:,k)*C2;
        
        C1 = ctranspose(seizure_data(:,k));
        C1 = C1 * seizure_func(:,z);
        C2 = ctranspose(seizure_func(:,k));
        C2 = C2 * seizure_func(:,z);
        seizure_func(:,z) = seizure_func(:,z) - seizure_data(:,k)*C1 - ...
            seizure_func(:,k)*C2;
    end
end

% throwing away unnecessary vectors, take only 90%
totesData = sum(normStoreData); totesFunc = sum(normStoreFunc);
% sort the norms
dataVecs = sort(normStoreData,'descend');
funcVecs = sort(normStoreFunc,'descend');

runSum1 = 0; runSum2 = 0;
for z = 1 : size(normStoreData,1)
    runSum1 = runSum1 + dataVecs(z);
    runSum2 = runSum2 + funcVecs(z);
    if runSum1 >= 0.9*totesData
        marker = z; break;
    end
    if runSum2 >= 0.9*totesFunc
        marker = z; break;
    end
end

end

function [marker] = plaingramsc(seizure_data)

marker = size(seizure_data,2);
for j = 1 : size(seizure_data,2)
    % center the data vectors
    seizure_data(:,j) = seizure_data(:,j) - mean(seizure_data(:,j));
end

normStoreData = zeros(size(seizure_data,2),1);
for k = 1 : size(seizure_data,2)
    normStoreData(k) = norm(seizure_data(:,k));
    
    % normalize data vector
    seizure_data(:,k) = seizure_data(:,k)/normStoreData(k);
        
    % for loop
    for z = k + 1 : size(seizure_data,2)
        C1 = seizure_data(:,k)';
        C1 = C1 * seizure_data(:,z);        
        seizure_data(:,z) = seizure_data(:,z) - seizure_data(:,k)*C1;                
    end
end

% throwing away unnecessary vectors, take only 90%
totesData = sum(normStoreData);
% sort the norms
dataVecs = sort(normStoreData,'descend');

runSum1 = 0;
for z = 1 : size(normStoreData,1)
    runSum1 = runSum1 + dataVecs(z);
    if runSum1 >= 0.9*totesData
        marker = z; break;
    end    
end

end
