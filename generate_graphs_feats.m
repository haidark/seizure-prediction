function generate_graphs_feats(matFile)
% generates Graph Adjacency lists for the EEG recordings

% File where the recording is saved.
% matFile = 'EEG_Mat/AH04_1.mat';

% PLV - Phase Locking Value
% For PLV, the higher the values the higher the coupling between the channels.
% We will use random sampling instead of thresholds

% Epoch length (in number of samples)
epochLen = 50;

% Overlapping percentage between successive epochs
overlap = 20;

load(matFile, 'seizure_data', 'icaData', 'samplingRate');

% Compute number of epochs that would be analyzed
% BOztan 11/22/2013 Made changes to T to capture 200 Hz option as well.
coeff = ceil(size(seizure_data,1) / (epochLen / 10 * samplingRate)...
    * 1 / (1 - overlap / 100));
if ((epochLen / 10 * samplingRate) * overlap / 100 + coeff * (...
        epochLen / 10 * samplingRate) * (1 - overlap / 100))...
        >= size(seizure_data,1)
    if ((epochLen / 10 * samplingRate) * overlap / 100 + (coeff ...
            - 1) * (epochLen / 10 * samplingRate) * (1 - ...
            overlap / 100)) >= size(seizure_data,1)
        N = coeff - 1;
    else
        N = coeff;
    end
end

% create a directory for adjacency lists
file_out_dir = strrep(matFile, '.mat', '');
if ~isdir(file_out_dir)
    mkdir(file_out_dir);
end
if ~isdir([file_out_dir '_ica'])
    mkdir([file_out_dir '_ica']);
end


data = seizure_data;
icadata = icaData;
node_list = 1 : size(data,2);
epochFeats = zeros(N,10);     % store epoch-based original signal features
epochFeatsICA = zeros(N,10);

for epoch_index = 1 : N
%% ----- Regular data -----------------------------------------------------
%     Get epoch data
    if epoch_index == N
        if (epochLen / 10 * samplingRate) * overlap / 100 + ...
                epoch_index * (epochLen / 10 * samplingRate) * (1 ...
                - overlap / 100) > size(seizure_data,1)
            epoch_data = data(size(seizure_data,1) - (epochLen ...
                /10*samplingRate)+1:size(seizure_data,1),:);
        else
            epoch_data = data((epoch_index-1)* (epochLen / 10 * ...
                samplingRate) * (1 - overlap / 100) + ...
                1 : (epochLen / 10 * samplingRate) * overlap / ...
                100 + epoch_index * (epochLen / 10 * samplingRate)...
                * (1 - overlap / 100),:);
        end
    else
        epoch_data = data((epoch_index-1)* (epochLen / 10 * ...
            samplingRate) * (1 - overlap / 100) +1:...
            (epochLen / 10 * samplingRate) * overlap / 100 ...
            + epoch_index * (epochLen / 10 * samplingRate) * (1 ...
            - overlap / 100),:);
    end

%     Compute features on the epoch - mean, variance, delta
%     mean, delta variance, change variance
    epochFeats(epoch_index,1) = mean(epoch_data(:));
    epochFeats(epoch_index,2) = var(epoch_data(:));
    startRange = epoch_data(1:end-1,:);
    endRange = epoch_data(2:end,:);
    tempData = endRange - startRange;
    epochFeats(epoch_index,3) = mean(tempData(:));
    epochFeats(epoch_index,4) = var(tempData(:));

%     look back up to 3 epoch windows for change-based features
    if epoch_index == 1
        epochFeats(epoch_index,5) = 0;
    elseif epoch_index == 2
        tempData = epoch_data - epochFeats(epoch_index-1,1);
        epochFeats(epoch_index,5) = var(tempData(:));
    elseif epoch_index >= 3
        tempData = epoch_data - epochFeats(epoch_index-1,1) ...
            - epochFeats(epoch_index-2,1);
        epochFeats(epoch_index,5) = var(tempData(:));
    end
    
%     Compute important features from Evrim's paper
%     Hjorth parameters - activity, mobility, complexity
    activity = var(epoch_data(:));
    epochFeats(epoch_index,6) = activity;

    tempData = startRange - endRange;
    diff1 = std(tempData(:));
    mobility = diff1/sqrt(activity);
    epochFeats(epoch_index,7) = mobility;

    startRange = epoch_data(1:end-2,:);
    endRange = epoch_data(3:end,:);
    tempData = endRange - startRange;
    complexity = (std(tempData(:))/diff1)/(diff1/sqrt(activity));
    epochFeats(epoch_index,8) = complexity;

    
%     skewness of amplitude spectrum, spectral entropy
    skAmp = zeros(size(epoch_data,2),1);
    specEnt = zeros(size(epoch_data,2),1);
    for iter = 1 : size(epoch_data,2)
        this_epoch = epoch_data(:,iter);
        Fthis = fft(this_epoch);
        ampSpec = abs(Fthis);
        skAmp(iter) = skewness(ampSpec);

        ampSpec = ampSpec/sum(ampSpec);
        specEnt(iter) = -sum(ampSpec.*log2(ampSpec));
    end
    skAmp(isnan(skAmp)) = 0; skAmp = mean(skAmp);
    specEnt(isnan(specEnt)) = 0; specEnt = mean(specEnt);
    epochFeats(epoch_index,9) = skAmp;
    epochFeats(epoch_index,10) = specEnt;
    
%     Compute the pair-wise PLV
    dist_list_plv = zeros(size(epoch_data,2),size(epoch_data,2));

    for node_1 = 1 : size(epoch_data,2)
        for node_2 = 1 : size(epoch_data,2)
            if node_1 ~= node_2
%                 Phase Locking Value between the channels
                PLV = exp(1i * (angle(hilbert(epoch_data(:,node_1)...
                    )) - angle(hilbert(epoch_data(:,node_2)))));
%                 Take the average along the epoch.
                dist_list_plv(node_1,node_2) = abs(mean(PLV));
            end
        end
    end


%     Instead of using thresholds as was previously done; we will
%     generate numbers uniformly at random (plv_threshold) and place an edge if
%     the number is below the PLV number. (since PLV = [0,1] and higher
%     indicates more coupling)
%     Haidar Khan - 5/7/15

    rng('shuffle') %seed the random number generator with the current time
    plv_threshold = rand(size(dist_list_plv));
    adj_mtx_plv = (dist_list_plv >= plv_threshold);

%     Generate Adjecency Lists and Write to file
    adj_list_name_plv = sprintf('%s/adj_%d.txt',...
        file_out_dir,epoch_index);

    fid_plv = fopen(adj_list_name_plv,'wt');

    for node_index = 1 : size(data,2)
        fprintf(fid_plv,'%d %d ',node_index,sum(...
            adj_mtx_plv(node_index,:)));
        if sum(adj_mtx_plv(node_index,:))
            edges = setdiff(unique(node_list .* ...
                adj_mtx_plv(node_index,:)),0);
            fprintf(fid_plv,'%d ',edges);
        end
        fprintf(fid_plv,'\r\n');
    end
    fclose(fid_plv);
% ------ END regular data ------------------------------------------------ 
%% ------ Same thing on ICA data -----------------------------------------   
%     Get epoch data from ICA signal
    if epoch_index == N
        if (epochLen / 10 * samplingRate) * overlap / 100 + ...
                epoch_index * (epochLen / 10 * samplingRate) * (1 ...
                - overlap / 100) > size(seizure_data,1)
            epoch_data_ica = icadata(size(seizure_data,1) - (epochLen ...
                /10*samplingRate)+1:size(seizure_data,1),:);
        else
            epoch_data_ica = icadata((epoch_index-1)* (epochLen / 10 * ...
                samplingRate) * (1 - overlap / 100) + ...
                1 : (epochLen / 10 * samplingRate) * overlap / ...
                100 + epoch_index * (epochLen / 10 * samplingRate)...
                * (1 - overlap / 100),:);
        end
    else
        epoch_data_ica = icadata((epoch_index-1)* (epochLen / 10 * ...
            samplingRate) * (1 - overlap / 100) +1:...
            (epochLen / 10 * samplingRate) * overlap / 100 ...
            + epoch_index * (epochLen / 10 * samplingRate) * (1 ...
            - overlap / 100),:);
    end
%     Compute features on the epoch - mean, variance, delta
%     mean, delta variance, change variance
    
    epochFeatsICA(epoch_index,1) = mean(epoch_data_ica(:));
    epochFeatsICA(epoch_index,2) = var(epoch_data_ica(:));
    startRange = epoch_data_ica(1:end-1,:);
    endRange = epoch_data_ica(2:end,:);
    tempData = endRange - startRange;
    epochFeatsICA(epoch_index,3) = mean(tempData(:));
    epochFeatsICA(epoch_index,4) = var(tempData(:));
    
%     look back up to 3 epoch windows for change-based features
    if epoch_index == 1
        epochFeatsICA(epoch_index,5) = 0;
    elseif epoch_index == 2
        tempData = epoch_data_ica - epochFeatsICA(epoch_index-1,1);
        epochFeatsICA(epoch_index,5) = var(tempData(:));
    elseif epoch_index >= 3
        tempData = epoch_data_ica - epochFeatsICA(epoch_index-1,1) ...
            - epochFeatsICA(epoch_index-2,1);
        epochFeatsICA(epoch_index,5) = var(tempData(:));
    end
    
%     Compute important features from Evrim's paper
%     Hjorth parameters - activity, mobility, complexity
    activity = var(epoch_data_ica(:));
    epochFeatsICA(epoch_index,6) = activity;

    tempData = startRange - endRange;
    diff1 = std(tempData(:));
    mobility = diff1/sqrt(activity);
    epochFeatsICA(epoch_index,7) = mobility;

    startRange = epoch_data_ica(1:end-2,:);
    endRange = epoch_data_ica(3:end,:);
    tempData = endRange - startRange;
    complexity = (std(tempData(:))/diff1)/(diff1/sqrt(activity));
    epochFeatsICA(epoch_index,8) = complexity;

%     skewness of amplitude spectrum, spectral entropy
    skAmp = zeros(size(epoch_data_ica,2),1);
    specEnt = zeros(size(epoch_data_ica,2),1);
    for iter = 1 : size(epoch_data_ica,2)
        this_epoch = epoch_data_ica(:,iter);
        Fthis = fft(this_epoch);
        ampSpec = abs(Fthis);
        skAmp(iter) = skewness(ampSpec);

        ampSpec = ampSpec/sum(ampSpec);
        specEnt(iter) = -sum(ampSpec.*log2(ampSpec));
    end
    skAmp(isnan(skAmp)) = 0; skAmp = mean(skAmp);
    specEnt(isnan(specEnt)) = 0; specEnt = mean(specEnt);
    epochFeatsICA(epoch_index,9) = skAmp;
    epochFeatsICA(epoch_index,10) = specEnt;
    
%     Compute the pair-wise PLV
    dist_list_plv = zeros(size(epoch_data_ica,2),size(epoch_data_ica,2));

    for node_1 = 1 : size(epoch_data_ica,2)
        for node_2 = 1 : size(epoch_data_ica,2)
            if node_1 ~= node_2
%                 Phase Locking Value between the channels
                PLV = exp(1i * (angle(hilbert(epoch_data_ica(:,node_1)...
                    )) - angle(hilbert(epoch_data_ica(:,node_2)))));
%                 Take the average along the epoch.
                dist_list_plv(node_1,node_2) = abs(mean(PLV));
            end
        end
    end


%     Instead of using thresholds as was previously done; we will
%     generate numbers uniformly at random (plv_threshold) and place an edge if
%     the number is below the PLV number. (since PLV = [0,1] and higher
%     indicates more coupling)
%     Haidar Khan - 5/7/15

    rng('shuffle') %seed the random number generator with the current time
    plv_threshold = rand(size(dist_list_plv));
    adj_mtx_plv = (dist_list_plv >= plv_threshold);

%     Generate Adjecency Lists and Write to file
    adj_list_name_plv = sprintf('%s_ica/adj_%d.txt',...
        file_out_dir,epoch_index);

    fid_plv = fopen(adj_list_name_plv,'wt');

    for node_index = 1 : size(icadata,2)
        fprintf(fid_plv,'%d %d ',node_index,sum(...
            adj_mtx_plv(node_index,:)));
        if sum(adj_mtx_plv(node_index,:))
            edges = setdiff(unique(node_list .* ...
                adj_mtx_plv(node_index,:)),0);
            fprintf(fid_plv,'%d ',edges);
        end
        fprintf(fid_plv,'\r\n');
    end
    fclose(fid_plv);
% ------  END ICA data --------------------------------------------- 
end
save(matFile, 'epochFeats', 'epochFeatsICA', '-append');


end


