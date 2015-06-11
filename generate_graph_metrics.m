function generate_graph_metrics(matFile)
% This script creates a batch file to extract graph features from the graph
% adjacency lists created in Generate_EEG_graphs.m file. It also loads the
% generated metrics into a matrix and appends them to the given mat file


% matFile = 'EEG_Mat/AH04_1.mat';
load(matFile, 'electrodes')
testDir = strrep(matFile, '.mat', '');
trainDir = strrep(matFile, '.mat', '_ica');

%% Testing data first

file_out_dir = sprintf('%s_metrics',testDir);
if ~isdir(file_out_dir)
    mkdir(file_out_dir);
end

file_names_dir = sprintf('%s/*.txt',testDir);
file_names = dir(file_names_dir);

% create the batch file
out_name = sprintf('graph_generator_test.sh');
fid = fopen(out_name,'wt');

for file_index = 1 : size(file_names,1)
    name = file_names(file_index).name;
    inFile = sprintf('%s/%s',testDir, name);
    outFile = sprintf('%s/%s', file_out_dir, strrep(name, 'adj', 'met'));
    fprintf(fid,'./GraphMetrics "%s" %d "%s"\n', inFile, electrodes, outFile);
end

fclose(fid);

% make the file executable and run it
ret = system(['chmod 755 ' out_name '; bash ' out_name '; rm ' out_name]);
if ret ~= 0
    disp('ERROR: system commands failed. exiting...')
    exit
end

% Now load the graph metrics data into a large matrix and save it to the
% matFile

load(matFile, 'epochFeats');
numEpochs = size(epochFeats,1);
% get the number of metrics measured by loading the first one
metFile = sprintf('%s/met_%d.txt', file_out_dir, 1);
firstRow = load(metFile);

metricsMat = zeros(numEpochs, size(firstRow, 2));
% for every epoch
for ide = 1:numEpochs
    metFile = sprintf('%s/met_%d.txt', file_out_dir, ide);
    metricsMat(ide,:) = load(metFile);    
end

save(matFile, 'metricsMat', '-append')

%% Now do the same for training data

file_out_dir = sprintf('%s_metrics',trainDir);
if ~isdir(file_out_dir)
    mkdir(file_out_dir);
end

file_names_dir = sprintf('%s/*.txt',trainDir);
file_names = dir(file_names_dir);

out_name = sprintf('graph_generator_train.sh');
fid = fopen(out_name,'wt');

for file_index = 1 : size(file_names,1)
    name = file_names(file_index).name;
    inFile = sprintf('%s/%s',trainDir, name);
    outFile = sprintf('%s/%s', file_out_dir, strrep(name, 'adj', 'met'));
    fprintf(fid,'./GraphMetrics "%s" %d "%s"\n', inFile, electrodes, outFile);
end

fclose(fid);

% make the file executable and run it
ret = system(['chmod 755 ' out_name '; bash ' out_name '; rm ' out_name]);
if ret ~= 0
    disp('ERROR: system commands failed. exiting...')
    exit
end

% Now load the graph metrics data into a large matrix and save it to the
% matFile

load(matFile, 'epochFeatsICA');
numEpochs = size(epochFeatsICA,1);
% get the number of metrics measured by loading the first one
metFile = sprintf('%s/met_%d.txt', file_out_dir, 1);
firstRow = load(metFile);

metricsMatICA = zeros(numEpochs, size(firstRow, 2));
% for every epoch
for ide = 1:numEpochs
    metFile = sprintf('%s/met_%d.txt', file_out_dir, ide);
    metricsMatICA(ide,:) = load(metFile);    
end

save(matFile, 'metricsMatICA', '-append')
end

