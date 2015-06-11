clear all;
clc;
close all;

matDir = 'EEG_Mat/';
addpath('FastICA');

%22 seems to be bad
D = dir([matDir '*.mat']);

% for idd = 1: size(D,1)
%     matFile = [matDir D(idd).name];
%     disp(['Processing: ' matFile])
% %     Generate Training Data for this recording
%     disp('(+) generating ICA data...')
%     generate_ICA_train(matFile);
%     
% %     Generate synchronization graphs and signal features
%     disp('(+) generating graphs and signal features...')
%     generate_graphs_feats(matFile);
%     
% %     extract graph metrics
%     disp('(+) generating graph metrics...')
%     generate_graph_metrics(matFile);
%     
% %     combine features
%     disp('(+) combining features...')
%     combine_features(matFile);
%     
%    
% end

% disp('Finished processing data')

for idd = 23:size(D,1)
    matFile = [matDir D(idd).name];
    disp(['Testing #' num2str(idd) ': ' matFile])
    %     generate predictions
    disp('(+) generating predictions...')
    generate_predictions(matFile);
    disp('(+) done!');
%     pause
end
disp('All Finished!')
exit