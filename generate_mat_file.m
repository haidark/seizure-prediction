folder = 'Data/SplitData/';
fileName = 'studyinterictalAH04';
headFile = [folder fileName '_header.txt'];
dataFile = [folder fileName '_data.txt'];

%         Make sure the data file exists
if ~exist(headFile, 'file')
    disp(['ERROR: Header file for does not exist!'])
    exit;
end
if ~exist(dataFile, 'file')
    disp(['ERROR: Data file for does not exist!'])
    exit;
end

disp(headFile)
%         read the header file
hFile = fopen(headFile,'r');
% skip preamble (first 15 lines), retreive sampling rate and number of electrodes
for i = 1 : 15
    tline = fgetl(hFile);
    if i == 8
        % retrieve the sampling rate
        disp('sampling rate:');
        T = strsplit(tline);
        samplingRate = str2num(T{end-1});
        disp(samplingRate);
    elseif i == 9
        % retrieve channels
        disp('electrodes:');
        T = strsplit(tline);
        electrodes = str2num(T{end});
        disp(electrodes);            
    end
end
%         Get number of lines in the data file
tline = fgetl(hFile);
%         retrieve the sampling rate
disp('Number of Lines:');
T = strsplit(tline);
numLines = str2num(T{end});
disp(numLines);
%         Get the seizure start and end times
seizureStart = [];
seizureEnd = [];
tline = fgetl(hFile);
while(tline ~= -1)
    T = strsplit(tline);
    disp('Seizure Start:');
    start = cell2mat(textscan(T{end-1}, '%d/%d/%d-%d:%d:%d'));
    disp(start)
    ending = cell2mat(textscan(T{end}, '%d/%d/%d-%d:%d:%d'));
    seizureStart = [seizureStart; start];
    seizureEnd = [seizureEnd; ending];
    tline = fgetl(hFile);            
end
fclose(hFile);

%         Read the data file into a matrix
dFile = fopen(dataFile, 'r');
seizure_data = zeros(numLines, electrodes);
i = 1;
tline = fgetl(dFile);
T = strsplit(tline);
beginLine = T;          % save the first line
while(tline ~= -1)
%             Replace the words with numbers
    tline = strrep(tline, 'AMPSAT', '10.0');
    tline = strrep(tline, 'SHORT', '0.0');
    T = strsplit(tline);
    temp = str2num(strjoin(T(4:end-1)));
    if isempty(temp) || length(temp) ~= electrodes
%                 There was an error so skip this row
        seizure_data(i,:) = [];
        if i > size(seizure_data,1), break, end
        continue;
    end
%             store the data
    seizure_data(i,:) = temp;
    i = i + 1;
    endLine = T;        % save the last line
    tline = fgetl(dFile);   %get the next line

    if i > size(seizure_data,1)
        break;
    end
end 
fclose(dFile);

% calculate recording start and end times
startDate = cell2mat(textscan(beginLine{1}, '%d/%d/%d'));
startTime = cell2mat(textscan(beginLine{2}, '%d:%d:%d'));
recordingStart = [startDate, startTime];
disp('Recording Start:')
disp(recordingStart);

endDate = cell2mat(textscan(endLine{1}, '%d/%d/%d'));
endTime = cell2mat(textscan(endLine{2}, '%d:%d:%d'));
recordingEnd = [endDate, endTime];
disp('Recording End:')
disp(recordingEnd);

origFileName = [fileName '.txt'];
matFile = fileName;
disp(['Done reading EEG Recording, saving as matfile: ' matFile]);
save(['EEG_Mat/' matFile], 'origFileName', 'seizure_data', 'samplingRate',...
    'seizureStart', 'seizureEnd', 'electrodes', 'recordingStart', 'recordingEnd');

