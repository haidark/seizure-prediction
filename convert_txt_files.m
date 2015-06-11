clc
clear
close all
% This script converts split EEG recordings to .mat files
if ~isdir('./EEG_Mat')
    mkdir('./EEG_Mat');
end

splitDataDir = 'Data/SplitData/';

patientNames = {'AH04', 'BD01', 'CP12', 'EF12', 'EG04', 'FS04', 'KW04', ...
    'LJ04', 'MD06', 'TH12'};

for idp = 1: size(patientNames,2);
    pName = patientNames{idp};
    
%     Get Header files for this patients recordings
    patHeadsGlob = sprintf([splitDataDir '*%s_header.txt'], pName);
    patHeads = dir(patHeadsGlob);
%     For each header file
    for idh = 1:size(patHeads,1)
        headFile = [splitDataDir patHeads(idh).name];
        dataFile = [headFile(1:end-11) '_data.txt'];
%         Make sure the data file exists
        if ~exist(dataFile, 'file')
            disp(['ERROR: Data file for ' headFile  ' does not exist!'])
            continue;
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
        
%         calculate recording start and end times
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
        
        origFileName = patHeads(idh).name;
        matFile = [pName '_' num2str(idh)];
        disp(['Done reading EEG Recording, saving as matfile: ' matFile]);
        save(['EEG_Mat/' matFile], 'origFileName', 'seizure_data', 'samplingRate',...
            'seizureStart', 'seizureEnd', 'electrodes', 'recordingStart', 'recordingEnd');
    end

end 
exit