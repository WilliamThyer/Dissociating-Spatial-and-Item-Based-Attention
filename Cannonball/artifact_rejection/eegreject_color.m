%% Read Me

%% Setup
clearvars
try
    eeglab
catch err
    if strcmp(err.identifier,'MATLAB:UndefinedFunction')
         addpath 'C:\Program Files\MATLAB\R2021b\eeglab2021.1'
         eeglab
    end
end
% addpath 'C:\Program Files\MATLAB\R2021b\eeglab2021.1'

%% Options
%mPath=fileparts(mfilename('fullpath'));
mPath=pwd;
subjectParentDir = fullfile(mPath(1:find(mPath==filesep,1,'last')), 'raw_data','C01');
% subjectDirectories = {'/01','/08','/09','/11','/13','/15','/20','/21','/22','/23','/24'};  % optionally {} for recursive search
% subjectDirectories = {'/08','/09','/11','/13','/15','/20','/21','/22','/23','/24'};  % exclude 01 because missing port codes
subjectDirectories={'/31','32'};
% subjectDirectories = {'/08'};
lowboundFilterHz = 0.01;
highboundFilterHz = 30;

rerefType = 'mastoid'; % 'none', 'average', or 'mastoid' 
rerefExcludeChans = {'HEOG', 'VEOG', 'StimTrak'};
customEquationList = '';  % optional

EYEEEGKeyword = 'SYNC';
startEvent = 1;
endEvent = 4;
eyeRecorded = 'both';  % 'both', 'left', or 'right'

binlistFile = 'manual_binlist_color.txt';  % if empty, will create one for you
trialStart = -400;
trialEnd = 1500;
baselineStart = -200;
baselineEnd = 0;
rejectionStart = -200;
rejectionEnd = 1000;

eyeMoveThresh = .75;  %deg
distFromScreen = 738; %mm
monitorWidth = 532;  %mm
monitorHeight = 300;  %mm
screenResX = 1920;  %px
screenResY = 1080;  %px

eogThresh = 75; %microv

eegThresh = 80; %microv
eegNoiseThresh = 100; %microv %100 works well for subjects with high alpha

eegResampleRate = 500; %hz

rejEye = true;
rejEog = false;
rejEegThresh = true;
rejEegNoise = true;
rejFlatline = true; %remove trials with any flatline data
eegResample = false;

%% Setup 

% Find all .vhdr files recursively if subjectDirectories is empty
if isempty(subjectDirectories)
    dirs = dir(subjectParentDir);
    for i=1:numel(dirs)
        d = dirs(i).name;
        if strcmp(d, '.') ||  strcmp(d, '..')
            continue
        end
        
        if ~isempty(dir(fullfile(subjectParentDir, d,'color', '*.vhdr')))
            subjectDirectories{end+1} = d; %#ok<SAGROW>
        end
    end
end

log = fopen('log.txt', 'a+t');
fprintf(log, ['Run started: ', datestr(now), '\n\n']);

maximumGazeDist = calcdeg2pix(eyeMoveThresh, distFromScreen, monitorWidth, monitorHeight, screenResX, screenResY);
%% Main loop

for subdir=1:numel(subjectDirectories)
    subdirPath = fullfile(subjectParentDir, subjectDirectories{subdir},'color');
    
    disp(['Running ', subdirPath])
%     fprintf(log, ['Running ', subdirPath, '\n\n']);
    
    vhdrDir = dir(fullfile(subdirPath, '*.vhdr'));
    
    if numel(vhdrDir) == 0
        warning(['Skipping ', subdirPath, '. No vhdr file found.'])
%         fprintf(log, ['Skipping ', subdirPath, '. No vhdr file found.\n\n']);
        continue
    elseif numel(vhdrDir) > 1
        warning(['Skipping ', subdirPath, '. More than one vhdr file found.'])
%         fprintf(log, ['Skipping ', subdirPath, '. More than one vhdr file found.\n\n']);
        continue
    end
    
    vhdrFilename = vhdrDir(1).name;
    
    ascDir = dir(fullfile(subdirPath, '*.asc'));
    edfDir = dir(fullfile(subdirPath, '*.edf'));
    if numel(ascDir) == 0
%        warning(['Skipping ', subdirPath, '. No asc file found.'])
%         fprintf(log, ['Skipping ', subdirPath, '. No vhdr file found.\n\n']);
        fprintf(log,'No .asc file found, looking for an edf file...\n');
        if numel(edfDir) == 1
           disp(['Converting edf file at ',fullfile(edfDir.folder,edfDir.name)])
           system(append('"C:\Program Files (x86)\SR Research\edfconverter\edfconverterW.exe" ',fullfile(edfDir.folder,edfDir.name)))
           ascDir = dir(fullfile(subdirPath, '*.asc'));
        else
            warning(['Skipping ', subdirPath, '. None or more than one edf file found.'])
            continue
        end
    elseif numel(ascDir) > 1
        warning(['Skipping ', subdirPath, '. More than one asc file found.'])
%         fprintf(log, ['Skipping ', subdirPath, '. More than one asc file found.\n\n']);
        continue
    end
    
    ascFullFilename = fullfile(subdirPath, ascDir(1).name);

    EEG = pop_loadbv(subdirPath, vhdrFilename);
    
    EEG.setname = vhdrFilename(1:end-5);
    
    if eegResample
        EEG = pop_resample(EEG,eegResampleRate);
    end
    
    if lowboundFilterHz ~= 0 && highboundFilterHz ~= 0
        fprintf(log, sprintf('Bandpass filtering with lowboundFilterHz = %f and highboundFilterHz=%f\n\n', lowboundFilterHz, highboundFilterHz));
        EEG = pop_basicfilter(EEG, 1:EEG.nbchan, 'Boundary', 'boundary', 'Cutoff', [lowboundFilterHz highboundFilterHz], 'Design', 'butter', 'Filter', 'bandpass', 'Order', 2);
    elseif highboundFilterHz ~= 0
        fprintf(log, sprintf('Lowpass filtering with highboundFilterHz=%f\n\n', highboundFilterHz));
        EEG = pop_basicfilter(EEG, 1:EEG.nbchan, 'Boundary', 'boundary', 'Cutoff', highboundFilterHz, 'Design', 'butter', 'Filter', 'lowpass', 'Order', 2);
    elseif lowboundFilterHz ~= 0
        fprintf(log, sprintf('Highpass filtering with lowboundFilterHz = %f\n\n', lowboundFilterHz));
        EEG = pop_basicfilter(EEG, 1:EEG.nbchan, 'Boundary', 'boundary', 'Cutoff', lowboundFilterHz, 'Design', 'butter', 'Filter', 'highpass', 'Order', 2);
    end
    
    if ~strcmp(rerefType, 'none')
        if ~strcmp(customEquationList, '')
            equationList = customEquationList;
        else
            equationList = get_chan_equations(EEG, rerefType, rerefExcludeChans);
        end
        
        fprintf(log, 'Rereferencing with following equation list:\n');
        fprintf(log, strjoin(equationList, '\n'));
        fprintf(log, '\n\n');
        
        EEG = pop_eegchanoperator(EEG, equationList);
    else
        fprintf(log, 'Skipping rereferencing because rerefType = "none"\n\n');
    end
    
    EYEEEGMatFilename = [ascFullFilename(1:end-4) '_eye.mat'];
    
    fprintf(log, sprintf('Parsing asc file: %s\n\n', ascFullFilename));
    parseeyelink(ascFullFilename, EYEEEGMatFilename, EYEEEGKeyword);

    diary 'log.txt'
    if strcmp(eyeRecorded, 'both')
        EEG = pop_importeyetracker(EEG, EYEEEGMatFilename, [startEvent endEvent], [2 3 5 6], {'L_GAZE_X' 'L_GAZE_Y' 'R_GAZE_X' 'R_GAZE_Y'}, 0, 1, 0, 0);
    else
        EEG = pop_importeyetracker(EEG, EYEEEGMatFilename, [startEvent endEvent], [2 3], {'GAZE_X' 'GAZE_Y'}, 0, 1, 0, 0);
    end
    diary off

    EEG = pop_creabasiceventlist(EEG, 'AlphanumericCleaning', 'on', 'BoundaryNumeric', {-99}, 'BoundaryString', {'boundary'}, 'Warning', 'off');
    
    if isempty(binlistFile)
        make_binlist(subdirPath, timelockCodes)
        binlistFile = fullfile(subdirPath, 'binlist.txt');
    end
    
    EEG = pop_binlister(EEG, 'BDF', binlistFile);
    EEG = pop_epochbin(EEG, [trialStart, trialEnd], sprintf('%d %d', baselineStart, baselineEnd));

    % PERFORM ARTIFACT REJECTION %
    allChanNumbers = 1:EEG.nbchan;
    
    if rejEye %flags trials where absolute eyetracking value is greater than maximumGazeDist
        eyetrackingIDX = allChanNumbers(ismember({EEG.chanlocs.labels}, {'L_GAZE_X','L_GAZE_Y','R_GAZE_X','R_GAZE_Y','GAZE_X','GAZE_Y'}));    
        EEG = pop_artextval(EEG , 'Channel',  eyetrackingIDX, 'Flag',  1, 'Threshold', [-maximumGazeDist maximumGazeDist], 'Twindow', [rejectionStart rejectionEnd]);
    end
    
    if rejEog %flags trials where absolute EOG value is greather than eogThresh
        eogIDX = allChanNumbers(ismember({EEG.chanlocs.labels}, {'HEOG','VEOG'}));    
        EEG = pop_artextval(EEG , 'Channel',  eogIDX, 'Flag',  2, 'Threshold', [-eogThresh eogThresh], 'Twindow', [rejectionStart rejectionEnd]);
    end
    
    eegIDX = allChanNumbers(~ismember({EEG.chanlocs.labels}, {'L_GAZE_X','L_GAZE_Y','R_GAZE_X','R_GAZE_Y','GAZE_X','GAZE_Y','HEOG','VEOG','StimTrak'}));
    if rejEegThresh %flags trials where absolute EEG value is greater than eegThresh
        EEG = pop_artextval(EEG , 'Channel',  eegIDX, 'Flag',  3, 'Threshold', [-eegThresh eegThresh], 'Twindow', [rejectionStart rejectionEnd]);
    end
    if rejEegNoise %flags trials where EEG peak to peak activity within moving window is greater than eegNoiseThresh 
        EEG  = pop_artmwppth( EEG , 'Channel',  eegIDX, 'Flag',  4, 'Threshold', eegNoiseThresh, 'Twindow', [rejectionStart rejectionEnd], 'Windowsize', 200, 'Windowstep', 100); 
    end
    
    if rejFlatline %flags trials where any channel has flatlined completely (usually eyetracking)
        flatlineIDX = allChanNumbers(~ismember({EEG.chanlocs.labels}, {'StimTrak','HEOG','VEOG'}));
        EEG  = pop_artflatline(EEG , 'Channel', flatlineIDX, 'Duration',  200, 'Flag', 5, 'Threshold', [0 0], 'Twindow', [rejectionStart rejectionEnd]);
    end 
    
    EEG = pop_saveset(EEG, 'filename', fullfile(subdirPath, [vhdrFilename(1:end-5) '_unchecked.set']));
    
    tot = sprintf('Total Trials Rejected: %.0f', sum(EEG.reject.rejmanual));
    per = sprintf('Percent Trials Rejected: %.2f%%', round(sum((EEG.reject.rejmanual)/EEG.trials)*100,1));
    disp(tot)
    disp(per)
end

%% Clean up
eeglab redraw;


fclose(log);

%% Helper Functions

function equationList = get_chan_equations(EEG, rerefType, excludes)
    if ~any(strcmp({'mastoid', 'average'}, rerefType))
        error('rerefType must be "mastoid" or "average"')
    end

    baseEquation = 'ch%d = ch%d - (%s) Label %s';
    
    allLocs = {EEG.chanlocs.labels};
    
    includedChanLabels = allLocs;
    includedChanLabels(ismember(allLocs, excludes)) = [];
    [~, includedChanIndexes] = ismember(includedChanLabels, allLocs);
    
    equationList = {};
  
    if strcmp(rerefType, 'average')
        equationString = sprintf('avgchan(%s)', mat2colon(includedChanIndexes));
    else
        refIdx = find(strcmp({EEG.chanlocs.labels}, 'TP9'));
        equationString = sprintf('.5 * ch%d', refIdx);
    end
    
    for i=includedChanIndexes
        equationList{end + 1} = sprintf(baseEquation, i, i, equationString, allLocs{i}); %#ok<AGROW>
    end
    
end

function make_binlist(subdirPath, timelockCodes)
    % creates a simple binlist  (needed for epoching)

    binfid = fopen(fullfile(subdirPath, 'binlist.txt'), 'w');

    for i=1:numel(timelockCodes)
        fprintf(binfid, sprintf('bin %d\n', i));
        fprintf(binfid, sprintf('%d\n', timelockCodes(i)));
        fprintf(binfid, sprintf('.{%d}\n\n', timelockCodes(i)));
    end
end

function [xPix, yPix] = calcdeg2pix(eyeMoveThresh, distFromScreen, monitorWidth, monitorHeight, screenResX, screenResY)
    % takes a visual angle and returns the (rounded) horizontal and vertical number of
    % pixels from fixation that would be

    pixSize.X = monitorWidth/screenResX; %mm
    pixSize.Y = monitorHeight/screenResY; %mm

    mmfromfix = (2*distFromScreen) * tan(.5 * deg2rad(eyeMoveThresh));

    xPix = round(mmfromfix/pixSize.X);
    yPix = round(mmfromfix/pixSize.Y);
end