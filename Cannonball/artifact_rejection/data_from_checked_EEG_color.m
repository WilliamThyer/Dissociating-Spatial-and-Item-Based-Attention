% subs = {'01','08','09','11','13','15','20','21','22','23','24'};  % optionally {} for recursive search
subs = {'08','09','11','13','15','20','21','22','23','24'};  % subject 1 has strange behavior
subs={'26'};
%BUG: There was an issue storing removed channels in pop_select - subject
%15 with 0 trials rejected

experiment = 'C01c';
numsubs= length(subs);
root = '.';
destination = fullfile('..','analysis_DSCopy','data',experiment,filesep);
cd(root)
eeglab

for isub = 1:numsubs
    %if you want to use unchecked!
    checked_file = fullfile('..','raw_data','C01',subs{isub},'color',[experiment,'_',subs{isub},'_unchecked.set']);
    try
        EEG = pop_loadset(checked_file);
    catch err
        warning(['Error in loading data for subject ' subs{isub} ', skipping...'])
        continue
    end
    EEG = pop_rejepoch(EEG,EEG.reject.rejmanual,0);
    %Titles
    title = [experiment, '_', EEG.setname(end-1:end)];
    xdata_filename = [destination, title, '_xdata.mat'];
    ydata_filename = [destination, title, '_ydata.mat'];
    idx_filename = [destination, title, '_artifact_idx.mat'];
    behavior_filename = [destination, title, '_behavior.csv'];
    info_filename = [destination, title, '_info.mat'];
    
    % XData
    [xdata,chan_idx] = create_xdata(EEG);
    save(xdata_filename, 'xdata','-v7');
    
    % YData
    ydata = create_ydata(EEG, xdata);
    save(ydata_filename, 'ydata','-v7');
    
    % Info
    unique_ID_file = fullfile('..','raw_data','C01',subs{isub},'color',[experiment,'_0',subs{isub},'_info.json']);
    [unique_id,chan_labels,chan_x,chan_y,chan_z,sampling_rate,times] = create_info(EEG, unique_ID_file,chan_idx);
    save(info_filename,'unique_id','chan_labels','chan_x','chan_y','chan_z','sampling_rate','times','-v7');
    
    % Artifact Index (for behavior to match EEG)
    unchecked_file = fullfile('..','raw_data','C01',subs{isub},'color',[experiment,'_',subs{isub},'_unchecked.set']);
    artifact_idx = create_artifact_index(EEG, unchecked_file);
    save(idx_filename,'artifact_idx','-v7')
    
    % Save copy of behavior csv
    behavior_file = fullfile('..','raw_data','C01',subs{isub},'color',[experiment,'_',subs{isub},'.csv']);
    copyfile(behavior_file,behavior_filename);
    
    clear labels num_trials templabel x y checked_trials 
end
disp("DATA EXTRACTION COMPLETE")

function [xdata, chan_idx] = create_xdata(EEG)
    % create xdata for saving to .mat

    num_chans = EEG.nbchan;
    all_chans = strings(num_chans,1);
    for chan = 1:num_chans
        all_chans(chan,:) = EEG.chanlocs(chan).labels;
    end
    chan_idx = ismember(all_chans,{'L-GAZE-X','L-GAZE-Y','R-GAZE-X','R-GAZE-Y','StimTrak','HEOG','VEOG','TP9','GAZE_X','GAZE_Y'});

    xdata = EEG.data(~chan_idx,:,:);
end

function [ydata] = create_ydata(EEG, xdata)
    % create ydata for saving to .mat. This will definitely change based on
    % your portcode structure!

    num_trials = size(xdata,3);
    ydata = zeros(num_trials,1);
    for x=1:num_trials
        sorted_labels = sort(EEG.epoch(x).eventbinlabel);
        char_labels = char(sorted_labels(end));
        ydata(x,:) = str2double(char_labels(5));
    end
end

function [unique_id,chan_labels,chan_x,chan_y,chan_z,sampling_rate,times] = create_info(EEG, unique_ID_file, chan_idx)

    % Gather info variables
    chan_labels = {EEG.chanlocs.labels}';
    chan_labels = char(chan_labels(~chan_idx));
    chan_x = [EEG.chanlocs.X];
    chan_y = [EEG.chanlocs.Y];
    chan_z = [EEG.chanlocs.Z];
    chan_x = chan_x(~chan_idx(1:31));
    chan_y = chan_y(~chan_idx(1:31));
    chan_z = chan_z(~chan_idx(1:31));
    sampling_rate = EEG.srate;
    times = EEG.times;
    
    val = jsondecode(fileread(unique_ID_file));
    unique_id = str2double(val.UniqueSubjectIdentifier);
end

function [artifact_idx] = create_artifact_index(EEG, unchecked_file)

    num_rows = size(EEG.event,2);
    all_trials = zeros(num_rows,1);
    for x = 1:num_rows
        all_trials(:,x) = EEG.event(x).bepoch;
    end
    checked_trials = unique(all_trials);
    
    
    EEG = pop_loadset(unchecked_file);
    unchecked_trials = [1:EEG.trials]';
    artifact_idx = ismember(unchecked_trials,checked_trials);
end


