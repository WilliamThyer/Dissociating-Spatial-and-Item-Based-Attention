function [t] = summary(EEG,txt)

    fprintf(txt, ['\nRunning ', EEG.setname, '\n\n']);
    
    ydata = zeros(EEG.trials,1);
    for x=1:EEG.trials
        sorted_labels = sort(EEG.epoch(x).eventbinlabel);
        char_labels = char(sorted_labels(end));
        ydata(x,:) = str2double(char_labels(6));
    end
    
    t = zeros(5,1);
    for x=1:5
        t(x) = sum(ydata(~EEG.reject.rejmanual)==x);
        fprintf(txt,'Condition %1.f:%1.f\n',x,t(x));
    end