function [t] = summary(EEG,txt)

    ydata = zeros(EEG.trials,1);
    for x=1:EEG.trials
        sorted_labels = sort(EEG.epoch(x).eventbinlabel);
        char_labels = char(sorted_labels(end));
        ydata(x,:) = str2double(char_labels(6));
    end
    
    t = zeros(5,1);
    for x=1:5
        t(x) = sum(ydata(~EEG.reject.rejmanual)==x);
        fprintf(txt,'Setsize %1.f:%1.f\n',x,t(x));
    end
    per = sprintf('\nPercent Trials Rejected: %.2f%%\n\n\n', round(sum((EEG.reject.rejmanual)/EEG.trials)*100,1));
    fprintf(txt,per);