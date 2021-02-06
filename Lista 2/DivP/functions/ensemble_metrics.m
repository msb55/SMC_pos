function [acc, f_measure, gmean, auc] = ensemble_metrics(ensemble, DATA, LABELS)
    scores = ensemble_predict(ensemble, DATA, LABELS);
    
    [~,~,~,auc] = perfcurve(LABELS, scores, 1, 'XCrit','ppv');
    metrics = Evaluate(LABELS, scores);
    
    acc = metrics(1);
    f_measure = metrics(6);
    
    if isnan(f_measure)
        f_measure = 0;
    end
    
    gmean = metrics(7);
end

