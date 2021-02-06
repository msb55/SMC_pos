function [ensemble_diversity] = q_statistic_ensemble(Aq)
    sum_diversity = 0;
    matrix_width = size(Aq, 1);
    for x = 1:matrix_width
        for y = x:matrix_width
            if x == y
                continue;
            end
            sum_diversity = sum_diversity + Aq(x, y);
        end
    end

    ensemble_diversity = (2/(matrix_width*(matrix_width-1))) * sum_diversity;
    
    if isnan(ensemble_diversity)
        ensemble_diversity = 0;
    end
end

