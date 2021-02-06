function [hard_instances, hard_instances_labels, easy_instances, easy_instances_labels] = kdn(DATA, LABELS, K)
    hard_instances = [];
    hard_instances_labels = [];
    easy_instances = [];
    easy_instances_labels = [];
    
    idx = knn(DATA, DATA, K);

    for i = 1:length(DATA)
        kdn_value = 0;
        for j = 1:length(idx(i, :)) % i neighbors
            if LABELS(idx(i, j)) ~= LABELS(i)
                kdn_value = kdn_value + 1;
            end
        end
        kdn_value = kdn_value / K;

        if kdn_value > 0.5
            hard_instances = [hard_instances; DATA(i, :)];
            hard_instances_labels = [hard_instances_labels; LABELS(i)];
        else
            easy_instances = [easy_instances; DATA(i, :)];
            easy_instances_labels = [easy_instances_labels; LABELS(i)];
        end
    end
    
    disp('Original instances size')
    disp(size(DATA))
    disp('Easy instances size')
    disp(size(easy_instances))
    disp('Hard instances size')
    disp(size(hard_instances))
end

function [idx] = knn(DATA, TARGET_INSTANCE, K)
    idx = knnsearch(DATA, TARGET_INSTANCE, "K", K+1);
end