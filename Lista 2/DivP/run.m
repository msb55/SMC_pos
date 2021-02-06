%%%%%%%%%%%%%%%%%%%%%
% Function: run
%
% Objective: The main function of the system
% 
% Input:
%
%   dataset - The dataset name
%   rebuild - 'true' - if you want to generate new sets to
%                           validation, training and test. 
%             'false' - otherwise
%
% Examples:
%
% >> run('wine',true);
% >> run('wine'); or run('wine',false);
%%%%%%%%%%%%%%%%%%%%%
function [] = run(dataset, rebuild)
    if nargin == 1
        rebuild = false;
    end

    prwarning(0);
    prwaitbar('off');
    warning('off','all');
    graph_destroy;

    folds = 10;

    L = 100;
    if rebuild
        build_folds(dataset, L);
    end

%   architecture_errors = zeros(folds, 1);
%   full_ensemble_errors = zeros(folds, 1);
%   single_best_errors = zeros(folds, 1);
%   oracle_errors = zeros(folds, 1);
% 
%   architecture_sizes = zeros(folds, 1);
%   architecture_weights = zeros(folds, 6);
%   architecture_final_ensembles = cell(folds, 1);

%   RESULTS_CSV = ['FOLD', 'q_not_pruned', 'q_standard', 'q_easy', 'q_hard', 'acc_not_pruned', 'acc_standard', 'acc_easy', 'acc_hard', ...
%                 'auc_not_pruned', 'auc_standard', 'auc_easy', 'auc_hard', 'g_mean_not_pruned', 'g_mean_standard', 'g_mean_easy', 'g_mean_hard', ...
%                 'f1_not_pruned', 'f1_standard', 'f1_easy', 'f1_hard', 'size_not_pruned', 'size_standard', 'size_easy', 'size_hard' ];
    RESULTS_CSV = [];
%     i = 2;
    for i=1:folds   
        load(sprintf('data/%s/%d/fold_%d/validation_1.mat', dataset, L, i));
        load(sprintf('data/%s/%d/fold_%d/validation_2.mat', dataset, L, i));

        load(sprintf('data/%s/%d/fold_%d/validation_1_easy.mat', dataset, L, i));
        load(sprintf('data/%s/%d/fold_%d/validation_2_easy.mat', dataset, L, i));

        load(sprintf('data/%s/%d/fold_%d/validation_1_hard.mat', dataset, L, i));
        load(sprintf('data/%s/%d/fold_%d/validation_2_hard.mat', dataset, L, i));

        fprintf('Running for NORMAL validation set \n');
        [~, ~, ~, ~, ~, ~, ~, ensemble_standard] = perform_fold(L, dataset, i, VALIDATION_1, ...
          VALIDATION_1_LABELS, VALIDATION_2, VALIDATION_2_LABELS);

        fprintf('Running for EASY validation set \n');
        [~, ~, ~, ~, ~, ~, ~, ensemble_easy] = perform_fold(L, dataset, i, VALIDATION_1_EASY, ...
          VALIDATION_1_EASY_LABELS, VALIDATION_2_EASY, VALIDATION_2_EASY_LABELS);

        fprintf('Running for HARD validation set \n');
        [~, ~, ~, ~, ~, ~, ~, ensemble_hard] = perform_fold(L, dataset, i, VALIDATION_1_HARD, ...
          VALIDATION_1_HARD_LABELS, VALIDATION_2_HARD, VALIDATION_2_HARD_LABELS);

        load(sprintf('data/%s/%d/fold_%d/test.mat', dataset, L, i));
        load(sprintf('data/%s/%d/fold_%d/ensemble.mat',     dataset, L, i));

        fprintf('Evaluating not pruned ensemble \n');
        T_DP_NOT_PRUNED = build_decision_profile(ensemble, TEST, TEST_LABELS);
        write_results(T_DP_NOT_PRUNED, 'decision_profile_not_pruned.csv', dataset, L, i)
        Aq = diversity_graph(ensemble, 'q_statistic', T_DP_NOT_PRUNED, TEST_LABELS);
        ensemble_diversity_not_pruned = q_statistic_ensemble(Aq);
        disp('Ensemble diversity (Not Pruned):')
        disp(ensemble_diversity_not_pruned)

        [acc_not_pruned, f_measure_not_pruned, gmean_not_pruned, auc_not_pruned] = ensemble_metrics(ensemble, TEST, TEST_LABELS);

        fprintf('Evaluating standard pruned ensemble \n');
        T_DP_STANDARD = build_decision_profile(ensemble_standard, TEST, TEST_LABELS);
        write_results(T_DP_STANDARD, 'decision_profile_standard.csv', dataset, L, i)
        Aq = diversity_graph(ensemble_standard, 'q_statistic', T_DP_STANDARD, TEST_LABELS);
        ensemble_diversity = q_statistic_ensemble(Aq);
        disp('Ensemble diversity (Normal):')
        disp(ensemble_diversity)

        [acc_standard, f_measure_standard, gmean_standard, auc_standard] = ensemble_metrics(ensemble_standard, TEST, TEST_LABELS);

        fprintf('Evaluating easy pruned ensemble \n');
        T_DP_EASY = build_decision_profile(ensemble_easy, TEST, TEST_LABELS);
        write_results(T_DP_EASY, 'decision_profile_easy.csv', dataset, L, i)
        Aq_EASY = diversity_graph(ensemble_easy, 'q_statistic', T_DP_EASY, TEST_LABELS);
        ensemble_diversity_easy = q_statistic_ensemble(Aq_EASY);
        disp('Ensemble diversity (Easy):')
        disp(ensemble_diversity_easy)

        [acc_easy, f_measure_easy, gmean_easy, auc_easy] = ensemble_metrics(ensemble_easy, TEST, TEST_LABELS);

        fprintf('Evaluating hard pruned ensemble \n');
        T_DP_HARD = build_decision_profile(ensemble_hard, TEST, TEST_LABELS);
        write_results(T_DP_HARD, 'decision_profile_hard.csv', dataset, L, i)
        Aq_HARD = diversity_graph(ensemble_hard, 'q_statistic', T_DP_HARD, TEST_LABELS);
        ensemble_diversity_hard = q_statistic_ensemble(Aq_HARD);
        disp('Ensemble diversity (Hard):')
        disp(ensemble_diversity_hard)

        [acc_hard, f_measure_hard, gmean_hard, auc_hard] = ensemble_metrics(ensemble_hard, TEST, TEST_LABELS);

        fprintf('Saving partial (incremental) results \n');
        RESULTS_CSV = [RESULTS_CSV; [i, ensemble_diversity_not_pruned, ensemble_diversity, ensemble_diversity_easy, ensemble_diversity_hard, ...
            acc_not_pruned, acc_standard, acc_easy, acc_hard, auc_not_pruned, auc_standard, auc_easy, auc_hard, gmean_not_pruned, gmean_standard, ...
            gmean_easy, gmean_hard, f_measure_not_pruned, f_measure_standard, f_measure_easy, f_measure_hard, length(ensemble), ...
            length(ensemble_standard), length(ensemble_easy), length(ensemble_hard)]];

        write_results(RESULTS_CSV, 'partial_results.csv', dataset, L, i)
    end
    fprintf('Saving final results\n');
    writematrix(RESULTS_CSV, sprintf('results/%s/final_results.csv', dataset));  
end

function [] = write_results(to_save, filename, dataset, ensemble_size, index)
    mkdir(sprintf('results/%s/%d/fold_%d', dataset, ensemble_size, index));
    writematrix(to_save, sprintf('results/%s/%d/fold_%d/%s', dataset, ensemble_size, index, filename))
end

function [architecture_error, architecture_weights, architecture_size, architecture_final_ensemble, ...
  full_ensemble_error, single_best_error, oracle_error, ensemble_select] = perform_fold(L, dataset, i, validation_1_set, ...
  validation_1_labels_set, validation_2_set, validation_2_labels_set)
  fprintf('Run %d #%d\n', L, i);
      
  load(sprintf('data/%s/%d/fold_%d/ensemble.mat',     dataset, L, i));
  load(sprintf('data/%s/%d/fold_%d/test.mat',         dataset, L, i));
  
  DP = build_decision_profile(ensemble, validation_1_set, validation_1_labels_set);
  
  % Using DP to calculate the diversity between classifiers
  [Adis, Aq, Ap, Ak, Adf] = build_measures_matrixes(ensemble, DP, validation_1_labels_set);

  %%%%%%%%%%%%%%% Aq NORMALIZATION TO INTERVAL [0;1] %%%%%%%%%%%%%%
  vec = unique(Aq);
  maxVec = vec(end-1);
  minVec = vec(1);
  Aq = ((Aq-minVec)./(maxVec-minVec));
  Aq(Aq >= 16) = 32;
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  Adis = 1 - Adis;
  Adis(Adis <= -31) = 32;
  
  FitnessHolder = @(X)ga_fitness_function(X, ensemble, Adis, Aq, Ap, Ak, Adf, validation_2_set, validation_2_labels_set);

  ip = [1,1,1,1,1,-5];
  ip = [ip;[-1,-1,-1,-1,-1,5]];
  
  graph_init(1500);
  BEST = ga(FitnessHolder, 6, [],[],[],[],[],[],[], ...
    gaoptimset('Display', 'iter', 'Vectorized', 'on', ...
      'PopulationSize', 22, 'PopInitRange', [-1;1], 'FitnessLimit', 0, 'StallGenLimit', 15, ...
      'InitialPopulation',ip,'EliteCount',3));
  graph_destroy;

  job_full = batch(@classify_dataset, 1, {ensemble, TEST, TEST_LABELS}, 'AdditionalPaths', 'prtools');
  job_oracle = batch(@oracle_classify, 1, {ensemble, TEST, TEST_LABELS}, 'AdditionalPaths', 'prtools');
  job_sbe = batch(@single_best_classifier, 1, {ensemble, TEST, TEST_LABELS}, 'AdditionalPaths', 'prtools');
  [best_ensemble, ~] = find_best(ensemble, Adis, Aq, Ap, Ak, Adf, validation_2_set, validation_2_labels_set, BEST(1), BEST(2), BEST(3), BEST(4), BEST(5), BEST(6));

  wait(job_oracle);
  wait(job_sbe);
  wait(job_full);

  architecture_error = test_best(dataset, L, i, ensemble(best_ensemble));
  architecture_weights = BEST;
  architecture_size = length(best_ensemble);
  %architecture_final_ensemble = sprintf('%d-', best_ensemble);
  architecture_final_ensemble = best_ensemble;
  ensemble_select = ensemble(best_ensemble);
  
  full_output = fetchOutputs(job_full);
  full_ensemble_error = full_output{1}; % Erro do ensemble por fold

  sbe_output = fetchOutputs(job_sbe);
  single_best_error = sbe_output{1};

  oracle_output = fetchOutputs(job_oracle);
  oracle_error = oracle_output{1};

  delete(job_oracle);
  delete(job_sbe);
  delete(job_full);
  
end

function [new_ensemble, error_rate] = find_best(ensemble, Adis, Aq, Ap, Ak, Adf, DATA, LABELS, w_dis, w_q, w_p, w_k, w_df, t)
  graph_init;

  Af = Adis * w_dis + Aq * w_q + Ap * w_p + Ak * w_k + Adf * w_df;

  % To eliminate the main diagonal
  MASK = xor(eye(length(ensemble)), ones(length(ensemble)));
  ADJACENCY_MATRIX = Af < t & MASK;

  [new_ensemble, error_rate] = build_color_ensemble(ensemble, ADJACENCY_MATRIX, DATA, LABELS);

  graph_destroy;
end

function [error_rate] = test_best(dataset, L, i, best_ensemble)
  load(sprintf('data/%s/%d/fold_%d/test.mat', dataset, L, i));

  error_rate = classify_dataset(best_ensemble, TEST, TEST_LABELS);

  graph_destroy;
end
