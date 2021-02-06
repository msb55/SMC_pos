%%%%%%%%%%%%%%%%%%%%%
% Function: build_folds 
% 
% Objective: Build the six folds
%
% Input:
%
%   dataset - The dataset name    
%   ensemble_size - The size of the ensemble
%
% Output:
%
%   No return, but the function saves the sets (TRAIN, VALIDATION1,
%   VALIDATION2 and TEST) to disc via "create_fold" function
%
%%%%%%%%%%%%%%%%%%%%%
function [] = build_folds(dataset, ensemble_size)
  [DATA, LABELS] = load_data(sprintf('data/%s.data', dataset));

  folds = 10;
  folds_for_data = crossvalind('Kfold', LABELS, 10, 'Classes', unique(LABELS));
  folds_indexes = 1:folds;

  PERMS = [4  9  2  8  7  6  1  3  10  5; 7  5  2  3  6  8  1  9  10  4;...
      3  7  8  10  5  9  6  1  4  2; 1  7  2  5  10  6  8  3  4  9;...
      6  10  8  3  9  5  4  7  1  2; 4  2  3  8  10  5  6  9  1  7;...
      7  4  5  1  6  9  2  3  8  10; 4  6  5  8  1  10  2  9  7  3;...
      7  1  2  4  6  3  9  8  5  10; 6  3  5  7  2  9  4  8  1  10];

  fprintf('Generating folds:\n');

  for i=1:folds
    create_fold(dataset, i, PERMS, folds_for_data, folds_indexes, DATA, LABELS, ensemble_size);
  end
    
  fprintf('\n');
  clear DATA LABELS folds_indexes;
end

function [] = create_fold(dataset, index, PERMS, folds_for_data, folds_indexes, DATA, LABELS, ensemble_size)
  prwarning(0);
  prwaitbar('off');
  warning('off','all');

  fprintf(' #%d', index);
  permutation = PERMS(index, :);

  training_indexes = permutation(1:7);
  validation_1_index = permutation(8);
  validation_2_index = permutation(9);
  test_index = permutation(10);

  mkdir(sprintf('data/%s/%d', dataset, ensemble_size));
  mkdir(sprintf('data/%s/%d/fold_%d', dataset, ensemble_size, index));
  TEST = DATA(folds_for_data==test_index, :);
  TEST_LABELS = LABELS(folds_for_data==test_index);
  save(sprintf('data/%s/%d/fold_%d/test.mat', dataset, ensemble_size, index), 'TEST', 'TEST_LABELS');

  VALIDATION_1 = DATA(folds_for_data==validation_1_index, :);
  VALIDATION_1_LABELS = LABELS(folds_for_data==validation_1_index);
  [VALIDATION_1_HARD, VALIDATION_1_HARD_LABELS, VALIDATION_1_EASY, VALIDATION_1_EASY_LABELS] = kdn(VALIDATION_1, VALIDATION_1_LABELS, 5);
  save(sprintf('data/%s/%d/fold_%d/validation_1.mat', dataset, ensemble_size, index), 'VALIDATION_1', 'VALIDATION_1_LABELS');
  save(sprintf('data/%s/%d/fold_%d/validation_1_hard.mat', dataset, ensemble_size, index), 'VALIDATION_1_HARD', 'VALIDATION_1_HARD_LABELS');
  save(sprintf('data/%s/%d/fold_%d/validation_1_easy.mat', dataset, ensemble_size, index), 'VALIDATION_1_EASY', 'VALIDATION_1_EASY_LABELS');

  VALIDATION_2 = DATA(folds_for_data==validation_2_index, :);
  VALIDATION_2_LABELS = LABELS(folds_for_data==validation_2_index);
  [VALIDATION_2_HARD, VALIDATION_2_HARD_LABELS, VALIDATION_2_EASY, VALIDATION_2_EASY_LABELS] = kdn(VALIDATION_2, VALIDATION_2_LABELS, 5);
  save(sprintf('data/%s/%d/fold_%d/validation_2.mat', dataset, ensemble_size, index), 'VALIDATION_2', 'VALIDATION_2_LABELS');
  save(sprintf('data/%s/%d/fold_%d/validation_2_hard.mat', dataset, ensemble_size, index), 'VALIDATION_2_HARD', 'VALIDATION_2_HARD_LABELS');
  save(sprintf('data/%s/%d/fold_%d/validation_2_easy.mat', dataset, ensemble_size, index), 'VALIDATION_2_EASY', 'VALIDATION_2_EASY_LABELS');

  training_indexes = folds_indexes(folds_indexes ~= test_index & folds_indexes ~= validation_1_index & folds_indexes ~= validation_2_index);

  TRAIN = [DATA(folds_for_data==training_indexes(1), :); DATA(folds_for_data==training_indexes(2), :); ...
    DATA(folds_for_data==training_indexes(3), :); DATA(folds_for_data==training_indexes(4), :); ...
    DATA(folds_for_data==training_indexes(5), :); DATA(folds_for_data==training_indexes(6), :); ...
    DATA(folds_for_data==training_indexes(7), :)];

  TRAIN_LABELS = [LABELS(folds_for_data==training_indexes(1)); LABELS(folds_for_data==training_indexes(2)); ...
    LABELS(folds_for_data==training_indexes(3)); LABELS(folds_for_data==training_indexes(4)); ...
    LABELS(folds_for_data==training_indexes(5)); LABELS(folds_for_data==training_indexes(6)); ...
    LABELS(folds_for_data==training_indexes(7))];

  save(sprintf('data/%s/%d/fold_%d/train.mat', dataset, ensemble_size, index), 'TRAIN', 'TRAIN_LABELS');

  % train_ensemble
  ensemble = train_ensemble_folds(ensemble_size, TRAIN, TRAIN_LABELS);
  save(sprintf('data/%s/%d/fold_%d/ensemble.mat', dataset, ensemble_size, index), 'ensemble');

  clear TEST VALIDATION_1 VALIDATION_2 TEST_LABELS VALIDATION_1_LABELS VALIDATION_2_LABELS TRAIN TRAIN_LABELS;
  clear training_indexes;
end
