
clear; clc; close all;



% Class '1' Patterns (Target = 1)
p1_1 = [0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0]; 
p1_2 = [0 1 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 1 1 1 0]; 
p1_3 = [0 1 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0]; 
p1_4 = [0 0 0 1 0; 0 0 1 1 0; 0 1 1 0 0; 1 1 0 0 0; 1 0 0 0 0]; 
p1_5 = [0 0 0 0 1; 0 0 0 1 1; 0 0 1 1 0; 0 1 1 0 0; 1 1 0 0 0]; 
p1_6 = [0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 1 1 0]; 

% Class '0' Patterns (Target = -1)
p0_1 = [1 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 1 1 1 1 1]; 
p0_2 = [0 1 1 1 0; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 0]; 
p0_3 = [0 1 1 0 0; 1 0 0 1 0; 1 0 0 1 0; 1 0 0 1 0; 0 1 1 0 0]; 
p0_4 = [0 0 1 1 0; 0 1 0 0 1; 0 1 0 0 1; 0 1 0 0 1; 0 0 1 1 0];
p0_5 = [1 1 1 1 0; 1 0 0 1 0; 1 0 0 1 0; 1 0 0 1 0; 1 1 1 1 0]; 
p0_6 = [0 1 1 1 1; 1 0 0 0 1; 1 0 0 0 1; 1 0 0 0 1; 0 1 1 1 1]; 

% Flatten to column vectors (25x1) and store in cell array

data_1 = {p1_1(:), p1_2(:), p1_3(:), p1_4(:), p1_5(:), p1_6(:)};
data_0 = {p0_1(:), p0_2(:), p0_3(:), p0_4(:), p0_5(:), p0_6(:)};


n_trials = 10;          % Repeat 10 times to get average performance 
max_epochs = 50;        % Max training loops
learning_rate = 0.1;    % Alpha 
input_dim = 25;         % Dimension of input vector

% Storage for results
trial_accuracies = zeros(n_trials, 1);
trial_epochs = zeros(n_trials, 1);
errors_last_trial = []; % To store error curve of the last trial for plotting

fprintf('Starting %d Simulation Trials...\n\n', n_trials);


for trial = 1:n_trials
    
    % Randomly Split Data (4 Train / 2 Test per class) 
    perm_1 = randperm(6);
    perm_0 = randperm(6);
    
    % Training Set 
    train_idx_1 = perm_1(1:4);
    train_idx_0 = perm_0(1:4);
    
    % Test Set 
    test_idx_1 = perm_1(5:6);
    test_idx_0 = perm_0(5:6);
    
    % Construct Training Matrix X and Label Vector D
    
    X_train = [data_1{train_idx_1}, data_0{train_idx_0}]; % 25x8 matrix
    D_train = [ones(1, 4), -ones(1, 4)];                  % 1x8 vector
    
    % Construct Test Matrix
    X_test = [data_1{test_idx_1}, data_0{test_idx_0}];    % 25x4 matrix
    D_test = [ones(1, 2), -ones(1, 2)];                   % 1x4 vector
    
    % Initialize Weights and Bias ---
    w = rand(input_dim, 1) - 0.5; % Small random weights
    b = rand() - 0.5;             % Small random bias
    
    % Training Loop (Perceptron Learning Rule) 
    epoch_errors = [];
    
    for epoch = 1:max_epochs
        total_error = 0;
        
        % Randomise presentation order within epoch 
        perm_train = randperm(8);
        
        for k = perm_train
            x_k = X_train(:, k);
            d_k = D_train(k);
            
            % Calculate Output: y = sign(w'*x + b) 
            v = w' * x_k + b;
            y_k = sign(v);
            if y_k == 0, y_k = 1; end % Handle sign(0) case if needed
            
            % Update Weights if Error Exists
            if y_k ~= d_k
                % Update Rules 
                w = w + learning_rate * (d_k - y_k) * x_k;
                b = b + learning_rate * (d_k - y_k);
                total_error = total_error + 1;
            end
        end
        
        epoch_errors = [epoch_errors, total_error];
        
        % Stop if converged 
        if total_error == 0
            break; 
        end
    end
    
    trial_epochs(trial) = length(epoch_errors);
    
    % Testing
    % Apply trained network to test patterns
    n_correct = 0;
    n_test_samples = size(X_test, 2);
    
    for k = 1:n_test_samples
        y_test = sign(w' * X_test(:, k) + b);
        if y_test == 0, y_test = 1; end
        
        if y_test == D_test(k)
            n_correct = n_correct + 1;
        end
    end
    
    % Calculate Percentage Error Rate 
    accuracy = (n_correct / n_test_samples) * 100;
    trial_accuracies(trial) = accuracy;
    
   
    if trial == n_trials
        errors_last_trial = epoch_errors;
    end
    
    fprintf('Trial %d: Converged in %d epochs | Test Accuracy: %.1f%%\n', ...
        trial, trial_epochs(trial), accuracy);
end



% Calculate Averages
avg_accuracy = mean(trial_accuracies);
avg_epochs = mean(trial_epochs);

fprintf('\n------------------------------------------------\n');
fprintf('RESULTS SUMMARY (Over %d Trials)\n', n_trials);
fprintf('------------------------------------------------\n');
fprintf('Average Test Accuracy: %.2f%%\n', avg_accuracy);
fprintf('Average Epochs to Converge: %.2f\n', avg_epochs);
fprintf('------------------------------------------------\n');

% Plot Learning Curve 
figure;
plot(1:length(errors_last_trial), errors_last_trial, '-ro', 'LineWidth', 2, 'MarkerFaceColor', 'r');
title('Learning Curve (Training Error vs. Epochs)');
xlabel('Epoch number');
ylabel('Total Classification Errors');
grid on;
axis([1 max(trial_epochs) -0.5 5]); 

% Visualize the learned weights 
% Reshape 25x1 weights back to 5x5 to see which pixels matter most
figure;
imagesc(reshape(w, 5, 5));
colormap('jet');
colorbar;
title('Visualisation of Learned Weights (5x5)');
axis square off;