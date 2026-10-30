
clear; clc; close all;


fprintf('Loading MNIST Data...\n');
if exist('mnist.mat', 'file')
    d = load('mnist.mat');
else
    error('Error: mnist.mat not found. Please ensure it is in the current folder.');
end

% Extract and Preprocess Training Data
% Transpose to make them column vectors 
X_train = double(d.trainX') / 255.0;  % Normalise to [0, 1]
Y_train_labels = double(d.trainY);    

% Extract and Preprocess Test Data
X_test = double(d.testX') / 255.0;    % Normalise to [0, 1]
Y_test_labels = double(d.testY);      



[input_dim, num_train] = size(X_train);
num_test = size(X_test, 2);
num_classes = 10; % Digits 0-9

% Convert Labels to One-Hot Target Vectors (10 x N)

D_train = zeros(num_classes, num_train);
for i = 1:num_train
    
    D_train(Y_train_labels(i) + 1, i) = 1; 
end



W = rand(num_classes, input_dim) * 0.01 - 0.005; % Small random weights
b = rand(num_classes, 1) * 0.01 - 0.005;         % Small random bias

learning_rate = 0.1;
max_epochs = 20; % Maximum Epochs

% Storage for plotting
train_error_history = [];
test_accuracy_history = [];

fprintf('Starting Training on %d samples...\n', num_train);


for epoch = 1:max_epochs
    
    total_epoch_error = 0;
    
    % Shuffle training data each epoch
    perm = randperm(num_train);
    
    for k = perm
        x_k = X_train(:, k); % Current image (784x1)
        d_k = D_train(:, k); % Target vector (10x1)
        
      
        % Calculate weighted sum: v = Wx + b
        v = W * x_k + b;
        
        % Sigmoid Activation: y = 1 / (1 + e^-v)
        y_k = 1 ./ (1 + exp(-v));
        
        
        % Error signal: e = d - y
        e = d_k - y_k;
        
        % Calculate local gradient delta
      
        delta = e .* y_k .* (1 - y_k);
        
        % Update Weights and Biases
        % W_new = W + alpha * delta * x'
        W = W + learning_rate * (delta * x_k');
        b = b + learning_rate * delta;
        
        % Sum of Squared Errors
        total_epoch_error = total_epoch_error + sum(e.^2);
    end
    
    % Store Error
    train_error_history = [train_error_history, total_epoch_error];
    
    % Evaluate on Test Set every epoch
    correct_count = 0;
    for t = 1:num_test
        x_t = X_test(:, t);
        
        % Forward pass
        v_t = W * x_t + b;
        y_t = 1 ./ (1 + exp(-v_t));
        
        % Classification: Max output index
        [~, predicted_digit_idx] = max(y_t);
        predicted_digit = predicted_digit_idx - 1; 
        
        if predicted_digit == Y_test_labels(t)
            correct_count = correct_count + 1;
        end
    end
    test_acc = (correct_count / num_test) * 100;
    test_accuracy_history = [test_accuracy_history, test_acc];
    
    fprintf('Epoch %d | Error: %.2f | Test Accuracy: %.2f%%\n', ...
        epoch, total_epoch_error, test_acc);
end



% Figure 1: Learning Curve
figure;
yyaxis left
plot(1:max_epochs, train_error_history, '-r', 'LineWidth', 2);
ylabel('Total Training Error (SSE)');
xlabel('Epochs');

yyaxis right
plot(1:max_epochs, test_accuracy_history, '-b', 'LineWidth', 2);
ylabel('Test Accuracy (%)');
title('Task 2: SLP Learning Curve');
legend('Training Error', 'Test Accuracy');
grid on;

% Figure 2: Visualising Learned Weights

figure;
for i = 1:10
    subplot(2, 5, i);
    % Reshape the row of weights for digit (i-1)
    weight_img = reshape(W(i, :), 28, 28);
    imagesc(weight_img'); 
    colormap('jet');
    axis square off;
    title(['Digit ' num2str(i-1)]);
end
sgtitle('Learned Weights for All 10 Perceptrons');

% Figure 3: Confusion Matrix
% Generate predictions for the whole test set for the matrix
all_preds = zeros(1, num_test);
for t = 1:num_test
    y_t = 1 ./ (1 + exp(-(W * X_test(:, t) + b)));
    [~, idx] = max(y_t);
    all_preds(t) = idx - 1;
end

figure;
confusionchart(Y_test_labels, all_preds);
title('Confusion Matrix (Task 2)');