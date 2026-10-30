
clear; clc; close all;
tic

fprintf('Loading MNIST Data...\n');
if exist('mnist.mat', 'file')
    d = load('mnist.mat');
else
    error('Error: mnist.mat not found.');
end

% Normalise Data
X_train = double(d.trainX') / 255.0;
Y_train_labels = double(d.trainY);
X_test = double(d.testX') / 255.0;
Y_test_labels = double(d.testY);

[input_dim, num_train] = size(X_train);
num_test = size(X_test, 2);
num_classes = 10;

% One-Hot Encoding for Targets
D_train = zeros(num_classes, num_train);
for i = 1:num_train
    D_train(Y_train_labels(i) + 1, i) = 1;
end


num_hidden = 20;  % Number of hidden neurons 
learning_rate = 0.1;
max_epochs = 20;  

% Initialise Weights 
% Input -> Hidden Weights (W_h) and Biases (b_h)
W_h = rand(num_hidden, input_dim) * 0.02 - 0.01;
b_h = rand(num_hidden, 1) * 0.02 - 0.01;

% Hidden -> Output Weights (W_o) and Biases (b_o)
W_o = rand(num_classes, num_hidden) * 0.02 - 0.01;
b_o = rand(num_classes, 1) * 0.02 - 0.01;

% Storage for plotting
train_error_history = [];
test_accuracy_history = [];

fprintf('Starting MLP Training (Hidden Nodes: %d)...\n', num_hidden);


for epoch = 1:max_epochs
    total_epoch_error = 0;
    perm = randperm(num_train);
    
    for k = perm
        x_k = X_train(:, k);
        d_k = D_train(:, k);
        
  
        % 1. Input to Hidden
        v_h = W_h * x_k + b_h;
        y_h = 1 ./ (1 + exp(-v_h)); % Sigmoid activation
        
        % 2. Hidden to Output
        v_o = W_o * y_h + b_o;
        y_o = 1 ./ (1 + exp(-v_o)); % Sigmoid activation
        
       
        
        % Calculate Output Error and Delta
        e_o = d_k - y_o;
        % delta_o = error * derivative of sigmoid
        delta_o = e_o .* y_o .* (1 - y_o);
        
        % Calculate Hidden Layer Delta 
        
        e_h = W_o' * delta_o;
        delta_h = e_h .* y_h .* (1 - y_h);
        
        % Update Weights and Biases
        % Update Hidden->Output
        W_o = W_o + learning_rate * (delta_o * y_h');
        b_o = b_o + learning_rate * delta_o;
        
        % Update Input->Hidden
        W_h = W_h + learning_rate * (delta_h * x_k');
        b_h = b_h + learning_rate * delta_h;
        
        total_epoch_error = total_epoch_error + sum(e_o.^2);
    end
    
    
    correct_count = 0;
    for t = 1:num_test
        x_t = X_test(:, t);
        % Forward pass only
        y_h_test = 1 ./ (1 + exp(-(W_h * x_t + b_h)));
        y_o_test = 1 ./ (1 + exp(-(W_o * y_h_test + b_o)));
        
        [~, pred] = max(y_o_test);
        if (pred - 1) == Y_test_labels(t)
            correct_count = correct_count + 1;
        end
    end
    test_acc = (correct_count / num_test) * 100;
    
    train_error_history = [train_error_history, total_epoch_error];
    test_accuracy_history = [test_accuracy_history, test_acc];
    
    fprintf('Epoch %d | Error: %.2f | Test Acc: %.2f%%\n', epoch, total_epoch_error, test_acc);
end



% Plot Learning Curve
figure;
yyaxis left
plot(1:max_epochs, train_error_history, '-r', 'LineWidth', 2);
ylabel('Training Error (SSE)');
xlabel('Epochs');
yyaxis right
plot(1:max_epochs, test_accuracy_history, '-b', 'LineWidth', 2);
ylabel('Test Accuracy (%)');
title(['Task 3: MLP Learning Curve (' num2str(num_hidden) ' Hidden Nodes)']);
grid on;

% Confusion Matrix for MLP
all_preds = zeros(1, num_test);
for t = 1:num_test
    x_t = X_test(:, t);
    y_h_test = 1 ./ (1 + exp(-(W_h * x_t + b_h)));
    y_o_test = 1 ./ (1 + exp(-(W_o * y_h_test + b_o)));
    [~, idx] = max(y_o_test);
    all_preds(t) = idx - 1;
end

figure;
confusionchart(Y_test_labels, all_preds);
title(['Confusion Matrix (MLP - ' num2str(num_hidden) ' Hidden Nodes)']);

toc