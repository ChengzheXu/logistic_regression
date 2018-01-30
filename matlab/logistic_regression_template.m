%% Clear workspace.
clear all;
close all;

%% Load data.
load mnist_train_small;
load mnist_valid;

%% TODO: Initialize hyperparameters.
% Learning rate
hyperparameters.learning_rate = 0.3;
% Weight regularization parameter
% hyperparameters.weight_regularization = 0.5;
% Number of iterations
hyperparameters.num_iterations = 40;
% Logistics regression weights
weights = normrnd(0, 1/(1+size(train_inputs_small,2)), [1+size(train_inputs_small,2), 1]);
%weights = normrnd(0, 1, [1+size(train_inputs_small,2), 1]);
%weights = unifrnd(-1, 1, [1+size(train_inputs_small,2), 1]);


%% Verify that your logistic function produces the right gradient, diff should be very close to 0
% this creates small random data with 20 examples and 10 dimensions and checks the gradient on
% that data.
nexamples = 20;
ndimensions = 10;
diff = checkgrad('logistic', ...
	             randn((ndimensions + 1), 1), ...   % weights
                 0.001,...                          % perturbation
                 randn(nexamples, ndimensions), ... % data        
                 rand(nexamples, 1), ...            % targets
                 hyperparameters)                   % other hyperparameters

N = size(train_inputs_small,1);
%% Begin learning with gradient descent.

cost_train_list = [];
accuracy_train_list = [];
cost_valid_list = [];
accuracy_valid_list = [];

for t = 1:hyperparameters.num_iterations
	% Find the negative log likelihood and derivative w.r.t. weights.
	[f, df, predictions] = logistic(weights, ...
                                    train_inputs_small, ...
                                    train_targets_small, ...
                                    hyperparameters);

    [cross_entropy_train, frac_correct_train] = evaluate(train_targets_small, predictions);
    
    cost_train_list = [cost_train_list, cross_entropy_train];
    accuracy_train_list = [accuracy_train_list, frac_correct_train];

	% Find the fraction of correctly classified validation examples.
% 	[temp, temp2, frac_correct_valid] = logistic(weights, ...
%                                                  valid_inputs, ...
%                                                  valid_targets, ...
%                                                  hyperparameters);

    if isnan(f) || isinf(f)
		error('nan/inf error');
    end

    %% Update parameters.
    weights = weights - hyperparameters.learning_rate .* df / N;

    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    
    cost_valid_list = [cost_valid_list, cross_entropy_valid];
    accuracy_valid_list = [accuracy_valid_list, frac_correct_valid];
        
	%% Print some stats.
	fprintf(1, 'ITERATION:%4i   NLOGL:%4.2f TRAIN CE %.6f TRAIN FRAC:%2.2f VALIC_CE %.6f VALID FRAC:%2.2f\n',...
			t, f/N, cross_entropy_train, frac_correct_train*100, cross_entropy_valid, frac_correct_valid*100);

end

subplot(211)
title('cross entropy')
plot(1:hyperparameters.num_iterations, cost_train_list)
hold on
plot(1:hyperparameters.num_iterations, cost_valid_list)
xlabel('epoch')
ylabel('cost')
grid on
legend('training','validation')
subplot(212)
title('accuracy')
plot(1:hyperparameters.num_iterations, accuracy_train_list)
hold on
plot(1:hyperparameters.num_iterations, accuracy_valid_list)
xlabel('epoch')
ylabel('accuracy')
grid on
legend('training','validation')