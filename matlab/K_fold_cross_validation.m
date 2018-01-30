close all
clear all

load mnist_train;
load mnist_valid;

learning_rate_list = (1:10)*0.1;
weight_regularization_power = -3:1;

inputs = [train_inputs;valid_inputs];
targets = [train_targets;valid_targets];

K = 10;
batch = size(inputs,1)/K;

hyperparameters.num_iterations = 100;

train_cost = [];
train_accuracy = [];
valid_cost = [];
valid_accuracy = [];
for each_weight_regularization = 10.^(weight_regularization_power)
    hyperparameters.weight_regularization = each_weight_regularization;
    cost_train_hold_lambda = [];
    accuracy_train_hold_lambda = [];
    cost_valid_hold_lambda = [];
    accuracy_valid_hold_lambda = [];
    for each_learning_rate = learning_rate_list
        hyperparameters.learning_rate = each_learning_rate;
        cost_train_list = zeros(1, hyperparameters.num_iterations);
        accuracy_train_list = zeros(1, hyperparameters.num_iterations);
        cost_valid_list = zeros(1, hyperparameters.num_iterations);
        accuracy_valid_list = zeros(1, hyperparameters.num_iterations);
        for k = 1:K
            valid_index = zeros(1, size(inputs,1));
            for j = ((k-1)*batch+1):k*batch
                valid_index(j) = 1;
            end
            train_index = ~valid_index;
            valid_index = ~train_index;
            
            train_inputs = inputs(train_index,:);
            train_targets = targets(train_index,:);
            valid_inputs = inputs(valid_index,:);
            valid_targets = targets(valid_index,:);
            
            result = Train(train_inputs, train_targets, valid_inputs, valid_targets, hyperparameters);
            cost_train_list = cost_train_list + result.cost_train_list;
            accuracy_train_list = accuracy_train_list + result.accuracy_train_list;
            cost_valid_list = cost_valid_list + result.cost_valid_list;
            accuracy_valid_list = accuracy_valid_list + result.accuracy_valid_list;
        end
        cost_train_list = cost_train_list./K;
        accuracy_train_list = accuracy_train_list./K;
        cost_valid_list = cost_valid_list./K;
        accuracy_valid_list = accuracy_valid_list./K;
        cost_train_hold_lambda = [cost_train_hold_lambda, max(cost_train_list)];
        accuracy_train_hold_lambda = [accuracy_train_hold_lambda, max(accuracy_train_list)];
        cost_valid_hold_lambda = [cost_valid_hold_lambda, max(cost_valid_list)];
        accuracy_valid_hold_lambda = [accuracy_valid_hold_lambda, max(accuracy_valid_list)];
    end
    train_cost = [train_cost;cost_train_hold_lambda];
    train_accuracy = [train_accuracy;accuracy_train_hold_lambda];
    valid_cost = [valid_cost;cost_valid_hold_lambda];
    valid_accuracy = [valid_accuracy;accuracy_valid_hold_lambda];
end

opti_lambda = 10^weight_regularization_power(mean(valid_accuracy, 2) == max(mean(valid_accuracy, 2)))
opti_eta = learning_rate_list(mean(valid_accuracy, 1) == max(mean(valid_accuracy, 1)))

subplot(221)
plot(weight_regularization_power,mean(train_cost, 2))
hold on
plot(weight_regularization_power,mean(valid_cost, 2),'x')
title('cost')
xlabel('log(lambda)')
ylabel('cost')
legend('train','validation')
grid on

subplot(222)
plot(weight_regularization_power,mean(train_accuracy, 2))
hold on
plot(weight_regularization_power,mean(valid_accuracy, 2),'x')
title('accuracy')
xlabel('log(lambda)')
ylabel('accuracy')
legend('train','validation')
grid on

subplot(223)
plot(learning_rate_list,mean(train_cost, 1))
hold on
plot(learning_rate_list,mean(valid_cost, 1),'x')
title('cost')
xlabel('learning rate')
ylabel('cost')
legend('train','validation')
grid on

subplot(224)
plot(learning_rate_list,mean(train_accuracy, 1))
hold on
plot(learning_rate_list,mean(valid_accuracy, 1),'x')
title('accuracy')
xlabel('learning rate')
ylabel('accuracy')
legend('train','validation')
grid on