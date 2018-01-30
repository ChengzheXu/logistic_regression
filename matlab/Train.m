function result = Train(train_inputs, train_targets, valid_inputs, valid_targets, hyperparameters)

result.cost_train_list = [];
result.accuracy_train_list = [];
result.cost_valid_list = [];
result.accuracy_valid_list = [];

weights = normrnd(0, 1/(1+size(train_inputs,2)), [1+size(train_inputs,2), 1]);
for t=1:hyperparameters.num_iterations
    [~, df, predictions] = logistic_pen(weights, ...
                                    train_inputs, ...
                                    train_targets, ...
                                    hyperparameters);
                        
    [cross_entropy_train, frac_correct_train] = evaluate(train_targets, predictions);
    
    result.cost_train_list = [result.cost_train_list, cross_entropy_train];
    result.accuracy_train_list = [result.accuracy_train_list, frac_correct_train];

    weights = weights - hyperparameters.learning_rate .* df;

    predictions_valid = logistic_predict(weights, valid_inputs);
    [cross_entropy_valid, frac_correct_valid] = evaluate(valid_targets, predictions_valid);
    
    result.cost_valid_list = [result.cost_valid_list, cross_entropy_valid];
    result.accuracy_valid_list = [result.accuracy_valid_list, frac_correct_valid];
end
end