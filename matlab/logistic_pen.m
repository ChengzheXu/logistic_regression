function [f, df, y] = logistic_pen(weights, data, targets, hyperparameters)
% Calculate log likelihood and derivatives with respect to weights.
%
% Note: N is the number of examples and 
%       M is the number of features per example.
%
% Inputs:
% 	weights:    (M+1) x 1 vector of weights, where the last element
%               corresponds to bias (intercepts).
% 	data:       N x M data matrix where each row corresponds 
%               to one data point.
%   targets:    N x 1 vector of targets class probabilities.
%   hyperparameters: The hyperparameter structure
%
% Outputs:
%	f:             The scalar error value.
%	df:            (M+1) x 1 vector of derivatives of error w.r.t. weights.
%   y:             N x 1 vector of probabilities. This is the output of the classifier.
%
e = 0.0000001;
input_data = [data, ones(size(data, 1), 1)];
y = sigmoid(input_data * weights);
f = -sum(targets.*log(y+e)+(1-targets).*log(1-y+e))+hyperparameters.weight_regularization*sum(weights(1:size(weights, 1)-1,:).^2)/2.0;
df = ((input_data'*(y-targets))/size(data, 1)+hyperparameters.weight_regularization.*[weights(1:size(weights, 1)-1,:);0]);

end
