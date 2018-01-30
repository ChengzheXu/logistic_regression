% Learn a Naive Bayes classifier on the digit dataset, evaluate its
% performance on training and test sets, then visualize the mean and variance
% for each class.

load mnist_train;
load mnist_test;

% Add your code here (it should be less than 10 lines)
[log_prior, class_mean, class_var] = train_nb(train_inputs, train_targets);
[~,train_accuracy] = test_nb(train_inputs, train_targets, log_prior, class_mean, class_var)
[~,test_accuracy] = test_nb(test_inputs, test_targets, log_prior, class_mean, class_var)
subplot(221);mesh(reshape(class_mean(1,:),[28,28]));title('mean image of class 0')
subplot(222);mesh(reshape(class_mean(2,:),[28,28]));title('mean image of class 1')
subplot(223);mesh(reshape(class_var(1,:),[28,28]));title('variance image of class 0')
subplot(224);mesh(reshape(class_var(2,:),[28,28]));title('variance image of class 1')
