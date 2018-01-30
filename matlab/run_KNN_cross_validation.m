close all
clear all

load('mnist_train.mat')
load('mnist_valid.mat')
load('mnist_test.mat')

k_list = [1,2,3,4,5,6,7,8,9,10];

subplot(211)
[k_opti, highest_classification] = KNN_cross_validation(k_list, train_inputs, train_targets, valid_inputs, valid_targets)
title('validation')
subplot(212)
[k_opti_test, highest_classification_test] = KNN_cross_validation(k_list, train_inputs, train_targets, test_inputs, test_targets)
title('test')

