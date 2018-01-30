function [k_opti, highest_classification] = KNN_cross_validation(k_list, train_data, train_labels, valid_data, valid_labels)
classification_list = [];
for k = k_list
    valid_pred_labels = run_knn(k, train_data, train_labels, valid_data);
    N_valid = size(valid_pred_labels, 1);
    count = 0;
    for i=1:N_valid
        if(valid_pred_labels(i)==valid_labels(i))
            count=count+1;
        end
    end
    count = count/N_valid;
    classification_list = [classification_list, count];
end
[highest_classification, k_index] = max(classification_list);
k_opti = k_list(k_index);
plot(k_list, classification_list)
xlabel('k')
ylabel('classification rate')
title('K-NN')
grid on