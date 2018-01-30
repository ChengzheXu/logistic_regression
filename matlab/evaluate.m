function [ce, frac_correct] = evaluate(targets, y)
%    Compute evaluation metrics.
%    Inputs:
%        targets : N x 1 vector of binary targets. Values should be either 0 or 1.
%        y       : N x 1 vector of probabilities.
%    Outputs:
%        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we
%                       want to compute CE(targets, y).
%        frac_correct : (scalar) Fraction of inputs classified correctly.
e = 0.0000001;
ce = -mean(targets.*log(y+e)+(1-targets).*log(1-y+e));
count = 0;
y_modify = zeros(size(y));
for i=1:size(y, 1)
    y_modify(i) = (y(i)>=0.5);
end
for i=1:size(targets, 1)
    if(targets(i)==y_modify(i))
        count=count+1;
    end
end
frac_correct = count/size(targets, 1);
end
