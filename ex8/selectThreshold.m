function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%	根据验证集和真实值的结果找到用于选择异常值的最佳阈值。

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;	% 步长
for epsilon = min(pval):stepsize:max(pval)	% 从最小预测值循环到最大预测值
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
	% 若预测的概率低于epsilon，认为它是异常。
	
	% yval says it's an anomaly and so algorithm does.
    tp = sum((yval == 1) & (pval < epsilon));

    % yval says it's not an anomaly,  but algorithm says anomaly.
    fp = sum((yval == 0) & (pval < epsilon));

    % yval says it's an anomaly,  but algorithm says not anomaly.
    fn = sum((yval == 1) & (pval >= epsilon));

    % precision and recall
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);

    % F1 value;
    F1 = (2 * prec * rec) / (prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
