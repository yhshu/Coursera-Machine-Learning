function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
% theta是权重。
% 本函数用于多元逻辑回归分类，并返回所有分类器到all_theta矩阵中。在all_theta
% 矩阵中，第i行记录第i个分类器。
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);% m是X的行数，即样本数量。
n = size(X, 2);% n是X的列数，即特征数量。

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1); % 为常数项多加一列

% Add ones to the X data matrix
% 在X矩阵的左边添加一列1。
X = [ones(m, 1) X]; % 为常数项多加一列

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%		theta(:)会返回一个列向量。
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
for k = 1:num_labels 				% 循环num_labels次，循环一次区分出每一种类型。
    initial_theta = zeros(n + 1, 1); 	% 初始化theta
    options = optimset('GradObj', 'on', 'MaxIter', 50);
	% 最多迭代50次。
    [theta] = fmincg(@(t)(lrCostFunction(t, X, (y==k), lambda)), initial_theta, options);
	% 此处使用了函数句柄。@是方法指针。(y==k)返回是否是第k类。
	% 使用lrCostFunction计算该样本在逻辑回归中的theta
    all_theta(k, :) = theta' ;		% 记录当前样本得到的theta
end  

% =========================================================================


end
