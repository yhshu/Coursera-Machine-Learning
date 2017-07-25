function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
%   输入的X是数据集，每行是n维数据点。输出的是n维向量mu，是数据集的平均值。
%   方差sigma2是n×1向量。

% Useful variables
[m, n] = size(X);	% m行n列

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%
%				mu(i)应当包含第i个特征的数据的平均值。
%				sigma2(i)应当包含第i个特征的方差。
mu = mean(X)';			% 转置为列向量。
sigma2 = var(X, 1)';	% 按列求方差，再转置为列向量。


% =============================================================


end
