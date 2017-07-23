function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%	通过计算已分类的点的平均值，返回一个新的质心。X中每行是一个数据点，
%	idx是一个列向量记录数据点属于的类别。K是聚类个数。你应该返回一个记录
%   质点的矩阵，矩阵中每行是一类点的平均值。

% Useful variables
[m n] = size(X);	% m行n列

% You need to return the following variables correctly.
centroids = zeros(K, n);	% K类，每类一行。


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for i = 1:K
	centroids(i, :) = mean(X([find(idx == i)], :));
end

% =============================================================


end

