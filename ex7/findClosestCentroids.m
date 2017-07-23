function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%	返回最近的质心。idx是用于给每个样本指定质心的m x 1的向量。

% Set K
K = size(centroids, 1);		% K是聚类数，每列存储一个质心

% You need to return the following variables correctly.
idx = zeros(size(X, 1), 1); % idx是m维列向量

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i = 1 : size(X, 1);
  x = X(i, :);				% x是行向量
  EucDis = zeros(K, 1);	
  for centroid_i = 1 : K;   % 计算x到各质心的欧式距离
    EucDis(centroid_i) = (x - centroids(centroid_i, :)) * (x - centroids(centroid_i, :))';
  end
  [value, idx(i)] = min(EucDis);	% 取距离最短的质心，idx(i)记录第几类
end


% =============================================================

end

