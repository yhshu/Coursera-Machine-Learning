function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%	计算X的协方差矩阵中的特征向量，返回特征向量U，即S中的特征值（矩阵主对角线）

% Useful values
[m, n] = size(X);	% m行n列矩阵

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%
Sigma = 1 / m * X' * X;
[U, S, V] = svd(Sigma);	% 奇异值分解

% =========================================================================

end
