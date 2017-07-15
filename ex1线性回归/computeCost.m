function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%	计算线性回归的代价。用theta作为参数拟合数据集。

% Initialize some useful values
m = length(y); % number of training examples；m是训练集大小

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

J = sum((X*theta - y).^2) / (2 * m);
% m组数据，1个特征，X是m*2的矩阵，theta是2*1的矩阵，sum计算矩阵所有元素之和

% =========================================================================

end
