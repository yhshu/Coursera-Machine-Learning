function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
H = X * theta;	% 预测值
J = sum((H - y) .^ 2) / (2 * m) + (sum(theta(2:size(theta)) .^ 2)) * lambda / (2 * m);

grad(1)= X(:, 1)' * (H - y) / m; % grad(1)没有正则项，因为theta0不正则化
grad(2:size(grad)) = X(: ,2:size(grad))' * (H - y) / m + theta(2:size(theta)) * lambda / m;

% =========================================================================

grad = grad(:);	% grad元素按列排列

end
