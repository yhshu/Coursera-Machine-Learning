function [J, grad] = lrCostFunction(theta, X, y, lambda)
% LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
% regularization
% 计算带有正则化的逻辑斯蒂回归的成本和梯度。

%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
%	
%	X是自变量，y是实际值，theta是系数。
%	lambda是正则化的参数。

% Initialize some useful values
m = length(y); % number of training examples
%	m是训练集样本个数。

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.计算一个特定theta的成本。
%               You should set J to the cost.应把J设成成本。
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%				计算偏导数，并将成本的偏导数设置为梯度
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%		
%           sigmoid(X * theta)
%		成本函数和梯度的计算能够高效地被向量化。例如这个sigmoid函数的计算。
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
h = sigmoid(X * theta);
thetatmp = [0; theta(2: end)]; % 在theta第一行加入0
J = (1/m) * (-y' * log(h) - (1-y') * log(1-h)) + (lambda / (2 * m)) * (thetatmp' * thetatmp);
grad = (1/m) * ( X' * (h-y)) + ((lambda/m) * thetatmp);

% =============================================================

grad = grad(:);

end
