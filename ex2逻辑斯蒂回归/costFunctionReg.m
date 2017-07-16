function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
Hx = sigmoid(X * theta);
J = 1/m * (-y'*log(Hx) - (1-y')*log(1-Hx)) + lambda/(2*m) * (theta(2:end)' * theta(2:end));  %正则化调整成本函数
% 对所有特征进行惩罚，不包括theta(0)

grad = 1/m * ((Hx-y)'*X) + lambda/m * theta'; %正则化调整梯度  
grad(1) = grad(1) - lambda/m * theta(1); 

% =============================================================

end
