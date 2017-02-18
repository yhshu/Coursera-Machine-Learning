function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	H = X * theta; % m组数据，1个特征，X是m*2的矩阵，theta是2*1的矩阵，H和y是m*1的矩阵
	T = [0 ; 0];% T用于求和,T=zero(2,1)，和theta维度相同
	for i = 1 : m,
		T = T + (H(i) - y(i)) * X(i,:)';	% 取X第i行，转置
	end
	
	theta = theta - (alpha * T) / m; % alpha是学习率
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
