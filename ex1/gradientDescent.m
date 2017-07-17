function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
%	以alpha为学习率，num_iters为迭代次数，更新theta

% Initialize some useful values
m = length(y); % number of training examples；m是训练集大小
J_history = zeros(num_iters, 1);	% 用于存储每次迭代得到的代价

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
	% 下面执行一个简单的梯度步骤
	% 打印每次迭代获得的代价有助于debug
	
	H = X * theta; 	% m组数据，1个特征，X是m*2的矩阵，theta是2*1的矩阵，H和y是m维向量
	T = [0 ; 0];	% T用于求和,T=zero(2,1)，和theta维度相同
	for i = 1 : m,
		T = T + (H(i) - y(i)) * X(i, :)';	% 取X第i行，转置
	end
	
	theta = theta - (alpha * T) / m;
	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
