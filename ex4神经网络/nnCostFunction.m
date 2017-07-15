function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%	nnCostFunction函数用于计算神经网络的成本和梯度。神经网络的参数没有被展开到
%   向量nn_params里，需要被转换回权重矩阵。
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% 重塑神经网络的权重矩阵。

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);	% m是X的行数，即样本数量。
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); % Theta1的梯度
Theta2_grad = zeros(size(Theta2)); % Theta2的梯度

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
%			前向传播神经网络并返回在变量J中的成本。
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%			执行反向传播算法来计算梯度Theta1_grad 和 Theta2_grad。你应该分别返回成本函数
%			的两个偏导数。执行完这个部分，你能通过执行checkNNGradients检查操作的正确性。
%			
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%				传入函数的y向量包含1到K的值，你需要将这个向量转换为二元的由0和1
%				组成的，这样能够被神经网络的成本函数所用。
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%				
%				如果你第一次执行的话，我们建议对训练集使用for循环执行反向传播。
%
% Part 3: Implement regularization with the cost function and gradients.
%			执行有成本函数和梯度的正则化。
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m , 1)  X];		% X大小5000 x 401

% Part 1: CostFunction 成本函数
% -------------------------------------------------------------

% Theta1大小25 x 401
% Theta2大小10 x 26
a1 = X; 					% 输入层 X大小5000 x 401
z2 = a1 * Theta1'; 			% 第二层输入 z2大小5000 x 25
a2 = sigmoid(z2); 			% 第二层输出
a2 = [ones(m, 1) a2]; 		% 加入偏置神经元，加入后a2大小5000 x 26
a3 = sigmoid(a2 * Theta2' );% 输出层 得到5000x10的矩阵

ry = eye(num_labels)(y, :); % ry是5000x10的矩阵，y是5000x1的矩阵。

cost = ry.* log(a3) + (1 - ry).* log(1 - a3); % 最大似然法。
J = -sum(sum(cost,2)) / m; 	% 成本函数。

reg = sum(sum(Theta1(:, 2:end).^2)) + sum(sum(Theta2(:, 2:end).^2));

J = J + lambda/(2 * m) * reg;% 正则化

% -------------------------------------------------------------

% Part 2: Backpropagation algorithm 反向传播
% -------------------------------------------------------------


delta3 = a3 - ry; % 第三层的误差
delta2 = (delta3*Theta2)(:,2:end) .* sigmoidGradient(z2);

% 估计误差
Delta1 = delta2'*a1;
Delta2 = delta3'*a2;

% 正则化
Theta1_grad = Delta1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m; % theta0不需要正则化
Theta2_grad = Delta2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;

%G1 = zeros(size(Theta1));
%G2 = zeros(size(Theta2));
%for i = 1 : m,
%	ra1 = X(i,:)';
%	rz2 = Theta1*ra1;
%	ra2 = sigmoid(rz2);
%	ra2 = [1;ra2];
%	rz3 = Theta2*ra2;
%	ra3 = sigmoid(rz3);
	
%	err3 = ra3 - ry(i,:)';
	
%	err2 = (Theta2'*err3)(2:end , 1) .* sigmoidGradient(rz2);
	
%	G1 = G1 + err2 * ra1';
%	G2 = G2 + err3 * ra2';
%end


%Theta1_grad = G1 / m + lambda*[zeros(hidden_layer_size , 1) Theta1(:,2:end)] / m;
%Theta2_grad = G2 / m + lambda*[zeros(num_labels , 1) Theta2(:,2:end)] / m;

% =========================================================================

% Unroll gradients 展示梯度
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
