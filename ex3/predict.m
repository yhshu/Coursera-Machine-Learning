function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%	Theta1是第1层到第2层的权重矩阵，Theta2是第2层到第3层的权重矩阵。
%
% Useful values
m = size(X, 1); % X的行数，即样本个数。
num_labels = size(Theta2, 1); % Theta2的行数，即特征个数。

% You need to return the following variables correctly 
p = zeros(m, 1); % 每个样本得到一个结果

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
A1 = [ones(1, m); X' ];
X2 = sigmoid(Theta1 * A1);
A2 = [ones(1, m); X2 ];
A3 = sigmoid(Theta2 * A2);
[x, xi] = max(A3' , [], 2); 	% max函数用于获取每一行的最大值。
p = xi;

% =========================================================================


end
