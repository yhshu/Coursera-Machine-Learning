function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%	返回协同过滤问题的成本和梯度。

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%				为协同过滤计算代价函数和梯度。具体地，你应该首先实现代价函数
%				（不含正则项）并保证它与我们给定的代价吻合。在此之后，你应该
%				实现梯度并使用checkCostFunciton程序检查梯度是否正确。最后，
%				应该实现正则化。
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%		 X是电影特征。Y是用户对电影的评级。
%		 如果用户j给电影i平分，则R(i,j)是1.
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

Delta = (X * Theta' - Y) .* R;

% 代价函数
J = 1/2 * sum(sum(Delta .^ 2)) + lambda / 2 * sum(sum(Theta .^ 2)) + lambda / 2 * sum(sum(X .^ 2));

% 对代价函数求偏导
X_grad = Delta * Theta + lambda * X;
Theta_grad = Delta' * X + lambda * Theta;

% Fold back into vectors.
grad = [X_grad(:); Theta_grad(:)];

end
