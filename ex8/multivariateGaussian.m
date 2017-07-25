function p = multivariateGaussian(X, mu, Sigma2)
%MULTIVARIATEGAUSSIAN Computes the probability density function of the
%multivariate gaussian distribution.
%    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability 
%    density function of the examples X under the multivariate gaussian 
%    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
%    treated as the covariance matrix. If Sigma2 is a vector, it is treated
%    as the \sigma^2 values of the variances in each dimension (a diagonal
%    covariance matrix)
%	 在参数mu和Sigma2的多元高斯分布下计算X的概率密度函数。 
%	 如果Sigma2是矩阵，则将其视为协方差矩阵。
%	 如果Sigma2是向量，则将其视为每个维度中的方差的σ^2值（对角协方差矩阵）。

k = length(mu);

if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
    Sigma2 = diag(Sigma2);	% 对角协方差矩阵
end

X = bsxfun(@minus, X, mu(:)');	% bsxfun两个数组间元素逐个计算，@minus是函数句柄
p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

end