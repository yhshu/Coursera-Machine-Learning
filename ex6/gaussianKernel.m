function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%	返回x1和x2之间的径向基函数内核。
%
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
% 使x1和x2成为列向量。
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%
sim = exp(-sum((x1 - x2) .^ 2) / (2 * sigma ^2));
% 高斯核函数，对应的支持向量机是高斯径向基函数分类器。

% =============================================================
    
end
