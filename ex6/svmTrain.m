function [model] = svmTrain(X, Y, C, kernelFunction, ...
                            tol, max_passes)
%SVMTRAIN Trains an SVM classifier using a simplified version of the SMO 
%algorithm. 
%   使用简单版的 SMO 算法（序列最小优化算法）训练一个支持向量机分类器。
%	本代码是求解 SVM 的核心部分，是 ex6 最值得研究的代码。
%   [model] = SVMTRAIN(X, Y, C, kernelFunction, tol, max_passes) trains an
%   SVM classifier and returns trained model. X is the matrix of training 
%   examples.  Each row is a training example, and the jth column holds the 
%   jth feature.  Y is a column matrix containing 1 for positive examples 
%   and 0 for negative examples.  C is the standard SVM regularization 
%   parameter.  tol is a tolerance value used for determining equality of 
%   floating point numbers. max_passes controls the number of iterations
%   over the dataset (without changes to alpha) before the algorithm quits.
%	训练一个 SVM 分类器并返回训练好的模型。X 是训练集矩阵。每一行是一个训练样本，
%	第 j 列记录第 j 个特征。Y 是列向量，1 与 0 表示类别。C 是标准支持向量机正则
%	化参数。tol 是用于确定浮点数相等的容差值。在算法退出之前，max_passes控制数
%	据集上的迭代次数（不改变 alpha的次数）。
%
% Note: This is a simplified version of the SMO algorithm for training
%       SVMs. In practice, if you want to train an SVM classifier, we
%       recommend using an optimized package such as:  
%		这是一个用于训练支持向量机的简化版SMO算法。如果你想训练支持向量机
%		分类器，我们建议使用如下的优化包。
%
%           LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
%           SVMLight (http://svmlight.joachims.org/)
%
%

% tol 的默认值是0.001
if ~exist('tol', 'var') || isempty(tol)
    tol = 1e-3; 
end

% max_passes 的默认值是5
if ~exist('max_passes', 'var') || isempty(max_passes)
    max_passes = 5;
end

% Data parameters
m = size(X, 1);
n = size(X, 2);

% Map 0 to -1
% 将标签0修改为标签-1
Y(Y==0) = -1; 

% Variables
alphas = zeros(m, 1);
b = 0;
E = zeros(m, 1);
passes = 0;
eta = 0;
L = 0;
H = 0;

% Pre-compute the Kernel Matrix since our dataset is small
% (in practice, optimized SVM packages that handle large datasets
%  gracefully will _not_ do this)
% 数据集较小，预计算核矩阵；实际运用中，针对大数据集不会做这一步。
% We have implemented optimized vectorized version of the Kernels here so
% that the svm training will run faster.
if strcmp(func2str(kernelFunction), 'linearKernel')
    % Vectorized computation for the Linear Kernel
    % This is equivalent to computing the kernel on every pair of examples
    K = X * X';
elseif strfind(func2str(kernelFunction), 'gaussianKernel')
    % Vectorized RBF Kernel
    % This is equivalent to computing the kernel on every pair of examples
    X2 = sum(X.^2, 2); % 按行求和
    K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
    K = kernelFunction(1, 0) .^ K;
else
    % Pre-compute the Kernel Matrix
    % The following can be slow due to the lack of vectorization
    K = zeros(m);
    for i = 1:m
        for j = i:m
             K(i,j) = kernelFunction(X(i,:)', X(j,:)');
             K(j,i) = K(i,j); %the matrix is symmetric % 对称矩阵
        end
    end
end

% Train
fprintf('\nTraining ...');
dots = 12;
while passes < max_passes, % 最大迭代次数以内
            
    num_changed_alphas = 0; % 发生alpha被改变的情况的迭代次数
    for i = 1:m, % 遍历数据集
        
        % Calculate Ei = f(x(i)) - y(i) using (2). 
        % E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
		% 预测值f(x(i))是 b + sum(alphas .* Y .* K(:, i))
        E(i) = b + sum(alphas .* Y .* K(:, i)) - Y(i);
		% 选择 i 时需要考虑
		% 如果误差很大，可考虑对 alpha 优化；对正间隔和负间隔都应测试
		% 如果 alphas(i) 已经在决策边界上，即等于0 或 C，就不必再优化
        if ((Y(i) * E(i) < -tol && alphas(i) < C) || (Y(i) * E(i) > tol && alphas(i) > 0)),
            % In practice, there are many heuristics one can use to select
            % the i and j. In this simplified code, we select them randomly.
			% 实际运用中，有许多启发式的方法去选择 i 和 j，此处我们随机选择。
            j = ceil(m * rand()); % 随机选择j，保证j与i不同；ceil函数是向右取整
            while j == i,  % Make sure i \neq j
                j = ceil(m * rand());
            end

            % Calculate Ej = f(x(j)) - y(j) using (2).
            E(j) = b + sum (alphas .* Y .* K(:,j)) - Y(j);

            % Save old alphas
            alpha_i_old = alphas(i);
            alpha_j_old = alphas(j);
            
            % Compute L and H by (10) or (11). 
            if (Y(i) == Y(j)),
                L = max(0, alphas(j) + alphas(i) - C);
                H = min(C, alphas(j) + alphas(i));
            else % if (Y(i) != Y(j))
                L = max(0, alphas(j) - alphas(i));
                H = min(C, C + alphas(j) - alphas(i));
            end
           
            if (L == H),
                % continue to next i. 
                continue;
            end

            % Compute eta by (14).
            eta = 2 * K(i, j) - K(i, i) - K(j, j);
            if (eta >= 0),
                % continue to next i. 
                continue;0
            end
            
            % Compute and clip new value for alpha j using (12) and (15).
			% 由目标函数对 alphas(j) 求偏导得：
            alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta; 
            
            % Clip
			% 为了 alphas(i) 符合 KKT 条件，确保 alphas(j) 在 H 与 L 之间.
            alphas(j) = min (H, alphas(j));
            alphas(j) = max (L, alphas(j));
            
            % Check if change in alpha is significant
            if (abs(alphas(j) - alpha_j_old) < tol), 
                % continue to next i. 
                % replace anyway		 % 如果变化的幅度很小
                alphas(j) = alpha_j_old; % alphas(j)不变化
                continue;				 % 跳转下一循环
            end
            
            % Determine value for alpha i using (16). 
			% 更新 alphas(i)
            alphas(i) = alphas(i) + Y(i) * Y(j) * (alpha_j_old - alphas(j));
            
            % Compute b1 and b2 using (17) and (18) respectively. 
            b1 = b - E(i) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
            b2 = b - E(j) ...
                 - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                 - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

            % Compute b by (19). 
            if (0 < alphas(i) && alphas(i) < C),
                b = b1;
            elseif (0 < alphas(j) && alphas(j) < C),
                b = b2;
            else
                b = (b1+b2)/2;
            end

            num_changed_alphas = num_changed_alphas + 1;

        end
        
    end
    
    if (num_changed_alphas == 0), % 此次迭代 alpha 未被修改
        passes = passes + 1;
    else % 此次迭代 alpha 被修改
        passes = 0;
    end

    fprintf('.');
    dots = dots + 1;
    if dots > 78
        dots = 0;
        fprintf('\n');
    end
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
end
fprintf(' Done! \n\n');

% Save the model
idx = alphas > 0;
model.X= X(idx,:);
model.y= Y(idx);
model.kernelFunction = kernelFunction;
model.b= b;
model.alphas= alphas(idx);
model.w = ((alphas .* Y)' * X)';

end
