%% Machine Learning Online Class - Exercise 1: Linear Regression

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
% x refers to the population size in 10,000s
% y refers to the profit in $10,000s
%

%% Initialization
clear all; close all; clc	% 清除变量，清理工作区，清屏命令行

%% ==================== Part 1: Basic Function ====================
% Complete warmUpExercise.m 
fprintf('Running warmUpExercise ... \n');
fprintf('5x5 Identity Matrix: \n');
warmUpExercise();

fprintf('Program paused. Press enter to continue.\n');
pause;		% 程序暂停


%% ======================= Part 2: Plotting =======================
fprintf('Plotting Data ...\n')
data = csvread('ex1data1.txt');		% 读取CSV文件数据，CSV即逗号分隔值文件格式
X = data(:, 1); y = data(:, 2);		% X是第一列所有数据，y是第二列所有数据
m = length(y); 						% number of training examples

% Plot Data
% Note: You have to complete the code in plotData.m
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =================== Part 3: Gradient descent ===================
fprintf('Running Gradient Descent ...\n')

X = [ones(m, 1), data(:, 1)]; 	% Add a column of ones to x
theta = zeros(2, 1); 			% initialize fitting parameters
								% zeros是两行一列的零向量

% Some gradient descent settings
iterations = 1500;				% 迭代次数
alpha = 0.01;					% 学习率

% compute and display initial cost
computeCost(X, y, theta)		% 计算初始代价

% run gradient descent			% 运行梯度下降
theta = gradientDescent(X, y, theta, alpha, iterations);

% print theta to screen			% 输出训练得到的theta
fprintf('Theta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

% Plot the linear fit			% 给线性拟合制图
hold on; % keep previous plot visible
plot(X(:, 2), X * theta, '-')	% 用'-'表示预测的数据点
legend('Training data', 'Linear regression')	% 图例
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============= Part 4: Visualizing J(theta_0, theta_1) =============
fprintf('Visualizing J(theta_0, theta_1) ...\n')

% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);	% 从-10到10，均分100个点
theta1_vals = linspace(-1, 4, 100);

% initialize J_vals to a matrix of 0's
J_vals = zeros(length(theta0_vals), length(theta1_vals));

% Fill out J_vals
for i = 1:length(theta0_vals)
    for j = 1:length(theta1_vals)
	  t = [theta0_vals(i); theta1_vals(j)];    
	  J_vals(i,j) = computeCost(X, y, t);
    end
end


% Because of the way meshgrids work in the surf command, we need to 
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';
% Surface plot
figure;		% 创建窗口
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
