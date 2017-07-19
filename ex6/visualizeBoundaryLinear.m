function visualizeBoundaryLinear(X, y, model)
%VISUALIZEBOUNDARYLINEAR plots a linear decision boundary learned by the
%SVM
%   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a linear decision boundary 
%   learned by the SVM and overlays the data on it
%	划出一条通过支持向量机学习得到的线性决策边界，并将数据可视化。

w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100);	% linspace均分计算指令
yp = - (w(1)*xp + b)/w(2);
plotData(X, y);
hold on;
plot(xp, yp, '-b'); 
hold off

end
