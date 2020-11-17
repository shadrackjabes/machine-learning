function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
%data=load('ex2data1.txt');
%X=data(:,[1,2]);
%y=data(:,3);

% Initialize some useful values
%[m, n] = size(X);
%X = [ones(m,1),X;];
[m, n] = size(X);

% Initialize the fitting parameters
%theta = zeros(n, 1);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
num_iters = 1;
alpha = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%for iter = 1:num_iters
z=X*theta;
htheta=1./(1+exp(-z));
A = (htheta-y);
grad=(1/(m))*(A'*X)
theta = theta' - (alpha.*grad);
theta = theta';
lnB=log(1-htheta);
lnB(lnB==-Inf)=0;
J = (1/(m))*((-y'*log(htheta))) - (1/(m))*((1-y)'*lnB)
%J = (1/(m))*((-y'*log(htheta))) - (1/(m))*((1-y)'*log(1-htheta))
%J_history(iter) = J;
%end

% =============================================================
%disp('grad');disp(grad);
%disp('cost');disp(J);
%plotDecisionBoundary(theta, X, y);
end
