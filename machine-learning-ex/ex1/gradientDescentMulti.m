function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
%data=load('ex1data2.txt')
% Initialize some useful values
m = length(y); % number of training examples
%X1=zeros(m,1)
%X2=zeros(m,1)
%X1=(X(:,1)-mean(X(:,1)))/max(X(:,1))
%X2=(X(:,2)-mean(X(:,2)))/max(X(:,2))
%XX=[ones(m,1),X1,X2]
%y=(X(:,3)-mean(X(:,3)))/max(X(:,3))
%theta=zeros(3,1)
%alpha = 0.01
%num_iters=100
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
htheta=X*theta
A=htheta-y
dJ = ((1/m)*alpha*(A'*X))
theta = theta' - dJ
theta = theta'
    % ============================================================

    % Save the cost J in every iteration    
J_history(iter) = computeCostMulti(X, y, theta);
end
end
