function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
% Initialize some useful values
%%%%%%%%%%%%%%%%%%%%
% predict active
%X=[1;4];
%y = [100;114.03];
%%%%%%%%%%%%%%%%%%%
% predict closed
X = [1;4];
y = [100;103.45];
%%%%%%%%%%%%%%%%%%%
[m,n]=size(X);
X  = [ones(m,1),X];
theta=[ones(n+1,1)];
alpha = 0.01;
num_iters=3000;
J_history = zeros(num_iters, 1);

for iter = 1:num_iters;

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %
htheta=X*theta;
A=htheta-y;
dJ = ((1/m)*alpha*(A'*X));
theta = theta' - dJ;
theta = theta';
    % ============================================================

    % Save the cost J in every iteration    
J_history(iter) = computeCostMulti(X, y, theta)
end
end
