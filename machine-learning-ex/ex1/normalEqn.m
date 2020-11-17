function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

%data=load('ex1data1.txt')
%X=data(:,1)
%y=data(:,2)
%theta = zeros(size(X, 2), 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
m=length(y)
XX=[ones(m,1),X]
theta=(pinv(XX'*XX))*(XX'*y)
% -------------------------------------------------------------
% ============================================================
%plot(X(:,1),y,'o',X(:,1),XX*theta,'-')
end
