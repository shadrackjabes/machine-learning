function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------
%%%%%%%%%%%%%%%%%%%%
% predict active
X=[1;4];
y = [100;114.03];
%%%%%%%%%%%%%%%%%%%
% predict closed
%X = [1;4];
%y = [100;103.45];
%%%%%%%%%%%%%%%%%%%

m=length(y)
XX=[ones(m,1),X]
theta=(pinv(XX'*XX))*(XX'*y)
% -------------------------------------------------------------
% ============================================================
%plot(X(:,1),y,'o',X(:,1),XX*theta,'-')
end
