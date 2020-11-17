function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
%data=load('ex1data2.txt');
%X=[data(:,1),data(:,2)];
%y=data(:,3);

% Initialize some useful values
m = length(y); % number of training examples
%X=[ones(m,1),X]
%alpha = 0.01
%theta=zeros(3,1)
%theta=[-1;2;3]
%iterations=1500

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
htheta=X*theta
A=htheta-y
J = (1/(2*m))*(A'*A)

% =========================================================================

end
