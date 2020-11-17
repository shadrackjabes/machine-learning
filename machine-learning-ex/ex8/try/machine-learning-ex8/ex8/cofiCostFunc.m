function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

htheta=(X*Theta');
A=(htheta-Y).*R;
J = 0;
for i=1:num_users;
	J = J + ((1./2).*(A(:,i)'*A(:,i)))
end

Theta_grad = X'*A;
Theta_grad = Theta_grad';
Theta_grad1 = lambda.*Theta;
Theta_grad = Theta_grad + Theta_grad1;

X_grad = A*Theta;
X_grad1 = lambda.*X;
X_grad = X_grad + X_grad1;

J1 = 0;
for i = 1:num_users;
	J1 = J1 + ((lambda./2).*(Theta(i,:)*Theta(i,:)'));
end

J2 = 0;
for i = 1:num_movies;
	J2 = J2 + ((lambda./2).*(X(i,:)*X(i,:)'));
end
J = J + J1 + J2;














% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
