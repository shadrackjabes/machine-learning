function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
%data=load('ex1data1.txt')
%X=data(:,1);
%y=data(:,2);
% Initialize some useful values
m = length(y); % number of training examples
%XX=[ones(m,1),X]
%theta=zeros(2,1)
%num_iters = 500
%alpha = 0.01
J_history = zeros(num_iters, 1);
theta0_val=zeros(num_iters,1)
theta1_val=zeros(num_iters,1)
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
htheta=X*theta
A=htheta-y
dJ = ((1/m)*alpha*(A'*X))
theta = theta' - dJ
theta = theta'

    % ============================================================
    % Save the cost J in every iteration    
J_history(iter) = computeCost(X, y, theta);
theta0_val(iter) = theta(1)
theta1_val(iter) = theta(2)
%fprintf('Theta, cost:\n%f,\n%f,\n%f',theta0(iter),theta1(iter),J_history(iter)
end
% Display gradient descent's result
fprintf('Theta computed from gradient descent:\n%f,\n%f',theta(1),theta(2))
% Plot the linear fit
hold on; % keep previous plot visible
plot(X(:,1),y,'o',X(:,1),X*theta, '-')
legend('Training data', 'Linear regression')
hold off % don't overlay any more plots on this figure

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] *theta;
fprintf('For population = 35,000, we predict a profit of %f\n', predict1*10000);
predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n', predict2*10000);


% Visualizing J(theta_0, theta_1):
% Grid over which we will calculate J
theta0_vals = linspace(-10, 10, 100);
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
J_vals = J_vals';
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
hold off;
end
