function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    theta1= theta(1) - (alpha/m) * sum((X*theta)-y);
    theta2=theta(2) - (alpha/m) * sum( ( (X*theta)-y ) .*X (:,2) );
    theta=[theta1;theta2];
    J_history(iter) = computeCost(X, y, theta);
end

end
