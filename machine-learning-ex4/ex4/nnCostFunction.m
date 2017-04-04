function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X=[ones(m,1), X];
a1=sigmoid(Theta1*X');
a1=[ones(1,m);a1]; %26*5000
a2=sigmoid(Theta2*a1); %10*26 * 26*5000 = 10*5000
%[x,ix]=max(a2,[],1); %ix has the index of the row which has the biggest scores, so, the prediction for that training example
a2=a2'; %5000*10
for i=1:m
    for j=1:num_labels
        binarysol=(y==j)(i);
        J=J+( -binarysol*log(a2(i,j)) - (1-binarysol) * log(1-a2(i,j)) );
    end
end

J=J/m;

reg1=0;
reg2=0;
for row=1:size(Theta1,1)
    for col=2:size(Theta1,2)
        reg1=reg1+Theta1(row,col)^2;
    end
end

for row=1:size(Theta2,1)
    for col=2:size(Theta2,2)
        reg2=reg2+Theta2(row,col)^2;
    end
end

regcost=((reg1+reg2)*lambda)/(2*m);
J=J+regcost;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

targets=[1:num_labels];
targets=targets(:);
for t=1:m
    inputVec=X(t,:); %pick one training example
    inputVec=inputVec(:);
    a1=inputVec; %no need to add 1, as it is already in X.
    z2=Theta1*a1;
    a2=sigmoid(z2);
    a2=a2(:);
    a2=[1; a2]; %25 *1 %add bias
    z3=Theta2*a2; %10*1 END OF FORWARD PROPAGATIONS
    a3=sigmoid(z3);
    binasol=targets==y(t); %1*10
    delta3=a3-binasol;%10*1 GET ERROR
    delta2=Theta2(:,2:end)'*delta3;%bias term is not used in backpropagation
    secterm=sigmoidGradient(z2); %slope of sigmoid function for zvalues os layer 2
    delta2=delta2.*secterm; %multiply the error by the slope of the sigmoid function->direction and magnitude of the gradient!!!
    %delta2=delta2(2:end);
    Theta2_grad=Theta2_grad + delta3*a2';
    Theta1_grad=Theta1_grad + delta2*a1';
end
Theta2_grad=Theta2_grad/m;
Theta1_grad=Theta1_grad/m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

for i=1:size(Theta1_grad,1)
    for j=2:size(Theta1_grad,2)
        Theta1_grad(i,j)=Theta1_grad(i,j)+(lambda/m)*Theta1(i,j);
    end
end

for i=1:size(Theta2_grad,1)
    for j=2:size(Theta2_grad,2)
        Theta2_grad(i,j)=Theta2_grad(i,j)+(lambda/m)*Theta2(i,j);
    end
end


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
