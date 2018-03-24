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

X = [ones(m, 1) X];
%fprintf("x.size = [%d, %d]\n", size(X, 1), size(X, 2));
%fprintf("input_layer_size = %d, hidden_layer_size = %d, num_labels = %d\n", input_layer_size, hidden_layer_size, num_labels);
%fprintf("Theta1.size = [%d, %d], Theta2.size = [%d, %d]\n", size(Theta1, 1), size(Theta1, 2), size(Theta2, 1), size(Theta2, 2));

delta3 = zeros(num_labels, 1); % CORRECT
delta2 = zeros(size(Theta2, 2) - 1, 1);
level2 = zeros(size(Theta2));
level1 = zeros(size(Theta1));

for i=1:m
    singleX = X(i, :);
    a1 = singleX';
    z2 = Theta1 * singleX';
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    localy = zeros(num_labels, 1);
    if (y(i) == 0)
        localy(num_labels) = 1;
    else
        localy(y(i)) = 1;
    endif

    delta3 = a3 - localy;

    % step 3
    temp = Theta2' * delta3;
    delta2 = temp(2:end) .* sigmoidGradient(z2);

    level2 = level2 + delta3 * a2';
    level1 = level1 + delta2 * a1';

    J = J - (localy' * log(a3) + (1 - localy)' * log(1-a3));
endfor

Theta2_grad = (level2 + lambda * Theta2)/m;
Theta1_grad = (level1 + lambda * Theta1)/m;
Theta2_grad(:, 1) = Theta2_grad(:, 1) - lambda * Theta2(:, 1) / m;
Theta1_grad(:, 1) = Theta1_grad(:, 1) - lambda * Theta1(:, 1) / m;

J = J / m;

reg = 0;
reg = reg + sum(sumsq(Theta1(:, 2:end)));
reg = reg + sum(sumsq(Theta2(:, 2:end)));
reg = reg * lambda / (2 * m);
J = J + reg;

% gradient

grad = [Theta1_grad(:); Theta2_grad(:)];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
