function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

[row, col] = size(z);
g = zeros(row, col);
for i = 1:row
    for j = 1:col
        g(i, j) = (1 ./ (1 + exp(-z(i, j)))) * (1 - 1 ./ (1 + exp(-z(i, j))));
    endfor
endfor












% =============================================================




end
