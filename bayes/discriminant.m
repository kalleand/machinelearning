% g = MxC matrix
% C = number of classes (hand or book)

function g = discriminant(data, mu, sigma, p)

% p = [alpha1, alpha2] (prior knowledge)
C = 2; % number of classes (hand or book)
M = size(data, 1); % number of datapoints.
% N = 2; % number of features

g = zeros(M,C);

for m = 1:M
    for i = 1:C
        sum1 = log(sigma(i,1)) + log(sigma(i,2));
        sum2 = (data(m,1) - mu(i,1))^2 / (2*sigma(i,1)^2) + (data(m,2) - mu(i,2))^2 / (2*sigma(i,2)^2);
        g(m,i) = log(p(i)) - sum1 - sum2;
    end
end
