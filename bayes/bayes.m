% data = 3x[number of pixels * number of images]
% mu = 2x2 (CxN)
% sigma = 2x2 (CxN)

function [mu, sigma] = bayes(data)

% Number of classes, 0 = hand and 1 = book
C = 2;

% Number of features, 1 = green and 2 = red
N = 2;


% total amount of classifications of hands and books
M = zeros(2,1);

% row = class (hand/book), column = x,y (green/red)
total = zeros(2,2);
for i = 1:size(data, 1)
    class = data(i,3) + 1;
    total(class, 1) = total(class, 1) + data(i,1);
    total(class, 2) = total(class, 2) + data(i,2);
    M(class) = M(class) + 1;
end

mu(1,1) = total(1, 1) / M(1); % mu for hand and green
mu(1,2) = total(1, 2) / M(1); % mu for hand and red
mu(2,1) = total(2, 1) / M(2); % mu for book and green
mu(2,2) = total(2, 2) / M(2); % mu for book and red


% row = class (hand/book), column = x,y (green/red)
err = zeros(2,2);

for i = 1:size(data, 1)
    % x value (green)
    class = data(i,3) + 1;
    err(class, 1) = err(class, 1) + (data(i,1) - mu(class, 1))^2;
    
    % y value (red)
    err(class, 2) = err(class, 2) + (data(i,2) - mu(class, 2))^2;
end

err
M
sigma(1,1) = sqrt(err(1,1) / M(1)); % sigma for hand and green
sigma(1,2) = sqrt(err(1,2) / M(1)); % sigma for hand and red
sigma(2,1) = sqrt(err(2,1) / M(2)); % sigma for book and green
sigma(2,2) = sqrt(err(2,2) / M(2)); % sigma for book and red
