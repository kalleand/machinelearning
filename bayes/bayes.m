% data = 3x[number of pixels * number of images]
% mu = 2x2 (CxN)
% sigma = 2x2 (CxN)

function [mu, sigma] = bayes(data)

[mu, sigma] = bayes_weight(data, ones(size(data, 1),1)/size(data, 1));
