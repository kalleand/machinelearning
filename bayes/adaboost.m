function [mu, sigma, p, alpha, classes] = adaboost(data, T)

% mu = average point
% sigma = average distance to average point
% p = prior
% alpha = 1/2*ln((1-et)/et) where et is the error in the hypothesis
% classes = its either one or zero... like always!
% data = feature vectors
% T = number of hypothesis
M = size(data, 1);
C = 2; % number of classes
N = 2; % number of features
mu = zeros(C,N,T);
sigma = zeros(C,N,T);
p = zeros(C,T);
alpha = zeros(T, 1);
classes = [0,1]'; % never used. 

% We initialize the weights to uniform 1/number of points
weights = zeros(M, T+1);
weights(:,1) = ones(M, 1) / M;

for it = 1:T
    p(:,it) = prior(data, weights(:,it));
    [mu_it, sigma_it] = bayes_weight(data, weights(:,it));
    mu(:,:,it) = mu_it;
    sigma(:,:,it) = sigma_it;
    g = discriminant(data(:,1:2), mu_it, sigma_it, p(:,it));
    [~, h] = max(g, [], 2);
    h = h - 1;
    cor = (h ==data(:,3));
    et = 1.0 - sum(weights(:,it)'*cor);
    alpha(it) = 0.5 * log((1-et)/et);
    for m = 1:M
        if cor(m) == 1
            factor = exp(-alpha(it));
        else
            factor = exp(alpha(it));
        end
        weights(m,it+1) = weights(m,it) * factor;
    end
    weights(:,it+1) = weights(:,it+1) / sum(weights(:,it+1));
end