function c = adaboost_discriminant(data, mu, sigma, p, alpha, classes, T)
% data = feature vectors
% mu = average point
% sigma = average distance to average point
% p = prior
% alpha = 1/2*ln((1-et)/et) where et is the error in the hypothesis
% classes = not even used. not mad at all.
% T = number of hypothesis
C = 2;
M = size(data, 1);
c = zeros(M, 1);

% calculate the hypothesis
hypothesis = zeros(M,T);

for i = 1:T
    g = discriminant(data, mu(:,:,i), sigma(:,:,i), p(:,i));
    [~, h] = max(g, [], 2);
    h = h - 1;
    hypothesis(:,i) = h;
end


for m = 1:M
    best_score = -1;
    best_class = -1;
    for i = 1:C
        score = 0;
        for t = 1:T
            if hypothesis(m,t) == (i - 1)
                score = score + alpha(t);
            end
        end

        if score > best_score
            best_score = score;
            best_class = i - 1;
        end
    end
    c(m) = best_class;
end



