function res = normalize_with_black(image)
h = size(image, 1);
w = size(image, 2);
res = zeros(h, w, 2);
for i = 1:h
    for j = 1:w
        s = sum(image(i,j,:));
        if s > 0
            res(i,j,:) = [double(image(i,j,1))/s, double(image(i,j,2))/s];
        end
    end
end