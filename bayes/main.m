hand = imread('hand.ppm', 'ppm');
book = imread('book.ppm', 'ppm');
% imagesc(hand);
% figure;
% imagesc(book);

data1 = normalize_and_label(hand, 0);
data1 = data1(1:10,:);
data2 = normalize_and_label(book, 1);
data2 = data2(1:10,:);
test_data = [data1; data2];
figure;
hold on;
plot(data2(:,1), data2(:,2), '.');
plot(data1(:,1), data1(:,2), '.r');
legend('Hand holding book', 'hand');
xlabel('green');
ylabel('red');


[mu, sigma] = bayes(test_data);

theta  = 0:0.01:2*pi;
x1 = 2*sigma(1,1) * cos(theta) + mu(1,1);
y1 = 2*sigma(1,2) * sin(theta) + mu(1,2);
x2 = 2*sigma(2,1) * cos(theta) + mu(2,1);
y2 = 2*sigma(2,2) * sin(theta) + mu(2,2);
plot(x1, y1, 'r');
plot(x2, y2);