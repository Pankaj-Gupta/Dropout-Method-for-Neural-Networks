%this code generates and prints two column
%of random numbers

num = 400
x1 = rand(num, 1);
x2 = rand(num, 1);
y = zeros(num, 1);
for i=1:num
    %if (x1(i) + 0.7*x2(i)  - 0.8)>0
    %    y(i)=1;
    %else
    %    y(i)=2;
    %end
    if ((x1(i)-0.5)^2 + (x2(i)-0.5)^2 - 0.37^2)>0
	y(i) =1;
    else
	y(i) = 2;
    end
end
wx = fopen("r.txt", "w");

for i=1:num
    fprintf(wx, '%.2f, %.2f, %d\n', x1(i), x2(i), y(i));
end

%fprintf(wx, '%.2f, %.2f, %d\n', (x1, x2, y));
fclose(wx);


fprintf(['Plotting data ']);

X = [x1 x2];
plotData(X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Feature 1')
ylabel('Feature 2')


hold off;
