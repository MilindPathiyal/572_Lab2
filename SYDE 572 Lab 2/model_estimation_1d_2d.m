%Driver Code for Lab 2 
%Model Estimation 1-D and 2-D case

close all;

%Load lab 2 dataset
load('lab2_1.mat');
y_a=zeros(size(a));
x_a = 0:0.01:max(a(1,:))+1;
y_b=zeros(size(b));
x_b = 0:0.01:max(b(1,:))+1;


%Define variables
mu = 5;
sigma = 1;
lambda = 1;

%% Parametric Estimation 1D – Gaussian
% Dataset A
%Gaussian Calculation
mu_estimation = sum(a) / length(a);
var_e = sum((a - mu_estimation).^2) / length(a);
sigma_estimation = sqrt(var_e);

gauss_true = normpdf(x_a,mu,sigma);
gauss_estimation = normpdf(x_a,mu_estimation, sigma_estimation);

%Plotting
figure(1);
hold on;
plot(x_a,gauss_true, 'b'); %Plot normal distribution
plot(x_a,gauss_estimation, 'r'); %Plot estimation distribution
scatter(a,y_a); %Plot dataset-a distribution
title('Model Estimation 1D Case - Gaussian - Set A');
legend('Normal Distribution', 'Estimated Distribution', 'Dataset-A Distribution');
grid on;
hold off;

% Dataset B

%Gaussian Calculation
mu_estimation = sum(b) / length(b);
var_e = sum((b - mu_estimation).^2) / length(b);
sigma_estimation = sqrt(var_e);

gauss_true = exppdf(x_b,1/lambda);
gauss_estimation = normpdf(x_b,mu_estimation, sigma_estimation);

%Plotting
figure(2);
hold on;
plot(x_b,gauss_true, 'b'); %Plot normal distribution
plot(x_b,gauss_estimation, 'r'); %Plot estimation distribution
scatter(b,y_b); %Plot dataset-b distribution
title('Model Estimation 1D Case - Gaussian - Set B');
legend('Normal Distribution', 'Estimated Distribution', 'Dataset-B Distribution');
grid on;
hold off;

%% Parametric Estimation 1D - Exponential
% Dataset A

lambda_estimation = 0;
for i = 1:length(a)
    lambda_estimation = lambda_estimation + a(i);
end

lambda_estimation = length(a) / lambda_estimation;

exp_true = normpdf(x_a,mu,sigma);
exp_estimation = exppdf(x_a, 1/lambda_estimation);

%Plotting
figure(3);
hold on;
plot(x_a,exp_true, 'b'); %Plot normal distribution
plot(x_a,exp_estimation, 'r'); %Plot estimation distribution
scatter(a,y_a); %Plot dataset-a distribution
title('Model Estimation 1D Case - Exponential - Set A');
legend('Normal Distribution', 'Estimated Distribution', 'Dataset-A Distribution');
grid on;
hold off;

% Dataset B
lambda_estimation = 0;
for i = 1:length(b)
    lambda_estimation = lambda_estimation + b(i);
end

lambda_estimation = length(b) / lambda_estimation;


exp_true = exppdf(x_b,1/lambda);
exp_estimation = exppdf(x_b, 1/lambda_estimation);

%Plotting
figure(4);
hold on;
plot(x_b,exp_true, 'b'); %Plot normal distribution
plot(x_b,exp_estimation, 'r');  %Plot estimation distribution
scatter(b,y_b); %Plot dataset-b distribution
title('Model Estimation 1D Case - Exponential - Set B');
legend('Normal Distribution', 'Estimated Distribution', 'Dataset-B Distribution');
grid on;
hold off;
%% Parametric Estimation 1D - Uniform
% Dataset A

ML_a_a = min(a);
ML_a_b = max(a);
uniform_est = unifpdf(x_a, ML_a_a, ML_a_b);
uniform_true = normpdf(x_a,mu,sigma);

% Plotting
figure(5);
hold on;
plot(x_a,uniform_true, 'b');
plot(x_a,uniform_est, 'r');
scatter(a,y_a);
title('Model Estimation 1D Case - Uniform - Set A');
legend('Normal Distribution', 'Estimated Distribution', 'Dataset-A Distribution');
grid on;
hold off;

% Dataset B
exp_true = exppdf(x_b,1/lambda);

ML_b_a = min(b);
ML_b_b = max(b);
uniform_est = unifpdf(x_b, ML_b_a, ML_b_b);

%Plotting
figure(6);
hold on;
grid on;
plot(x_b,exp_true, 'b');
plot(x_b,uniform_est, 'r');
scatter(b,y_b);
title('Model Estimation 1D Case - Uniform - Set B');
legend('Normal Distribution', 'Estimated Distribution', 'Dataset-B Distribution');
hold off;

%% Non-Parametric Estimation 1D 

% Dataset A
N_a = length(a);
N_b = length(b);
min_range = min(a(1,:))-1;
max_range = max(a(1,:))+1;
x_a_parzen = min_range:0.01:max_range;

true_1 = normpdf(x_a_parzen,mu,sigma);
est_1 = parzen_1d(a,x_a_parzen,N_a,0.1);
est_2 = parzen_1d(a,x_a_parzen,N_a,0.4);

figure(7);
hold on;
grid on;
plot(x_a_parzen,true_1, 'b');
plot(x_a_parzen,est_1, 'r');
plot(x_a_parzen,est_2, 'g');
scatter(a,y_a);
title('Model Estimation 1D Case - Non-Parametric Estimation - Set A');
legend('Normal Distribution', 'Estimated Distribution 1', 'Estimated Distribution 2', 'Dataset-A Distribution');
hold off;

% Dataset B
exp_true = exppdf(x_b,1/lambda);
est_3 = parzen_1d(b,x_b,N_b,0.1);
est_4 = parzen_1d(b,x_b,N_b,0.4);

figure(8);
hold on;
grid on;
plot(x_b,exp_true, 'b');
plot(x_b, est_3, 'r');
plot(x_b, est_4, 'g');
scatter(b,y_b);
title('Model Estimation 1D Case - Non-Parametric Estimation - Set B');
legend('Normal Distribution', 'Estimated Distribution 1', 'Estimated Distribution 2', 'Dataset-B Distribution');
hold off;


%% Model Estimation 2D 

clear;
% close all;
load('lab2_2.mat');

%% Parametric Estimation
lowx = min([min(al(:,1)), min(bl(:,1)), min(cl(:,1))]) - 10;
lowy = min([min(al(:,2)), min(bl(:,2)), min(cl(:,2))]) - 10;
highx = max([max(al(:,1)), max(bl(:,1)), max(cl(:,1))]) + 10;
highy = max([max(al(:,2)), max(bl(:,2)), max(cl(:,2))]) + 10;
step_size = 1.0;

x = lowx:step_size:highx;
y = lowy:step_size:highy;
[xx, yy] = meshgrid(x, y);

mu_al = sum(al) / length(al);
cov_al = cov(al);
mu_bl = sum(bl) / length(bl);
cov_bl = cov(bl);
mu_cl = sum(cl) / length(cl);
cov_cl = cov(cl);

%Perform ML Calculation
ML_ab = zeros(size(xx, 1), size(yy, 2));
ML_ac = zeros(size(xx, 1), size(yy, 2));
ML_bc = zeros(size(xx, 1), size(yy, 2));

inv_cov_al = inv(cov_al);
inv_cov_bl = inv(cov_bl);
inv_cov_cl = inv(cov_cl);

for i = 1:size(xx, 1)
    for j = 1:size(yy, 2)
    point = [xx(i,j), yy(i,j)];
    dist_al = (point - mu_al) * inv_cov_al * transpose(point - mu_al);
    dist_bl = (point - mu_bl) * inv_cov_bl * transpose(point - mu_bl);
    dist_cl = (point - mu_cl) * inv_cov_cl * transpose(point - mu_cl);
    
    ML_ab(i,j) = dist_bl - dist_al;
    ML_ac(i,j) = dist_al - dist_cl;
    ML_bc(i,j) = dist_cl - dist_bl;

    end
end

ML_boundaries = zeros(size(xx, 1), size(yy, 2));
for i = 1:size(xx,1)
    for j = 1:size(yy,2)
        [ignore, class] = min([ML_ab(i,j), ML_ac(i,j), ML_bc(i,j)]);
        ML_boundaries(i,j) = class;
    end
end

figure(9);
hold on;

map = [
    1, 0.5, 0.5
    0.5, 0.5, 1
    0.6,0.6,0.6];
colormap(map);

% Plotting ML decision boundary in black
contourf(xx, yy, ML_boundaries, 'Color', 'black');

title('Model Estimation 2D Case - Parametric Estimation');
legend('ML decision boundaries', 'Cluster a', 'Cluster b', 'Cluster c');
class_c = scatter(at(:, 1), at(:, 2), 'b+');
class_d = scatter(bt(:, 1), bt(:, 2), 'r*');
class_e = scatter(ct(:, 1), ct(:, 2), 'ko');
hold off;

%% Non-Parametric Estimation 2D 

sigma_sqr = 400;
cov_mat = [sigma_sqr 0; 0 sigma_sqr];
mu_2 = [sigma_sqr/2 sigma_sqr/2];

step_size = 1;
[x1,x2] = meshgrid(1:1:sigma_sqr);
par_window = mvnpdf([x1(:) x2(:)], mu_2, cov_mat);
par_window = reshape(par_window,length(x2),length(x1));

min_x = min([min(al(:,1)), min(bl(:,1)), min(cl(:,1))]) - 10;
min_y = min([min(al(:,2)), min(bl(:,2)), min(cl(:,2))]) - 10;
max_x = max([max(al(:,1)), max(bl(:,1)), max(cl(:,1))]) + 10;
max_y = max([max(al(:,2)), max(bl(:,2)), max(cl(:,2))]) + 10;

% Defining the area of interest
area = [step_size min_x min_y max_x max_y];

% Calculating distributions
[par_a, x_a, y_a] = parzen_2d(al,area, par_window);
[par_b, x_b, y_b] = parzen_2d(bl,area, par_window);
[par_c, x_c, y_c] = parzen_2d(cl,area, par_window);
    
% Plotting the Parzen window density estimate
figure(10);
hold on;
scatter(al(:,1),al(:,2));
contour(x_a,y_a,par_a);
scatter(bl(:,1),bl(:,2));
contour(x_b,y_b,par_b);
scatter(cl(:,1),cl(:,2));
contour(x_c,y_c,par_c);
title('Model Estimation 2D Case - Non-Parametric Estimation');
legend('Cluster a', 'Contour a', 'Cluster b', 'Contour b', 'Cluster c', 'Contour b');
hold off;

[x_2,y_2] = meshgrid(x_a, y_a);
ML_2 = zeros(size(x_2));
for i = 1:size(x_2,1)
   for j = 1:size(y_2,2)
       [max_par, class] = max([par_a(i,j), par_b(i,j), par_c(i,j)]);
       ML_2(i,j) = class;
   end
end

figure(11);
hold on;

% ML decision boundary 
contourf(x_2, y_2, ML_2);

map = [
    1, 0.5, 0.5
    0.5, 0.5, 1
    0.6,0.6,0.6];
colormap(map);

a = scatter(at(:, 1), at(:, 2), 'r*');
b = scatter(bt(:, 1), bt(:, 2), 'b+');
c = scatter(ct(:, 1), ct(:, 2), 'ko');
title('Model Estimation 2D Case - Non-Parametric Estimation');
legend('ML decision boundaries', 'Cluster a', 'Cluster b', 'Cluster c');

hold off;

%% Sequential 
% Compute 3 sequential classifiers and plot them
J = inf;
N = 3;
plot_boundary = true;
plot_error = false;

run sequential

% Limit to 5 sequential discriminants, but learn each of them 20 times and plot the error rates
J = 5;
N = 20;
plot_boundary = false;
plot_error = true;

run sequential

