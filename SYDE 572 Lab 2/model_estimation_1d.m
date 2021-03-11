%Driver Code for Part 2 
%Model Estimation 1-D case

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
[mu_estimation, sigma_estimation] = model_estimation_1d_gaussian(a);

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
[mu_estimation, sigma_estimation] = model_estimation_1d_gaussian(b);

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
lambda_estimation = model_estimation_1d_exponential(a);

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
lambda_estimation = model_estimation_1d_exponential(b);

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
% [Insert Princely's Part]

%% Non-Parametric Estimation 1D 
% [Insert Princely's Part]