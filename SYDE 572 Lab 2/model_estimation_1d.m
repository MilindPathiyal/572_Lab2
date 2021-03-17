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
% Dataset A

[ML_a_a, ML_a_b] = model_estimation_1d_uniform(a);
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

[ML_b_a, ML_b_b] = model_estimation_1d_uniform(b);
uniform_est = unifpdf(x_b, ML_b_a, ML_b_b);

%Plotting
figure(6);
hold on;
plot(x_b,exp_true, 'b');
plot(x_b,uniform_est, 'r');
scatter(b,y_b);
title('Model Estimation 1D Case - Uniform - Set B');
legend('Normal Distribution', 'Estimated Distribution', 'Dataset-B Distribution');
hold off;

%% Non-Parametric Estimation 1D 

% Dataset B
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
plot(x_b,exp_true, 'b');
plot(x_b, est_3, 'r');
plot(x_b, est_4, 'g');
scatter(b,y_b);
title('Model Estimation 1D Case - Non-Parametric Estimation - Set B');
legend('Normal Distribution', 'Estimated Distribution 1', 'Estimated Distribution 2', 'Dataset-B Distribution');
hold off;




