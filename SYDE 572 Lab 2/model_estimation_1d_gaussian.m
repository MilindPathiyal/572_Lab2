function [ mu_estimation, sigma_estimation ] = model_estimation_1d_gaussian( X )
mu_estimation = sum(X) / length(X);
var_e = sum((X - mu_estimation).^2) / length(X);
sigma_estimation = sqrt(var_e);
end