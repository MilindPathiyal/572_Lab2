function [ mu_e, cov_e ] = model_estimation_2d_gaussian( X )
%GAUSSIAN2 Summary of this function goes here
%   Detailed explanation goes here

mu_e = sum(X) / length(X);
cov_e = cov(X);
end