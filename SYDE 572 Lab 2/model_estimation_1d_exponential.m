function[ lambda_estimation ] = model_estimation_1d_exponential( a )
lambda_estimation = 0;
for i = 1:length(a)
    lambda_estimation = lambda_estimation + a(i);
end

lambda_estimation = length(a) / lambda_estimation;
end