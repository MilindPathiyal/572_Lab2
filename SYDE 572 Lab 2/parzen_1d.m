function [est] = parzen_1d( a, x, length, d)
%PARZEN_1D Summary of this function goes here
est = zeros(size(x));
for i=1:size(x,2)
    sum = 0;
    for j=1:size(a,2)
        sum = sum + normpdf(x(i), a(j), d);
    end
    est(i) = 1/length * sum;
end
end

