function z = generateDataGivenCovar(n,d,Sigma)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

z= randn(n,d)';
z= bsxfun(@minus, z, mean(z));
z= z*inv(chol(cov(z)));
z= z*chol(Sigma);
z=z';
end