clear;clc;
%% Define data
n=5;e=ones(n,1);
Sigma=randn(n);Sigma=Sigma*Sigma';
s=0.1;
mu_l=rand(n,1);
mu_u=mu_l+rand(n,1);


%% Run CVX
cvx_begin sdp quiet
cvx_precision high
% Define variables
variable x(n,1);
variable g(1);
variable D(n,n) diagonal;
% Cost function
maximize(g)
% Constraints
subject to
% Constraints on x: 0<=x_i<=1; sum_i(x_i)=1;
e'*x==1;
for i=1:n,
    x(i)>=0;
    x(i)<=1;
end
% Constraint on risk: x'*Sigma*x<=s
[s,x';x,inv(Sigma)]>=0;

% SDR
D>=0;
[mu_l'*D*mu_u-g,0.5*(x'-(mu_l+mu_u)'*D);
    (0.5*(x'-(mu_l+mu_u)'*D))',D]>=0;

cvx_end
x'
g
D=full(D)

%% Check solution
x
e'*x-1
eig([s,x';x,inv(Sigma)])
eig([mu_l'*D*mu_u-g,0.5*(x'-(mu_l+mu_u)'*D);
    (0.5*(x'-(mu_l+mu_u)'*D))',D])
D

