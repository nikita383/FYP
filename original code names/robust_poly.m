function [w, risk, ret, sharpe] = robust_poly(n, m,p,f,asset_ret,omega,alpha,Gamma)
%% Set uncertainty paramaters

y=asset_ret;
B=f';
C=[ones(p,1) B'];
reg_result=(C'*C)^(-1)*C'*y; %[mu0, V0_1, ... ,V0_m]' 
mu0=reg_result(1,:)'*10;
V0=reg_result(2:m+1,:);
D=cov(y-C*reg_result);
F=cov(f);
V = (y'/B)';

% A = V - v0
A = V - V0;%tiny numbers
% M? - covar of uncert params ^ (-1/2) - checkkkk idk if i defined this
% right
%M=B*B'-(1/p)*(B*ones(p,1))*(B*ones(p,1))';
M = cov(A');
M = sqrtm(M);
M = inv(M);%huge numbers
A_tilde = M * A;

%unbiased estimate of variance - used for rho and gam
s2=zeros(n,1);
for i=1:n
    s2(i)=(1/(p-m-1))*(norm(y(:,i)-C*reg_result(:,i)))^2;
end

%critical value for f-dist
c_w=finv(omega,m+1,p-m-1);%critical value

% Delta? %%%%%%%%%%%%% HELP
pot_deltas = zeros(n,1);
for i=1:n
    pot_deltas(i)=sqrt((m+1)*c_w*s2(i));
end

Delta = min(pot_deltas);
Delta_hat = Gamma*Delta;

% gam - mu bound
gam=zeros(n,1);
tmp=(C'*C)^(-1);
tmp=tmp(1,1);
for i=1:n
    gam(i)=sqrt((m+1)*tmp*c_w*s2(i));
end

% F_ml=F_0- nominal covar matrix
F_0=(1/(p-1))*(B*B'-(1/p)*(B*ones(p,1))*(B*ones(p,1))');
single_omega=omega^(1/m);

% brute force to find value of eta 2 d.p.
x = 0:0.01:2;
y1 = gamcdf(x,(p+1)/2,2/(p-1));
gamval=[x;y1]';
%gamval=array2table(gamval);
y2=zeros(1,101);
for i=1:100
    y2(i+1)=y1(101+i)-y1(101-i);
    if y2(i+1)>=single_omega
        eta=0.01*i;
        break
    end      
end

% we have M, Delta, A tilde, F
% need y0 = v0*w, D_bar
D_bar = D;
e_n = ones(n,1);
e_m = ones(m,1);
zeros_m = zeros(m,1);
F_Minv = M\F;

%% Run CVX
cvx_begin sdp quiet
cvx_precision high

% Define variables
variable w(n,1);
variable phi(n,1);
variable delta(1);
variable nu_tau(1);
variable tau(1);
variable nu(1);

% Cost function
minimize(nu + delta)

% Constraints
subject to

% Constraints on w
for i=1:n,
    w(i)>=0;
    w(i)<=1;
end

e_n'*w==1;

% 
% Constraint on risk
[delta, w'; w, inv(D_bar)] >= 0; 

% Uncertainty set on mu
mu0'*w + gam'*phi >= alpha;
for i=1:n,
    phi(i)>=w(i);
    phi(i)>=-w(i);
end

% Uncertainty set on V
y0 = V0*w;
tau >= 0;

[(1-eta)*nu_tau zeros(1,m) y0' Delta_hat*w'*e_n; zeros(m,1) e_m*e_m' inv(M)' zeros(m,1); y0 inv(M) tau*F zeros(m,1); Delta_hat*e_n'*w zeros(1,m) zeros(1,m) eye(1)]>=0;

[(1+nu) nu 1 (nu_tau + tau); nu (1+nu) 0 0; 1 0 (1+nu) 0; (nu_tau + tau) 0 0 (1+nu)]>=0; %% doesnt work outside the seed

cvx_end

risk = w'*(V'*F*V)*w;
ret = (mu0-gam)'*w;
sharpe=ret/risk;
end