%% Generate synthetic data

rng(42)
n = 5;
m = 3;
p = 100;

rf=3;
sz=[n, 1];
mu0=unifrnd(rf-2,rf+2,sz);
mu = repmat(mu0, 1, p);

cond_F = 21;
while cond_F >= 20
    F = generateSPDmatrix(m);
    cond_F = cond(F);
    %cond_F = vpa(cond_F);
end

V = randn(m,n);

D_diag_norm = norm(V'*F*V);
while D_diag_norm > 0.1*norm(V'*F*V)
    D_diag = rand(sz);
    D_diag_norm = norm(D_diag);
end
D = diag(D_diag);

f_epsilon = generateDataGivenCovar(m+n,p,blkdiag(F,D));

f = f_epsilon(1:m,:);
epsilon = f_epsilon((m+1):(n+m),:);

asset_ret=mu+V'*f+epsilon;

omega = 0.99;
alpha = 0;

%% Set uncertainty paramaters

%(mu0,V0)-least squares estimate of mu over all periods
y=asset_ret';
B=f;
C=[ones(p,1) B'];
reg_result=(C'*C)^(-1)*C'*y; %[mu0, V0_1, ... ,V0_m]' 
mu0=reg_result(1,:)';
V0=reg_result(2:m+1,:);

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
c_w=finv(power(omega,1/n),m+1,p-m-1);%critical value

% Delta? %%%%%%%%%%%%% HELP
pot_deltas = zeros(n,1);
for i=1:n
    pot_deltas(i)=sqrt((m+1)*c_w*s2(i));
end

Delta = min(pot_deltas);
Delta_hat = n * Delta;

% gam - mu bound
gam=zeros(n,1);
tmp=(C'*C)^(-1);
tmp=tmp(1,1);
for i=1:n
    gam(i)=sqrt((m+1)*tmp*c_w*s2(i));
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
variable tau_p(1);
variable inv_tau_p_nu(1);
variable inv_tau_b_tau_p(1);
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
tau_p >= 0;

inv_tau_b_tau_p >= 0;

[inv_tau_p_nu zeros(1,m) y0' Delta_hat*w'*e_n; zeros(m,1) e_m*e_m' inv(M)' zeros(m,1); y0 inv(M) tau_p*F zeros(m,1); Delta_hat*e_n'*w zeros(1,m) zeros(1,m) 1]>=0;

[inv_tau_p_nu zeros(1,m) y0' m*Delta*w'*e_n; zeros(m,1) inv_tau_b_tau_p*e_m*e_m' inv(M)' zeros(m,1); y0 inv(M) tau_p*F zeros(m,1); m*Delta*e_n'*w zeros(1,m) zeros(1,m) inv_tau_b_tau_p]>=0;

[nu inv_tau_p_nu; tau_p 1] >=0;

cvx_end
w
w'*V'*F*V*w