clear; clc;

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
c_w=finv(omega,m+1,p-m-1);%critical value

% Delta?
pot_deltas = zeros(n,1);%+0.000001*ones(n_assets,1);
for i=1:n
    pot_deltas(i)=sqrt((m+1)*c_w*s2(i));
end
Delta = min(pot_deltas);

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
%% Run YALMIP
yalmip('clear')

%variables
w= sdpvar(n,1);
phi = sdpvar(n,1);
nu= sdpvar(1);
delta= sdpvar(1);
tau= sdpvar(1);

%constraints
Constraints = [sum(w) == 1];

for i = 1 : n,
  Constraints = [Constraints, w(i)>=0, w(i)<=1];
end
% 
Constraints = [Constraints, [delta,w';w,inv(D_bar)]>=0]; 
% 
% Uncertainty set on mu
Constraints = [Constraints, mu0'*w + gam'*phi >= alpha];
for i=1:n,
    Constraints = [Constraints,phi(i)>=w(i)];
    Constraints = [Constraints,phi(i)>=-w(i)];
end
% 
% Uncertainty set on V
y0 = V0*w;
Constraints = [Constraints,tau >= 0];
Constraints = [Constraints,[nu-y0'*F*y0,-y0'*F_Minv;-inv(M)'*F'*y0,-inv(M)'*F_Minv] - tau*[Delta^2*w'*(e_n*e_n')*w,zeros_m';zeros_m,e_m*e_m'] >= 0];

%obj
Objective = nu + delta;
options = sdpsettings('solver','bmibnb');
sol = optimize(Constraints,Objective,options);

value(w)
