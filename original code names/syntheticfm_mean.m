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

%%%%%%%%% CHANGE DEPENDING ON SYMMETRIC OR ASYMMETRIC
V = randn(m,n);
%%%%%%%

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

%--------sharpe ratio comparison as a function of CL---------------
%omega_vec=0.001:0.005:0.999;
omega_vec=0.01:0.005:0.999;
sz_omega=size(omega_vec);
sz_omega=sz_omega(2);

sharpe_comp=zeros(size(omega_vec));
ret_comp=zeros(size(omega_vec));
risk_comp=zeros(size(omega_vec));

clas_policy=zeros(n,sz_omega);
rob_policy=zeros(n,sz_omega);

alpha=0;

%--------sharpe ratio comparison as a function of CL---------------
omega_vec=0.001:0.005:0.999;
sz_omega=size(omega_vec);
sz_omega=sz_omega(2);

sharpe_comp=zeros(size(omega_vec));
ret_comp=zeros(size(omega_vec));
risk_comp=zeros(size(omega_vec));

clas_policy=zeros(n,sz_omega);
rob_policy=zeros(n,sz_omega);

for i=1:sz_omega
    %omega=0.001;
    %--------classical optimisation---------
%     [w,risk,ret,sharpe]= markowitz(n,p,f',asset_ret',alpha);
    [w,risk,ret,sharpe]= classical_minvar(n,p,f',asset_ret',alpha);
    %clas_policy=w
    ret_vec(i)=ret;
    %--------robust optimisation---------
    [w2,risk2,ret2,sharpe2,gam,rho,G]= robust_elips(n, m,p,V,F,f,D,asset_ret,omega_vec(i),alpha);
    %rob_policy=w2
    
    sharpe_comp(i)=sharpe2/sharpe;
    %ret_comp=[ret; ret2]
    %risk_comp=[risk; risk2]
    ret_comp(i)=ret2/ret;
    risk_comp(i)=risk2/risk;
    gam;
    rho;
    G;
end

figure;

subplot(3,1,1)
plot(omega_vec,sharpe_comp)
title('Robust/Classical Mean Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')

subplot(3,1,2)
plot(omega_vec,ret_comp)
title('Robust/Classical Mean Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')

subplot(3,1,3)
plot(omega_vec,risk_comp)
title('Robust/Classical Mean Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')
% risk-risk2
% ret-ret2
% sharpe-sharpe2
