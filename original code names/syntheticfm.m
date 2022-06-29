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

omega = 0.95;
alpha = 0;
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

for i=1:sz_omega
    %omega=0.001;
    %--------classical optimisation---------
    [w_clas,risk_clas,ret_clas,sharpe_clas]= markowitz(n,p,f',asset_ret',alpha);
    %[w,risk,ret,sharpe]= classical_minvar(n,p,f',asset_ret',alpha);
    %clas_policy=w
%     ret_vec(i)=ret;
    %--------robust optimisation---------
    [w_rob,risk_rob,ret_rob,sharpe_rob,rho,w_coef,eig_H,D_bar,mu0,gam]= robust2(n, m,p,f',asset_ret',alpha,omega_vec(i));
    
    %-------worst case classical---------------
%     %equally weighted worst case
%     w_clas=(1/n)*ones(n,1);
    [risk_wc,ret_wc,sharpe_wc]=wc_classical(n,m,w_clas,rho,w_coef,eig_H,D_bar,mu0,gam);
    

    sharpe_comp_rob(i)=sharpe_rob/sharpe_wc;
    %ret_comp=[ret; ret2]
    %risk_comp=[risk; risk2]
    ret_comp_rob(i)=ret_rob/ret_wc;
    risk_comp_rob(i)=risk_rob/risk_wc;
    sharpe_comp_clas(i)=sharpe_clas/sharpe_wc;
    %ret_comp=[ret; ret2]
    %risk_comp=[risk; risk2]
    ret_comp_clas(i)=ret_clas/ret_wc;
    risk_comp_clas(i)=risk_clas/risk_wc;
end

figure;
%WC plots
subplot(3,1,1)
plot(omega_vec,sharpe_comp_rob)
title('Robust/Classical Worst Case Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')

subplot(3,1,2)
plot(omega_vec,ret_comp_rob)
title('Robust/Classical Worst Case Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')

subplot(3,1,3)
plot(omega_vec,risk_comp_rob)
title('Robust/Classical Worst Case Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')

