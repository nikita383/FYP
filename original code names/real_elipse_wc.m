clear;clc;

asset_ret=readtable('dataset2_monthlyreturns.xlsx','Sheet','%ret');
asset_ret=table2array(asset_ret);
[p, n_assets]=size(asset_ret);
f=readtable('dataset2_monthlyreturns.xlsx','Sheet','f%ret');
f=table2array(f);
[p,m_factors]=size(f);


%---------remove later, for sake of generating F and D -----------
% y=asset_ret;
% B=f';
% A=[ones(p,1) B'];
% reg_result=(A'*A)^(-1)*A'*y; %[mu0, V0_1, ... ,V0_m]' 
% mu0=reg_result(1,:)'*10;
% V0=reg_result(2:m_factors+1,:);
% 
% D=cov(y-A*reg_result);
% F=cov(f);
% alpha=0;
%-----------------------------------------------------------------



%--------sharpe ratio comparison as a function of CL---------------
%omega_vec=0.001:0.005:0.999;
omega_vec=0.01:0.05:0.99;
sz_omega=size(omega_vec);
sz_omega=sz_omega(2);

sharpe_comp=zeros(size(omega_vec));
ret_comp=zeros(size(omega_vec));
risk_comp=zeros(size(omega_vec));

clas_policy=zeros(n_assets,sz_omega);
rob_policy=zeros(n_assets,sz_omega);

alpha=0;

%--------classical optimisation---------
    [w_clas,risk_clas,ret_clas,sharpe_clas]= markowitz(n_assets,p,f,asset_ret,alpha);

for i=1:sz_omega
  
    %--------robust optimisation---------
    [w_rob,risk_rob,ret_rob,sharpe_rob,rho,w_coef,eig_H,D_bar,mu0,gam]= robust2(n_assets,m_factors,p,f,asset_ret,alpha,omega_vec(i));
    
    
    %----------Worst-case----------------
    [risk_wc,ret_wc,sharpe_wc]=wc_classical(n_assets,m_factors,w_clas,rho,w_coef,eig_H,D_bar,mu0,gam);
    
%     %-----wc comparisons----
%     sharpe_comp(i)=sharpe_rob/sharpe_wc;
%     %ret_comp=[ret; ret2]
%     %risk_comp=[risk; risk2]
%     ret_comp(i)=ret_rob/ret_wc;
%     risk_comp(i)=risk_rob/risk_wc;
%        
    
    %-----mean comparisons----
    sharpe_comp(i)=sharpe_rob/sharpe_wc;
    %ret_comp=[ret; ret2]
    %risk_comp=[risk; risk2]
    ret_comp(i)=ret_rob/ret_wc;
    risk_comp(i)=risk_rob/risk_wc;
   
end
figure
%WC plots
% subplot(3,1,1)
% plot(omega_vec,sharpe_comp)
% title('Robust/Classical Worst Case Sharpe Ratio')
% xlabel('Confidence Level')
% ylabel('Ratio of Sharpe ratios')
% 
% subplot(3,1,2)
% plot(omega_vec,ret_comp)
% title('Robust/Classical Worst Case Returns')
% xlabel('Confidence Level')
% ylabel('Ratio of Returns')
% 
% subplot(3,1,3)
% plot(omega_vec,risk_comp)
% title('Robust/Classical Worst Case Risk')
% xlabel('Confidence Level')
% ylabel('Ratio of Risk')



%---------mean plots------------
subplot(3,1,1)
plot(omega_vec,sharpe_comp)
title('Robust/Classical Worst Case Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')

subplot(3,1,2)
plot(omega_vec,ret_comp)
title('Robust/Classical Worst Case Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')

subplot(3,1,3)
plot(omega_vec,risk_comp)
title('Robust/Classical Worst Case Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')