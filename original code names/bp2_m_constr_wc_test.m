clear;clc;

%% Generate synthetic data
rng(42)
n = 5;
m = 3;
p = 100;

rf=3;
sz=[n, 1];
mu0=unifrnd(rf-2,rf+2,sz);
mu0 = sort(mu0,'descend');
mu0=sort(mu0);
mu = repmat(mu0, 1, p);

cond_F = 21;
while cond_F >= 20
    F = generateSPDmatrix(m);
    cond_F = cond(F);
%     %cond_F = vpa(cond_F);
end

V = randn(m,n);
V=sort(V,1);

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
alpha=0;
f=f';
asset_ret = asset_ret';


%% sharpe ratio comparison as a function of CL
%omega_vec=0.001:0.005:0.999;
Gamma_vec = [1,5];
omega_vec=0.01:0.05:0.99;
sz_omega=size(omega_vec);
sz_omega=sz_omega(2);

sharpe_comp=zeros(size(Gamma_vec,2),size(omega_vec,2));
ret_comp=zeros(size(Gamma_vec,2),size(omega_vec,2));
risk_comp=zeros(size(Gamma_vec,2),size(omega_vec,2));
sharpe_comp_el=zeros(size(omega_vec));
ret_comp_el=zeros(size(omega_vec));
risk_comp_el=zeros(size(omega_vec));
alpha=0;


for i=1:sz_omega
    w_clas= markowitz(n,p,f,asset_ret,alpha);
    
    [w_rob,~,~,~,rho,w_coef,eig_H,D_bar,mu0,gam]= robust_elips(n, m,p,f,asset_ret,alpha,omega_vec(i));
    [risk_rob,ret_rob,sharpe_rob] = test_data(m,p,f_t,asset_ret_t, w_rob);
    [risk_wc,ret_wc,sharpe_wc]=wc_classical(n,m,w_clas,rho,w_coef,eig_H,D_bar,mu0,gam);
    
    sharpe_comp_el(i)=sharpe_rob/sharpe_wc;
    ret_comp_el(i)=ret_rob/ret_wc;
    risk_comp_el(i)=risk_rob/risk_wc;
    for j = 1:size(Gamma_vec,2)
        [w2,risk2,ret2,sharpe2]= robust_box_poly_2(n, m,p,f,asset_ret,omega_vec(i),alpha, Gamma_vec(j));
        
        sharpe_comp(j,i)=sharpe2/sharpe_wc;
        ret_comp(j,i)=ret2/ret_wc;
        risk_comp(j,i)=risk2/risk_wc;
       
    end
end

%% Plot
figure

subplot(3,1,1)
hold on
plot(omega_vec,sharpe_comp)
plot(omega_vec,sharpe_comp_el)
title('Robust/Classical Worst Case Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')
hold off

subplot(3,1,2)
hold on
plot(omega_vec,ret_comp)
plot(omega_vec,ret_comp_el)
title('Robust/Classical Worst Case Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')
hold off

subplot(3,1,3)
hold on
plot(omega_vec,risk_comp)
plot(omega_vec,risk_comp_el)
title('Robust/Classical Worst Case Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')
hold off