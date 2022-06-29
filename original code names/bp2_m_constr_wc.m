clear;clc;
rng(15)
%% Generate synthetic data
n = 10;
m = 3;
p = 100;
alpha = 0;
rf=3;
mu_dev = 2;

[asset_ret,f] = generate_data(n,m,p,rf,mu_dev);


%% sharpe ratio comparison as a function of CL
%omega_vec=0.001:0.005:0.999;
Gamma_vec = [1,5];
omega_vec=0.001:0.005:0.999;
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
        %--------classical optimisation---------
    [w_clas,risk_clas,ret_clas,sharpe_clas]= markowitz(n,p,f,asset_ret,alpha);
        %--------robust optimisation---------
    [w_rob,risk_rob,ret_rob,sharpe_rob,rho,w_coef,eig_H,D_bar,mu0,gam]= robust_elips(n, m,p,f,asset_ret,alpha,omega_vec(i));
    %-------worst case classical---------------
    [risk_wc,ret_wc,sharpe_wc]=wc_classical(n,m,w_clas,rho,w_coef,eig_H,D_bar,mu0,gam);
    sharpe_comp_el(i)=sharpe_rob/sharpe_wc;
    ret_comp_el(i)=ret_rob/ret_wc;
    risk_comp_el(i)=risk_rob/risk_wc;
    for j = 1:size(Gamma_vec,2)
        %--------robust optimisation---------
        [w2,risk2,ret2,sharpe2]= robust_box_poly_2(n, m,p,f,asset_ret,omega_vec(i),alpha, Gamma_vec(j));
        
        %-----mean comparisons----
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
plot(omega_vec,ones(size(omega_vec)))
title('Worst-Case Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Sharpe ratio')
legend('\Gamma = 1', '\Gamma = 5','Ellipsoidal', 'Classical')
hold off

subplot(3,1,2)
hold on
plot(omega_vec,ret_comp)
plot(omega_vec,ret_comp_el)
plot(omega_vec,ones(size(omega_vec)))
title('Worst-Case Returns')
xlabel('Confidence Level')
ylabel('Returns')
hold off

subplot(3,1,3)
hold on
plot(omega_vec,risk_comp)
plot(omega_vec,risk_comp_el)
plot(omega_vec,ones(size(omega_vec)))
title('Worst-Case Risk')
xlabel('Confidence Level')
ylabel('Risk')
hold off

