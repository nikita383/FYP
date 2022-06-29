clear; clc;
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


for i=1:sz_omega
    [w,risk,ret,sharpe]= markowitz(n,p,f,asset_ret,alpha);
    [w3,risk3,ret3,sharpe3]= robust_elips(n, m,p,f,asset_ret,alpha,omega_vec(i));
    sharpe_comp_el(i)=sharpe3/sharpe;
    ret_comp_el(i)=ret3/ret;
    risk_comp_el(i)=risk3/risk;
    for j = 1:size(Gamma_vec,2)

        % robust optimisation
        [w2,risk2,ret2,sharpe2]= robust_box_poly_2(n, m,p,f,asset_ret,omega_vec(i),alpha, Gamma_vec(j));
        
        sharpe_comp(j,i)=sharpe2/sharpe;
        ret_comp(j,i)=ret2/ret;
        risk_comp(j,i)=risk2/risk;
        
    end
end

%% Plot

figure

subplot(3,1,1)
hold on
plot(omega_vec,sharpe_comp)
plot(omega_vec,sharpe_comp_el)
title('Robust/Classical Mean Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')
legend('\Gamma = 1', '\Gamma = 5','Ellipsoidal')
hold off

subplot(3,1,2)
hold on
plot(omega_vec,ret_comp)
plot(omega_vec,ret_comp_el)
title('Robust/Classical Mean Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')
hold off

subplot(3,1,3)
hold on
plot(omega_vec,risk_comp)
plot(omega_vec,risk_comp_el)
title('Robust/Classical Mean Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')
hold off


figure

subplot(3,1,1)
hold on
plot(omega_vec,sharpe_comp)
plot(omega_vec,sharpe_comp_el)
title('Robust/Classical Mean Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')
legend('\Gamma = 1', '\Gamma = 5','Ellipsoidal')
xlim([0.9 1])
hold off

subplot(3,1,2)
hold on
plot(omega_vec,ret_comp)
plot(omega_vec,ret_comp_el)
title('Robust/Classical Mean Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')
xlim([0.9 1])
hold off

subplot(3,1,3)
hold on
plot(omega_vec,risk_comp)
plot(omega_vec,risk_comp_el)
title('Robust/Classical Mean Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')
xlim([0.9 1])
hold off
