clear;clc;

asset_ret=readtable('dataset_dailyreturns.xlsx','Sheet','%ret');
asset_ret=table2array(asset_ret);

[p, n]=size(asset_ret);

f=readtable('dataset_dailyreturns.xlsx','Sheet','f%ret');
f=table2array(f);

[p,m]=size(f);

% asset_ret=readtable('dataset2_monthlyreturns.xlsx','Sheet','%ret');
% asset_ret=table2array(asset_ret);
% [p, n]=size(asset_ret);
% f=readtable('dataset2_monthlyreturns.xlsx','Sheet','f%ret');
% f=table2array(f);
% [p,m]=size(f);
alpha = 0;

% %% sharpe ratio comparison as a function of CL
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


for i=1:sz_omega
    [w,risk,ret,sharpe]= markowitz_rl(n,p,f,asset_ret,alpha);
    [w3,risk3,ret3,sharpe3]= robust_elips_rl(n, m,p,f,asset_ret,alpha,omega_vec(i));
    sharpe_comp_el(i)=sharpe3/sharpe;
    ret_comp_el(i)=ret3/ret;
    risk_comp_el(i)=risk3/risk;
    for j = 1:size(Gamma_vec,2)

        % robust optimisation
        [w2,risk2,ret2,sharpe2]= robust_poly_rl(n, m,p,f,asset_ret,omega_vec(i),alpha, Gamma_vec(j));
        
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
plot(omega_vec,ones(size(omega_vec)))
title('Mean Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Sharpe ratio')
hold off

subplot(3,1,2)
hold on
plot(omega_vec,ret_comp)
plot(omega_vec,ret_comp_el)
plot(omega_vec,ones(size(omega_vec)))
title('Mean Returns')
xlabel('Confidence Level')
ylabel('Returns')
hold off

subplot(3,1,3)
hold on
plot(omega_vec,risk_comp)
plot(omega_vec,risk_comp_el)
plot(omega_vec,ones(size(omega_vec)))
title('Mean Risk')
xlabel('Confidence Level')
ylabel('Risk')
legend('\Gamma = 1', '\Gamma = 5','Ellipsoidal', 'Classical')
hold off

