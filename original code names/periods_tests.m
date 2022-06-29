clear;clc;

full_asset_ret=readtable('dataset_dailyreturns.xlsx','Sheet','%ret');
full_asset_ret=table2array(full_asset_ret);

[p, n]=size(full_asset_ret);

full_f=readtable('dataset_dailyreturns.xlsx','Sheet','f%ret');
full_f=table2array(full_f);

[p,m]=size(full_f);

alpha = 0;

% %% sharpe ratio comparison as a function of CL
Gamma_vec = [5];
omega_vec=0.95;

periods = 5;
sharpe_comp=zeros(size(Gamma_vec,2),5);
ret_comp=zeros(size(Gamma_vec,2),5);
risk_comp=zeros(size(Gamma_vec,2),5);
sharpe_comp_el=zeros([1 5]);
ret_comp_el=zeros([1 5]);
risk_comp_el=zeros([1 5]);


for i=1:periods
    asset_ret = full_asset_ret(104*(i-1)+1:104*i,:);
    f = full_f(104*(i-1)+1:104*i,:);
    p = 520/periods;
    [w,risk,ret,sharpe]= markowitz_rl(n,p,f,asset_ret,alpha);
    [w3,risk3,ret3,sharpe3]= robust2(n, m,p,f,asset_ret,alpha,omega_vec);
    sharpe_comp_el(i)=sharpe3/sharpe;
    ret_comp_el(i)=ret3/ret;
    risk_comp_el(i)=risk3/risk;
    for j = 1:size(Gamma_vec,2)

        % robust optimisation
        [w2,risk2,ret2,sharpe2]= robust_box_poly_rl_2(n, m,p,f,asset_ret,omega_vec,alpha, Gamma_vec(j));
        
        sharpe_comp(j,i)=sharpe2/sharpe;
        ret_comp(j,i)=ret2/ret;
        risk_comp(j,i)=risk2/risk;
        
    end
end

%% Plot

figure

subplot(3,1,1)
hold on
plot(1:1:5,sharpe_comp)
plot(1:1:5,sharpe_comp_el)
title('Robust/Classical Mean Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')
hold off

subplot(3,1,2)
hold on
plot(1:1:5,ret_comp)
plot(1:1:5,ret_comp_el)
ylim([-1 5])
title('Robust/Classical Mean Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')
hold off

subplot(3,1,3)
hold on
plot(1:1:5,risk_comp)
plot(1:1:5,risk_comp_el)
title('Robust/Classical Mean Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')
hold off
