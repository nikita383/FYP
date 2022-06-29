clear;clc;

asset_ret=readtable('dataset_dailyreturns.xlsx','Sheet','%ret');
asset_ret=table2array(asset_ret);

[p, n]=size(asset_ret);

f=readtable('dataset_dailyreturns.xlsx','Sheet','f%ret');
f=table2array(f);

[p,m]=size(f);

date=readtable('dataset_dailyreturns.xlsx','Sheet','asset_ret');
date=table2array(date(1:end-2,1));

M = movmean(asset_ret, 5);
M = mean(M,2);
% figure
% plot(M)

v = movvar(asset_ret, 5);
v = mean(v,2);
% figure
% plot(v)

y=asset_ret;
B=f';
C=[ones(p,1) B'];
reg_result=(C'*C)^(-1)*C'*y; %[mu0, V0_1, ... ,V0_m]' 
mu0=reg_result(1,:)'*10;
V0=reg_result(2:m+1,:);
D=cov(y-C*reg_result);
F=cov(f);
V = (y'/B)';

norm(V'*F*V)
norm(D)


epsilon = y-C*reg_result;
ay = abs(repmat(mu0,1,520)')+abs((V0'*B)')+abs(epsilon);
% eps = mean(epsilon,2);
% eps = movmean(eps,15);
% ay = mean(ay,2);
% ay = movmean(ay,15);
eps_vs_y2 = abs(epsilon./ay);
eps_vs_y = 100*median(eps_vs_y2,2)';
figure
hold on
plot(date,eps_vs_y)
title("Percentage of asset returns explained by \epsilon")
ylabel("Percentage (%)")
xlabel("Date")
ylim([0 100])
% xlim([datetime(['31.05.19'], 'InputFormat', 'dd.MM.yy') , datetime(['25.05.21'], 'InputFormat', 'dd.MM.yy')])


c=corrcoef(mean((V0'*f')',2),mean(epsilon,2));