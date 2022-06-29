clear;clc;

%% Generate synthetic data + plot risk-return
rng(15)
n = 10;
m = 3;
p = 100;
alpha = 0;
rf=3;
mu_dev = 2;

[asset_ret,f] = generate_data(n,m,p,rf,mu_dev);


y=asset_ret;
B=f';
C=[ones(p,1) B'];
reg_result=(C'*C)^(-1)*C'*y; %[mu0, V0_1, ... ,V0_m]' 
mu0=reg_result(1,:)';
V0=reg_result(2:m+1,:);
D=cov(y-C*reg_result);
F=cov(f);
V = (y'/B)';

omega = 0.95;
alpha = 0;
[gamew]= robust_elips(n, m,p,f,asset_ret,alpha,omega);

Gamma = 5;
[gam5w]= robust_poly(n, m,p,f,asset_ret,omega,alpha, Gamma);

Gamma = 1;
[gam1w,~,~,~]= robust_poly(n, m,p,f,asset_ret,omega,alpha, Gamma);

risks = [];
for i = 1:n
    e= eye(n);
    e_i = e(:,i);
    risks = [risks; e_i'*(V0'*F*V0)*e_i ];
end

[risks_sorted,I] = sort(risks);

rel = [gam5w(I) gam1w(I) gamew(I) mu0(I) risks_sorted];
big_5w = [];
big_1w = [];
big_ew = [];

for i=1:n
    ai = rel(i,:);
    if ai(1) >0.01
        big_5w = [big_5w; ai];
    end
    if ai(2) >0.01
        big_1w = [big_1w; ai];
    end

    if ai(3) >0.01
        big_ew = [big_ew; ai];
    end
end

figure
hold on
s=scatter(rel(:,5), rel(:,4),'MarkerFaceColor','k','MarkerEdgeColor','k');
s.SizeData = 10;
b=scatter(big_5w(:,5), big_5w(:,4),'MarkerFaceColor','b','MarkerEdgeColor','b');
b.SizeData = 150;
a=scatter(big_1w(:,5), big_1w(:,4),'MarkerFaceColor','r','MarkerEdgeColor','r');
a.SizeData = 80;
c=scatter(big_ew(:,5), big_ew(:,4),'MarkerFaceColor','g','MarkerEdgeColor','g');
c.SizeData = 30;
xlabel('Nominal Risk')
ylabel('Nomominal Returns')
legend('Unselected assets','\Gamma = 5', '\Gamma = 1', 'Ellipsoidal')
title('Risk-return characteristics of universe of assets')

%% increase Gamma + plot performance

Gamma_vec = 1:1:90;
omega = 0.95;
ret_comp = length(Gamma_vec);
risk_comp = length(Gamma_vec);
sharpe_comp = length(Gamma_vec);

lambda_hat_comp = zeros([m+1 length(Gamma_vec)]);
i = 1;
for Gamma = Gamma_vec
    [w, risk_comp(i), ret_comp(i),sharpe_comp(i), ~,~,~, lambda_hat_comp(:,i)]= robust_poly(n, m,p,f,asset_ret,omega,alpha, Gamma);
    i=i+1;
end


figure
plot(Gamma_vec, lambda_hat_comp)
title('lambda hat for increasing Gamma')
xlabel('Gamma')
ylabel('Sharpe ratio')

figure
subplot(3,1,1)
hold on
plot(Gamma_vec,sharpe_comp)
title('Robust Sharpe Ratio')
xlabel('Gamma')
ylabel('Sharpe ratio')
hold off

subplot(3,1,2)
hold on
plot(Gamma_vec,ret_comp)
title('Robust Return')
xlabel('Gamma')
ylabel('Return')
hold off

subplot(3,1,3)
hold on
plot(Gamma_vec,risk_comp)
title('Robust Risk')
xlabel('Gamma')
ylabel('Risk')
hold off
