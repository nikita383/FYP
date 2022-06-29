clear; clc;

%% Generate synthetic data
rng(42)
n = 5;
m = 3;
p = 100;

rf=3;
sz=[n, 1];
mu0=unifrnd(rf-2,rf+2,sz);
mu0 = sort(mu0,'descend');
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

%% Generate test data
f_epsilon_t = generateDataGivenCovar(m+n,p,blkdiag(F,D));

f_t = f_epsilon_t(1:m,:);
epsilon_t = f_epsilon_t((m+1):(n+m),:);

asset_ret_t=mu+V'*f_t+epsilon_t;
f_t=f_t';
asset_ret_t = asset_ret_t';

%% sharpe ratio comparison as a function of CL
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
    w = markowitz(n,p,f,asset_ret,alpha);
    [risk,ret,sharpe] = test_data(m,p,f_t,asset_ret_t, w);
    w3= robust_elips(n, m,p,f,asset_ret,alpha,omega_vec(i));
    [risk3,ret3,sharpe3] = test_data(m,p,f_t,asset_ret_t, w3);
    sharpe_comp_el(i)=sharpe3/sharpe;
    ret_comp_el(i)=ret3/ret;
    risk_comp_el(i)=risk3/risk;
    for j = 1:size(Gamma_vec,2)

        % robust optimisation
        w2= robust_box_poly_2(n, m,p,f,asset_ret,omega_vec(i),alpha, Gamma_vec(j));
        [risk2,ret2,sharpe2] = test_data(m,p,f_t,asset_ret_t, w2);
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