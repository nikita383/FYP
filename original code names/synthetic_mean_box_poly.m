clear; clc;

%% Generate synthetic data
rng(42)
n = 5;
m = 3;
p = 1000;

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

V = randn(m,n);

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

%% sharpe ratio comparison as a function of CL
omega_vec=0.01:0.025:0.99;
sz_omega=size(omega_vec);
sz_omega=sz_omega(2);

sharpe_comp=zeros(size(omega_vec));
ret_comp=zeros(size(omega_vec));
risk_comp=zeros(size(omega_vec));

for i=1:sz_omega
    % classical optimisation
    [w,risk,ret,sharpe]= markowitz(n,p,V,F,f,D,asset_ret,alpha);
    
    % robust optimisation
    [w2,risk2,ret2,sharpe2]= robust_box_poly(n, m,p,V,F,f,D,asset_ret,omega_vec(i),alpha);
    
    sharpe_comp(i)=sharpe2;%/sharpe;
    ret_comp(i)=ret2;%/ret;
    risk_comp(i)=w2'*V'*F*V*w2;%;%/risk;

end

%% Plot

figure

subplot(3,1,1)
plot(omega_vec,sharpe_comp)
title('Robust/Classical Mean Sharpe Ratio')
xlabel('Confidence Level')
ylabel('Ratio of Sharpe ratios')

subplot(3,1,2)
plot(omega_vec,ret_comp)
title('Robust/Classical Mean Returns')
xlabel('Confidence Level')
ylabel('Ratio of Returns')

subplot(3,1,3)
plot(omega_vec,risk_comp)
title('Robust/Classical Mean Risk')
xlabel('Confidence Level')
ylabel('Ratio of Risk')
