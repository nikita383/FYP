
% %% Generate synthetic data
% 
% %rng(42)
% n = 5;
% m = 3;
% p = 100;
% 
% rf=3;
% sz=[n, 1];
% mu0=unifrnd(rf-2,rf+2,sz);
% mu = repmat(mu0, 1, p);
% 
% cond_F = 21;
% while cond_F >= 20
%     F = generateSPDmatrix(m);
%     cond_F = cond(F);
%     %cond_F = vpa(cond_F);
% end
% 
% V = randn(m,n);
% 
% D_diag_norm = norm(V'*F*V);
% while D_diag_norm > 0.1*norm(V'*F*V)
%     D_diag = rand(sz);
%     D_diag_norm = norm(D_diag);
% end
% D = diag(D_diag);
% 
% f_epsilon = generateDataGivenCovar(m+n,p,blkdiag(F,D));
% 
% f = f_epsilon(1:m,:);
% epsilon = f_epsilon((m+1):(n+m),:);
% 
% asset_ret=mu+V'*f+epsilon;
% 
% f=f';
% asset_ret = asset_ret';

asset_ret=readtable('dataset_dailyreturns.xlsx','Sheet','%ret');
asset_ret=table2array(asset_ret);

[p, n]=size(asset_ret);

f=readtable('dataset_dailyreturns.xlsx','Sheet','f%ret');
f=table2array(f);

[p,m]=size(f);

omega = 0.95;
alpha = 0;
Gamma = 5;
[w3,risk3,ret3,sharpe3]= robust2(n, m,p,f,asset_ret,alpha,omega);
[w,risk,ret,sharpe]= robust_box_poly_rl_2(n, m,p,f,asset_ret,omega,alpha, Gamma);
w
sum(w)
risk
ret