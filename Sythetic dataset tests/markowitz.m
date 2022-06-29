function [w,risk,ret,sharpe]= markowitz(n_assets,p,f,asset_ret,alpha)

y=asset_ret;
B=f';
A=[ones(p,1) B'];
reg_result=(A'*A)^(-1)*A'*y; %[mu0, V0_1, ... ,V0_m]' 
mu_bar=reg_result(1,:)';
D=cov(y-A*reg_result);
F=cov(f);
V = ((A(:,2:end)*reg_result(2:end,:))'/B)';
% function [w,risk,ret,sharpe]= classical_minvar(n_assets,mu,asset_ret,alpha)
% mu_bar=mean(mu,2);


e=ones(n_assets,1);

%CHANGE WHEN CONSIDERING UNCERTAINTY IN D
%S=V'*F*V+D;
%%??
S=cov(asset_ret);
w=quadprog(2*S,zeros(n_assets,1),-mu_bar',-alpha,e',1,zeros(n_assets,1),e);

risk = w'*(V'*F*V)*w;
ret = mu_bar'*w;
sharpe=ret/risk;
end