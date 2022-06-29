function [risk,ret, sharpe] = test_data(m,p,f,asset_ret, w)
y=asset_ret;
B=f';
C=[ones(p,1) B'];
reg_result=(C'*C)^(-1)*C'*y; %[mu0, V0_1, ... ,V0_m]' 
mu0=reg_result(1,:)';
V0=reg_result(2:m+1,:);
D=cov(y-C*reg_result);
F=cov(f);
V = (y'/B)';

risk = w'*(V'*F*V)*w+w'*D*w;
ret=mu0'*w;
sharpe=ret/risk;
end