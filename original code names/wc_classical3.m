function [risk_classical_wc,wc_ret,sharpe_wc]=wc_classical3(n_assets,m_factors,classical_weights,rho,w_coef,eig_H,D_bar,mu0,gam)
%x=[phi,t,sigma,tau,nu] - (n+m+3) dimensional vector

cone_i= m_factors+1; % m+1 cone constraints


%D1_3
A4=[2*rho',zeros(1,m_factors),0,0,0;zeros(1,n_assets),zeros(1,m_factors), 1,-1,0];
b4=[0;0];
d4=[zeros(1,n_assets),zeros(1,m_factors),1,1,0]';
g4=0;
socConstraints4(cone_i) = secondordercone(A4,b4,d4,g4);
cone_i=cone_i-1;

%D1_4
tmp_t=eye(m_factors);
for i=1:m_factors
    A4= [2*w_coef(m_factors+1-i,:),zeros(1,m_factors),0,0,0; zeros(1,n_assets),-tmp_t(:,m_factors+1-i)', -eig_H(m_factors+1-i),0,0];
    b4=[0;-1];
    d4=[zeros(1,n_assets),tmp_t(:,m_factors+1-i)',-eig_H(m_factors+1-i),0,0]';
    g4=-1;
    socConstraints4(cone_i) = secondordercone(A4,b4,d4,g4);
    cone_i=cone_i-1;
    i=i+1;
end

Aineq4=[zeros(1,n_assets),ones(1,m_factors),0,1,-1; zeros(1,n_assets),zeros(1,m_factors),1,0,0]; %d2
bineq4=[0;1/eig_H(m_factors)];%d2
   

Aeq4=[ones(1,n_assets),zeros(1,m_factors),0,0,0];%socp5
beq4=[1];
% Aeq2=[];
% beq2=[];
    
%upper and lower bounds: w fixed from classical,sigma>0,tau>o,t>0
lb4=[classical_weights;zeros(m_factors,1);0;0;-Inf];
ub4=[classical_weights;Inf*ones(m_factors,1);Inf;Inf;Inf];

%objective function
f4=[zeros(1,n_assets),zeros(1,m_factors),0,0,1]';
[x4,fval4] = coneprog(f4,socConstraints4,Aineq4,bineq4,Aeq4,beq4,lb4,ub4);
%w=x(1:n_assets);
wc_ret=(mu0-gam)'*classical_weights;
risk_classical_wc=(fval4+classical_weights'*D_bar*classical_weights)/10;
sharpe_wc=wc_ret/risk_classical_wc;