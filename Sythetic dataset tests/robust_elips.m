%asset_ret=mu+V'*f+epsilon; calulated outside function
%hence, V epsilon and mu not input paramters
% inc F

function [w,risk,ret,sharpe,rho,w_coef,eig_H,D_bar,mu0,gam,fval] = robust_elips(n_assets, m_factors,p,f,asset_ret,alpha,omega)

%--------setting uncertainty paramaters-----------------------------------------

%(mu0,V0)-least squares estimate of mu over all periods
y=asset_ret;
B=f';
A=[ones(p,1) B'];
reg_result=(A'*A)^(-1)*A'*y; %[mu0, V0_1, ... ,V0_m]' 
mu0=reg_result(1,:)';
V0=reg_result(2:m_factors+1,:);
D=cov(y-A*reg_result);
V = ((A(:,2:end)*reg_result(2:end,:))'/B)';
F=cov(f);

%unbiased estimate of variance - used for rho and gam
s2=zeros(n_assets,1);
for i=1:n_assets
    s2(i)=(1/(p-m_factors-1))*(norm(y(:,i)-A*reg_result(:,i)))^2;
end

%critical value for f-dist
c_w=finv(omega,m_factors+1,p-m_factors-1);%critical value

% G- norm selection for elliptical norm
G=B*B'-(1/p)*(B*ones(p,1))*(B*ones(p,1))';

% rho - norm bound
rho = zeros(n_assets,1);%+0.000001*ones(n_assets,1);
for i=1:n_assets
    rho(i)=sqrt((m_factors+1)*c_w*s2(i));
end

% gam - mu bound
gam=zeros(n_assets,1);
tmp=(A'*A)^(-1);
tmp=tmp(1,1);
for i=1:n_assets
    gam(i)=sqrt((m_factors+1)*tmp*c_w*s2(i));
end


% eta - F bound

% F_ml=F_0- nominal covar matrix
F_0=(1/(p-1))*(B*B'-(1/p)*(B*ones(p,1))*(B*ones(p,1))');
single_omega=omega^(1/m_factors);

% brute force to find value of eta 2 d.p.
x = 0:0.01:2;
y1 = gamcdf(x,(p+1)/2,2/(p-1));
gamval=[x;y1]';
%gamval=array2table(gamval);
y2=zeros(1,101);
for i=1:100
    y2(i+1)=y1(101+i)-y1(101-i);
    if y2(i+1)>=single_omega
        eta=0.01*i;
        break
    end      
end


%definition 1 constraint parameters
H=G^(-1/2)*F_0*G^(-1/2);
[Q,L] = qdwheig(H); 

% uncertainty in residual risk covar
%CHANGE LATER
D_bar=D;

%--------------optimisation---------------------------------
%x=[phi,psi,t,sigma,tau,nu,delta] - (2n+m+4) dimensional vector
cone_i= m_factors+2; % m+2 cone constraints
%cons_dim=2*n_assets+m_factors+4;

%definition constraints
w_coef=Q'*H^(1/2)*G^(1/2)*V0;
w_coef=real(w_coef);
eig_H=diag(L);

%D1_3
A=[zeros(1,n_assets),2*rho',zeros(1,m_factors),0,0,0,0;
   zeros(1,n_assets),zeros(1,n_assets),zeros(1,m_factors), 1,-1,0,0];
b=[0;0];
d=[zeros(1,n_assets),zeros(1,n_assets),zeros(1,m_factors),1,1,0,0]';
g=0;
socConstraints(cone_i) = secondordercone(A,b,d,g);
cone_i=cone_i-1;

%D1_4
tmp_t=eye(m_factors);
for i=1:m_factors
    
    A= [2*w_coef(m_factors+1-i,:),zeros(1,n_assets),zeros(1,m_factors),0,0,0,0;
   zeros(1,n_assets),zeros(1,n_assets),-tmp_t(:,m_factors+1-i)', -eig_H(m_factors+1-i),0,0,0];
    b=[0;-1];
    d=[zeros(1,n_assets),zeros(1,n_assets),tmp_t(:,m_factors+1-i)',-eig_H(m_factors+1-i),0,0,0]';
    g=-1;
    socConstraints(cone_i) = secondordercone(A,b,d,g);
    cone_i=cone_i-1;
    i=i+1;
end

%SOCP_1
A=[2*D_bar^(1/2),zeros(n_assets),zeros(n_assets,m_factors),zeros(n_assets,4);
    zeros(1,2*n_assets+m_factors+3),-1];
b=[zeros(n_assets,1);-1];
d=[zeros(1,n_assets),zeros(1,n_assets),zeros(1,m_factors),0,0,0,1]';
g=-1;
socConstraints(cone_i) = secondordercone(A,b,d,g);

%d1
A=[zeros(1,n_assets),zeros(1,n_assets),ones(1,m_factors),0,1,-(1-eta),0;%d1
   zeros(1,n_assets),zeros(1,n_assets),zeros(1,m_factors),1,0,0,0; %d2
   -mu0',gam',zeros(1,m_factors),0,0,0,0;%socp2
   eye(n_assets),-eye(n_assets),zeros(n_assets,m_factors),zeros(n_assets,4);%socp3
   -eye(n_assets),-eye(n_assets),zeros(n_assets,m_factors),zeros(n_assets,4);%socp4
    ];
b=[0;%d1
   1/eig_H(m_factors);%d2
   -alpha;%socp2
   zeros(n_assets,1);%socp3
   zeros(n_assets,1);%socp4
   ];

Aeq=[ones(1,n_assets),zeros(1,n_assets),zeros(1,m_factors),0,0,0,0];%socp5
beq=[1];

%upper and lower bounds: 0<w<1,sigma>0,tau>o,t>0
lb=[zeros(n_assets,1);-Inf*ones(n_assets,1);zeros(m_factors,1);0;0;-Inf;-Inf];
ub=[ones(n_assets,1);Inf*ones(n_assets,1);Inf*ones(m_factors,1);Inf;Inf;Inf;Inf];

%objective function
f=[zeros(1,n_assets),zeros(1,n_assets),zeros(1,m_factors),0,0,1,1]';
[x,fval] = coneprog(f,socConstraints,A,b,Aeq,beq,lb,ub);

w=x(1:n_assets);

risk =  w'*(V'*F*V)*w;
ret=(mu0-gam)'*w;
sharpe=ret/risk;

end