clear;clc;
% 
n = 15;
m_vec = 2:1:14;
p = 100;
alpha = 0;
rf=3;
mu_dev = 2;
omega = 0.95;

%% risk test
Gamma = 5;
ret_comp = length(m_vec);
risk_comp = zeros([2 length(m_vec)]);
sharpe_comp = length(m_vec);
time_comp = zeros([2 length(m_vec)]);
i = 1;
for m = m_vec
    [asset_ret,f] = generate_data(n,m,p,rf,mu_dev);
    tic
    [w, risk_comp(1,i)]= robust_elips(n, m,p,f,asset_ret,omega,alpha);
    time_comp(1,i)=toc;
    tic
    [w, risk_comp(2,i), ret_comp(i),sharpe_comp(i)]= robust_poly(n, m,p,f,asset_ret,omega,alpha, Gamma);
    time_comp(2,i)=toc;
    i=i+1;
end

figure
hold on
plot(m_vec,risk_comp)
title('Portfolio Risk as m increases')
xlabel('m')
ylabel('Risk')
legend('Ellipsoidal', 'Polyhedral, \Gamma = 5')
hold off

%% Computation time test
m = 3;
n_vec = 5:1:20;
p = 100;
alpha = 0;
rf=3;
mu_dev = 2;
omega = 0.95;

Gamma = 5;
risk_comp = zeros([2 length(n_vec)]);
time_comp = zeros([2 length(n_vec)]);
i = 1;
for n = n_vec
    [asset_ret,f] = generate_data(n,m,p,rf,mu_dev);
    tic
    [w, risk_comp(1,i)] = robust_elips(n, m,p,f,asset_ret,omega,alpha);
    time_comp(1,i)=toc;
    tic
    [w, risk_comp(2,i), ret_comp(i),sharpe_comp(i)]= robust_poly(n, m,p,f,asset_ret,omega,alpha, Gamma);
    time_comp(2,i)=toc;
    i=i+1;
end

figure
hold on
plot(n_vec,time_comp)
title('Computation time for increasing n')
xlabel('n')
ylabel('Time (s)')
hold off