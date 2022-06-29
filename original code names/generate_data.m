function [asset_ret,f] = generate_data(n,m,p,rf,mu_dev)

tradeoff = false;

while ~tradeoff
    sz=[n, 1];
    mu0=unifrnd(rf-mu_dev,rf+mu_dev,sz);
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
    
    risks = zeros(n,1);
    for i = 1:n
        e= eye(n);
        e_i = e(:,i);
        risks(i) = [e_i'*V'*F*V*e_i];
    end
    
    [risks,I] = sort(risks);
    
    tradeoff = check_ascending(mu0(I),risks);
    
end

asset_ret=mu+V'*f+epsilon;
f=f';
asset_ret = asset_ret';
end