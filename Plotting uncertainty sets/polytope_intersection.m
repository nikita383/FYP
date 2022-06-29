function [r_upper,r_lower] = polytope_intersection(R,a,b,Gamma)
L_upper = [(ones(1,167) .*max(-a,linspace(-Gamma*a,0,167))) (ones(1,167) .*min(a,linspace(0,Gamma*a,167)));...
    (ones(1,167) .*min(b,linspace(0,Gamma*b,167))) (ones(1,167) .*min(b,linspace(Gamma*b,0,167)))];
L_lower = [(ones(1,167) .*max(-a,linspace(-Gamma*a,0,167))) (ones(1,167) .*min(a,linspace(0,Gamma*a,167)));...
    (ones(1,167) .*max(-b,linspace(0,-Gamma*b,167))) (ones(1,167) .*max(-b,linspace(-Gamma*b,0,167)))];
r_upper = L_upper' * R;
r_lower = L_lower' * R;
end
