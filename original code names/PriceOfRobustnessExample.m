% ROME Example: Simple Portflio Example from "The Price of Robustess"
%
% See: price_of_robustness_example.pdf
%
% Modified by:
% 1. Melvyn   (Created 10 Sep 2009)

% Display welcome message
disp('Simple Portfolio Example');

% Paramter setup
n  = 150;                       % Number of stocks 
mu  = 1.15+ 0.05/150*(1:n)';    % Mean return
sigma = 0.05/450*sqrt(2*n*(n+1)*(1:n)'); % Deviation
Gamma = 4;  

% begin rome
h = rome_begin;   

% Bertsimas and Sim's uncertain set 
newvar z(n) uncertain;
rome_box(z, -1, 1);    
rome_constraint(norm1(z)<=Gamma);   % Budget of uncertainty approach
%rome_constraint(norm2(z)<=Gamma);  % Ben-Tal and Nemirovski approach


% Returns relations with uncertain factors
r = mu + sigma.*z;

% Portfolo weights
newvar x(n) nonneg;

% Objective
rome_maximize(r'*x);   % Note that r is uncertain and depends in z.     
                       % The objective should be interpreted as 
                       % max{y : y<= r(z)'x for z in uncertainty set}
                       
rome_constraint(sum(x)==1); 

% solve
h.solve;
obj = h.objective;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
xx   = h.eval(x);

rome_end;

disp('Budget of Uncertainty')
disp(Gamma);
disp('Objective')
disp(obj);
disp('Expected return:');
disp(xx'*mu);



% ROME: Copyright (C) 2009 by Joel Goh and Melvyn Sim
% See the file COPYING.txt for full copyright information.