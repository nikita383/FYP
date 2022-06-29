% clear; clc;
rng(5)
s = [2 2];
r_gen = randn(334,1);
r1 = normrnd(s(1).*r_gen,1);
r2 = normrnd(s(2).*r_gen,1);
% y = [r1 r2];
y=data;
M = cov(y);%2x2
% Calculate the eigenvectors and eigenvalues
[eigenvec, eigenval ] = eig(M);

% Get the index of the largest eigenvector
[largest_eigenvec_ind_c, r] = find(eigenval == max(max(eigenval)));
largest_eigenvec = eigenvec(:, largest_eigenvec_ind_c);
% Get the largest eigenvalue
largest_eigenval = max(max(eigenval));
% Get the smallest eigenvector and eigenvalue
if (largest_eigenvec_ind_c == 1)
    smallest_eigenval = max(eigenval(:,2));
    smallest_eigenvec = eigenvec(:,2);
    else
    smallest_eigenval = max(eigenval(:,1));
    smallest_eigenvec = eigenvec(1,:);
end

% Calculate the angle between the x−axis and the largest eigenvector
angle = atan2(largest_eigenvec(2), largest_eigenvec(1));
if(angle < 0)
angle = angle + 2*pi;
end

% Get the 95% confidence interval error
chisquare_val = 2.4477;
phi = angle;
a=chisquare_val*sqrt(largest_eigenval);
b=chisquare_val*sqrt(smallest_eigenval);

%Rotation
R = [ cos(phi) sin(phi); - sin(phi) cos(phi) ];

figure
hold on

%% Intersection norm
colour = ['k', 'g', 'm', 'b'];
ltype = ["-"; ":"; ":"; "-"];
lw = [1 1 1.5 1];
c=1;
for Gamma = [2, 1.5, 1.3, 1]
    [r_u ,r_d] = polytope_intersection(R, a, b, Gamma);
    i_1(c) = plot( r_u(:,1), r_u(:,2), 'Color', colour(c),'LineStyle', ltype(c),LineWidth =lw(c));
    plot( r_d(:,1), r_d(:,2), 'Color', colour(c),'LineStyle', ltype(c), LineWidth=lw(c));
    c = c+1;
end

%% elipse
% theta_grid = 0:(2*pi/(length(r_gen)-1)) :2*pi;
% ellipse_x_r = a*cos( theta_grid );
% ellipse_y_r = b*sin( theta_grid );
% r_ellipse = [ellipse_x_r;ellipse_y_r]' * R;
% e = plot( r_ellipse(:,1), r_ellipse(:,2), 'Color',[0.4660 0.6740 0.1880], LineWidth=1);

%% data
d= scatter(y(:,1), y(:,2), 'filled');
d.SizeData = 2;
d.AlphaData = 0.75;
mindata = min(min(y));
maxdata = max(max(y));
xlim([(mindata ) (maxdata)]);
ylim([(mindata ) (maxdata)]);

%% Legend
hleglines = [];
for c = [1,2,3,4]
    hleglines = [hleglines i_1(c)];
end
% hleglines = [hleglines e d];
% create the legend
hleg = legend(hleglines,'\Gamma = 2','\Gamma = 1.5','\Gamma = 1.3','\Gamma = 1', 'Data');
legend()
xlabel('x_1')
ylabel('x_2')
title ('Distribution of arbitrary data and the corresponding 95% confidence polytopes')
