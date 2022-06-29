function A = generateSPDmatrix(n)
%% Generates Symmetric Positive Definite Matrix
    A = rand(n);
    A = 0.5 * (A + A');
    A = A + (n * eye(n));
end