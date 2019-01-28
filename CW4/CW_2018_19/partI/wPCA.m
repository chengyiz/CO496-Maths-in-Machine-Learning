% to be completed
function U_reduc = wPCA(fea_Train, N)
fea_Train = fea_Train.';
%fea_Train = fea_Train(1:end, 1:end-1);
X_bar = fea_Train * (eye(N+1) - ones(N+1,1) * ones(1,N+1) / (N+1));
k = X_bar.' * X_bar;
K = (k + k.') /2;
[V, D] = eig(K);
V = V(1:end, 2:end);
D = D(2:end, 2:end);
for i=1:N
    D(i,i) = abs(D(i,i) ^ -1);
end
U_reduc = X_bar * V * D;
U_reduc = fliplr(U_reduc);
%U_reduc(1:10,1:10);


end
