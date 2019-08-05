% to be completed
% function [W,d,Q] = LDA(fea_Train,gnd_Train)
function W = LDA(fea_Train, gnd_Train)
N = size(fea_Train, 1);
C = max(gnd_Train)-min(gnd_Train) + 1;
Nc = zeros(1, C);
M = zeros(1);

for i=1:C
    fea_i = fea_Train(gnd_Train==i, 1:end).'; % f * N(i)
    Nc(i) = size(fea_i, 2);
    M = blkdiag(M, ones(Nc(i))/Nc(i));
end
M = M(2:end, 2:end);
mat = (eye(N)-M)*fea_Train;
[Vw, Dw] = eig(mat * mat.');
for i=1:N
    Dw(i,i) = abs(Dw(i,i)^-1);
end
% Vw = fliplr(Vw);
% Dw = diag(fliplr(diag(Dw)));
d= diag(Dw);
U = mat.' * Vw(1:end, 1+C:end) * Dw(1+C:end, 1+C:end);
Xb = U.' * fea_Train.' * M;
[Q, Db] = eig(Xb * Xb.');
Q = Q(1:end, end-C+2:end);
Q = fliplr(Q);
W = U * Q;
end