function [g] = dSiLU(A,X)
% calcul σ(z)
% g(z) = σ(z)*(1 + z*(1-σ(z)))
[N,n]=size(A);
[~,m]=size(X);
g=zeros(N,m);
I=ones(n,1);
for i=1:N
    for j=1:m
        z=A(i,:)*X(:,j);
        g(i,j) = sigmoid(z)*(1 + z*(1 - sigmoid(z)));
    end
end
            
end