function [g_x]=gradi_x(e,A,X,x)

[N,n]=size(A);
[~,m]=size(X);
g_x=zeros(m,1);

for k=1:m
    SUM=0;
    for i=1:N
        SUM=SUM+e(i)*(1/(dSiLU(A(i,:),X)*x))*dSiLU(A(i,:),X(:,k))...
            +(1-e(i))*(1/(1-dSiLU(A(i,:),X)*x))...
            *dSiLU(A(i,:),X(:,k));
    end
    g_x(k)=-SUM/N;
end