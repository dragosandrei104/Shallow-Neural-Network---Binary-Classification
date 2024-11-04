function [L]=Loss_f(e,A,X,x)
y=dSiLU(A,X)*x;
SUM=0;
[N,~]=size(y);
for i=1:N
    SUM = SUM + e(i)*log(y(i))+(1-e(i))*log(1-y(i));
end
L=(-1/N)*SUM;
end