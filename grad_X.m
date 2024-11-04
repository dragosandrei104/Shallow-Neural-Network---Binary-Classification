function [g_X]=grad_X(e,A,X,x)

[N,n]=size(A);
[~,m]=size(X);
g_X=zeros(n,m);


for k=1:n
    SUM=0;
    for i=1:N
        z=A(i,:)*X;     % z 
        dz=A(i,k);      % z'
        a=sigmoid(z).^2 .* exp(-z);

        dg=a.*dz + dz.*sigmoid(z) + z.*a.*dz - dz.*(sigmoid(z).^2) -...
            z.*2.*sigmoid(z).*a.*dz; % g'

        SUM=SUM+e(i)*(1/(dSiLU(A(i,:),X)*x))*...
            dg.*x' +...%%deriv
        (1-e(i))*(1/(1-dSiLU(A(i,:),X)*x))*...
            dg.*x';
    end
    g_X(k,:)=-SUM./N;
end
end

% dg/dz = ( -z*exp(3*z)+2*exp(3*z)+4*exp(2*z)+2*exp(z)+z*exp(z) ) /
% ((exp(z)+1)^4)