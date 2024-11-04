% proiect optimizari
% functie dSiLU ----> g(z) = σ(z)*(1 + z*(1-σ(z))
clc
clear
%% Citirea si preprocesare datelor
dataFolder = 'train';
categories = {'Marble', 'Quartz'};

imds = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

dataFolder = 'test';
imds_t = imageDatastore(fullfile(dataFolder, categories), 'LabelSource', 'foldernames', 'IncludeSubfolders', true);

data=[];                            % date de antrenare
data_t =[];                         % date de test
while hasdata(imds) 
    img = read(imds) ;              % citeste o imagine din datastore
    img = imresize(img, [227 227]);
    %figure, imshow(img); % decomentati pentru a vizualiza pozele din
    %pause                        %baza de dat 
    img = double(im2gray(img));
    data =[data, reshape(img, [], 1)];

end
eticheta = double(imds.Labels == 'Quartz'); % eticheta 1 - Quartz, 0 - Marble
while hasdata(imds_t) 
    img = read(imds_t) ;              % citeste o imagine din datastore
    img = imresize(img, [227 227]); 
    %figure, imshow(img);    % decomentati pentru a vizualiza pozele din
    % pause                  %baza de date
    img = double(rgb2gray(img));
    data_t =[data_t, reshape(img, [], 1)];
    
end
eticheta_t = double(imds_t.Labels == 'Quartz'); % eticheta 1 - Quartz, 0 - Marble
%Reducerea dimensiuni
data_pca = pca(data, 'NumComponents', 45)';
data_pca_t = pca(data_t, 'NumComponents', 45)';

clear categories imds imds_t img dataFolder data data_t

[n, N] = size(data_pca);
%%

A=data_pca';
m=12;
A_1=[data_pca',ones(N,1)];
A_copy=A_1;
perm=randperm(N);
A_shuffled=A_copy(perm,:);
eticheta_shuffled=eticheta(perm);
A_T=[data_pca_t',ones(75,1)];

x=randn(m,1)/10;
%x_copy=x;
X=randn(n+1,m)/10;
%X_copy=X;

iter=0;
maxIter=1000;

L=Loss_f(eticheta_shuffled,A_shuffled,X,x);
crit=0;

epsilon=1e-3;

evol_crit=[];
evol_L=[];
evol_timp=[];
evol_timp_L=[];

%timp=tic();

while abs(L-crit)>=epsilon && iter<=maxIter
    alfa_X = 1/max(eig(X'*X));
    alfa_x = 1/max(eig(x'*x));

    %evol_L=[evol_L,L];
    evol_crit=[evol_crit,abs(L-crit)];
    crit=L;
    
    g_X=grad_X(eticheta,A_1,X,x);
    g_x=gradi_x(eticheta,A_1,X,x);

    X = X + alfa_X * g_X;
    x = x + alfa_x * g_x;

    %timpL=tic();
    L=abs(Loss_f(eticheta,A_1,X,x));
    %evol_timp_L=[evol_timp_L,toc(timpL)];

    %evol_timp=[evol_timp,toc(timp)];
    iter
    iter=iter+1;
end

%pentru metoda grad. stocastic
%{
while abs(L-crit)>=epsilon && iter<=maxIter
    alfa_X = 1/max(eig(X'*X));
    alfa_x = 1/max(eig(x'*x));

    %evol_L=[evol_L,L];
    evol_crit=[evol_crit,abs(L-crit)];
    crit=L;
    
    g_X=grad_X(eticheta_shuffled,A_shuffled,X,x);
    g_x=gradi_x(eticheta_shuffled,A_shuffled,X,x);

    X = X + alfa_X * g_X;
    x = x + alfa_x * g_x;

    %timpL=tic();
    L=abs(Loss_f(eticheta_shuffled,A_shuffled,X,x));
    %evol_timp_L=[evol_timp_L,toc(timpL)];

    %evol_timp=[evol_timp,toc(timp)];
    iter
    iter=iter+1;
end
%}
predict=rescale(dSiLU(A_T,X)*x);

for i=1:75
    if(predict(i)<0.5)
        predict(i)=0;
    else
        predict(i)=1;
    end
end


C=confusionmat(eticheta_t, predict)

TP=C(1,1);
FN=C(1,2);
FP=C(2,1);
TN=C(2,2);


precision=TP/(TP+FP)
recall=TP/(TP+FN)
f1Score=2*(precision*recall)/(precision+recall)


interval=1:iter;
figure()
fig1=semilogy(interval,evol_crit(1:iter))
grid on
%hold on
%fig2=semilogy(interval,evol_L(1:iter))
%hold on


