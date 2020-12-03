function z_k = LASSOADMM(A,b,Lambda,MaxIter,Rho,Alpha)
% A: d-by-n
% b: d-by-m
% 
% https://statweb.stanford.edu/~candes/math301/Lectures/Consensus.pdf
% http://www.mathworks.com/help/stats/lasso.html#bs25w54-6
if ~exist('Lambda','var')
    Lambda = 1e-3;
end
if ~exist('MaxIter','var')
    MaxIter = 50;
end
if ~exist('Rho','var')
    Rho = 1;
end
if ~exist('Alpha','var')
    Alpha = 1;
end
kappa=Lambda/Rho;
x_k=zeros(size(A,2),size(b,2));
z_k=x_k;
u_k=z_k;
tmp=eye(size(A,2))/(A'*A+(Rho+Lambda*(1-Alpha))*eye(size(A,2)));
tmp2=tmp*A'*b;
for k =1:MaxIter
    x_k=tmp2+tmp*Rho*(z_k-u_k);
    z_k=max(x_k+u_k-kappa,0)-max(-x_k-u_k-kappa,0);
    u_k=u_k+Rho*(x_k-z_k);%u_k+x_k-z_k;%
end


