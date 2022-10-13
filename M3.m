clc;  clear;

syms x1 x2 x3 x4 x5 x6 x7 x8 x9 x10

c = [-6.089 -17.164 -34.054 -5.914 -24.721 -14.986 -24.1 -10.708 -26.662 -22.179];

Sum = @(x)x(:,1).^2+x(:,2).^2+x(:,3).^2+x(:,4).^2+x(:,5).^2+x(:,6).^2+x(:,7).^2+x(:,8).^2+x(:,9).^2+x(:,10).^2;
g = @(x)x(:,1).*(c(1)+log(x(:,1).^2./(Sum(x))))+x(:,2).*(c(2)+log(x(:,2).^2./(Sum(x))))+x(:,3).*(c(3)+log(x(:,3).^2./(Sum(x))))+x(:,4).*(c(4)+log(x(:,4).^2./(Sum(x))))+x(:,5).*(c(5)+log(x(:,5).^2./(Sum(x))))+x(:,6).*(c(6)+log(x(:,6).^2./(Sum(x))))+x(:,7).*(c(7)+log(x(:,7).^2./(Sum(x))))+x(:,8).*(c(8)+log(x(:,8).^2./(Sum(x))))+x(:,9).*(c(9)+log(x(:,9).^2./(Sum(x))))+x(:,10).*(c(10)+log(x(:,10).^2./(Sum(x))));
  
Sum1 = x1.^2+x2.^2+x3.^2+x4.^2+x5.^2+x6.^2+x7.^2+x8.^2+x9.^2+x10.^2;
G = x1.*(c(1)+log(x1.^2./(Sum1)))+x2.*(c(2)+log(x2.^2./(Sum1)))+x3.*(c(3)+log(x3.^2./(Sum1)))+x4.*(c(4)+log(x4.^2./(Sum1)))+x5.*(c(5)+log(x5.^2./(Sum1)))+x6.*(c(6)+log(x6.^2./(Sum1)))+x7.*(c(7)+log(x7.^2./(Sum1)))+x8.*(c(8)+log(x8.^2./(Sum1)))+x9.*(c(9)+log(x9.^2./(Sum1)))+x10.*(c(10)+log(x10.^2./(Sum1)));

grad_f = [diff(G,x1),diff(G,x2),diff(G,x3),diff(G,x4),diff(G,x5),diff(G,x6),diff(G,x7),diff(G,x8) ,diff(G,x9),diff(G,x10) ];

% Partial derivative function
for i = 1:10
   Pd{i} = matlabFunction(grad_f(i));
end
 
lb = ones(1,10);     % Lower bound of input parameter
ub = 10.*ones(1,10); % Upper bound of input parameter

%% Sampling

N = 30; N1 = 1000; n = 10;  % Training sample size ; test sample size ; input dimension 

% Generate samples 
pp = sobolset(n,'Skip',50); u = net(pp,N);   
pp1 = sobolset(n,'Skip',10000,'Leap',N1); u1 = net(pp1,N1);  

% Transform samples to orginal space
for i=1:n
  x(:,i) = u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i) = u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y = g(x); y1 = g(xtest); % Model response
 
for i = 1:N
   Par = [];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8),x(i,9),x(i,10));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
end

%% Test of partial derivative incorporated GE-Kriging model

for i = 1:n
    par.polytype{i} = 'Legendre'; % Orthogonal polynomial of each dimension
end

par.pceorder = 4;         % Maximum order of polynomial order
par.q_truncation = 0.75;  % q-norm truncation
par.dim = n;              % Input dimension
par.lb = lb;              % Input lower bound
par.ub = ub;              % Inpur upper bound

GEPCE_model = GEPCE_fit(x,y,grad_y,par);        % Training GE-PCE 
[Mean Variance] = GEPCE_predict(xtest,GEPCE_model);  % Making prediction
MSE = mean((Mean-y1).^2)/var(y1)

PCE_model = PCE_fit(x,y,par);                    % Training PCE
[Mean Variance] = PCE_predict(xtest,PCE_model);  % Making prediction
MSE1 = mean((Mean-y1).^2)/var(y1)



