
clc;  clear;

format long;

syms x1 x2 x3 x4 x5 

% Performance function
g = @(x)((log(x(:,1).^2)).^2+(log(x(:,1))).^2)+((log(x(:,2).^2)).^2+(log(x(:,2))).^2)+((log(x(:,3).^2)).^2+(log(x(:,3))).^2)+((log(x(:,4).^2)).^2+(log(x(:,4))).^2)+((log(x(:,5).^2)).^2+(log(x(:,5))).^2)-(x(:,1).*x(:,2).*x(:,3).*x(:,4).*x(:,5)).^0.2;

% Symbolic performance function
G = ((log(x1.^2)).^2+(log(x1)).^2)+((log(x2.^2)).^2+(log(x2)).^2)+((log(x3.^2)).^2+(log(x3)).^2)+((log(x4.^2)).^2+(log(x4)).^2)+((log(x5.^2)).^2+(log(x5)).^2)-(x1.*x2.*x3.*x4.*x5).^0.2;

grad_f = [diff(G,x1),diff(G,x2),diff(G,x3),diff(G,x4),diff(G,x5)];

% Partial derivative function
for i = 1:5
   Pd{i} = matlabFunction(grad_f(i));
end
 
lb = ones(1,5);     % Lower bound of input parameter
ub = 10.*ones(1,5); % Upper bound of input parameter

%% Sampling

N = 20; N1 = 1000; n = 5;  % Training sample size ; test sample size ; input dimension 

% Generate samples 
pp = sobolset(n,'Skip',50); u = net(pp,N);   
pp1 = sobolset(n,'Skip',10000,'Leap',N1); u1 = net(pp1,N1);  

% Transform samples to orginal space
for i = 1:n
  x(:,i) = u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i) = u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y = g(x); y1 = g(xtest);  % Model response

% Gradient information
for i = 1:N
   Par = [];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
end

%% Training GE-PCE and PCE model 

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



