clear all;  clc;

format long;

% Performance function
g = @(x)sin(x(:,1))+7.*sin(x(:,2)).^2+0.1.*(x(:,3)).^4.*sin(x(:,1));

% Partial derivative function
Pd{1} = @(x) cos(x(:,1))+0.1.*(x(:,3)).^4.*cos(x(:,1));
Pd{2} = @(x) 14.*cos(x(:,2)).*sin(x(:,2));
Pd{3} = @(x) 0.4.*(x(:,3)).^3.*sin(x(:,1));

%% Sampling 

N = 30; N1 = 1000;  n = 3;  % Training sample size ; test sample size ; input dimension 

lb = -pi.*ones(1,n);  ub = pi.*ones(1,n); % Lower bound and upper bound of input parameter

% generate samples 
pp = sobolset(n,'Skip',3); u = net(pp,N);  
pp1 = sobolset(n,'Skip',100,'Leap',N1); u1 = net(pp1,N1);  

% Transform samples to orginal space
for i = 1:n
  x(:,i) = u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i) = u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y = g(x);   y1 = g(xtest);   % model response

% Gradient information
for i = 1:N
   Par = [];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,:));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
end

%% Training GE-PCE and PCE model

for i = 1:n
  par.polytype{i} = 'Legendre'; 
end

par.pceorder = 12;
par.q_truncation = 1;
par.dim = n;
par.lb = lb;
par.ub = ub;

%% Gradient enhanced sparse PCE 

GEPCE_model = GEPCE_fit(x,y,grad_y,par);
[Mean Variance] = GEPCE_predict(xtest,GEPCE_model);
MSE = mean((Mean-y1).^2)/var(y1)

%% Gradient-free sparse PCE 
% 
PCE_model = PCE_fit(x,y,par);
[Mean1 Variance1] = PCE_predict(xtest,PCE_model);
MSE1 = mean((Mean1-y1).^2)/var(y1)
