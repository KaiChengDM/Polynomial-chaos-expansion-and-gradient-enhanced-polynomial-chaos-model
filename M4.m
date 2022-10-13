clc;  clear;
format long;
syms x1 x2;

% Performance function
g = @(x)(4-2.1.*x(:,1).^2+x(:,1).^4./3).*x(:,1).^2+x(:,1).*x(:,2)+(-4+4.*x(:,2).^2).*x(:,2).^2;

% Symbolic performance function
G = (4-2.1.*x1.^2+x1.^4./3).*x1.^2+x1.*x2+(-4+4.*x2.^2).*x2.^2;

grad_f = [diff(G,x1),diff(G,x2)];

% Partial derivative function
for i = 1:2
  Pd{i} = matlabFunction(grad_f(i));
end

%% Sampling

lb=[-2 -1];  ub=[2 1];  % Lower bound and upper bound of input parameter

N = 15; N1 = 3000;  n = 2; % Training sample size ; test sample size ; input dimension 

% Generate samples 
pp = sobolset(n,'Skip',3); u=net(pp,N);  
pp1 = sobolset(n,'Skip',100,'Leap',N1); u1=net(pp1,N1);  

% Transform samples to orginal space
for i = 1 : n
  x(:,i) = u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i) = u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y = g(x);   y1 = g(xtest);   % Model response

% Gradient information
for i = 1:N
   Par = [];
  for j = 1:n
    Par_output(i) = Pd{j}(x(i,1),x(i,2));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
end

%% Training GE-PCE and PCE model

for i = 1:n
 par.polytype{i} = 'Legendre'; 
end

par.pceorder = 10;
par.q_truncation = 1; 
par.dim = n;
par.lb = lb;
par.ub = ub;
%%

GEPCE_model = GEPCE_fit(x,y,grad_y,par);            % Training GE-PCE 
[Mean Variance] = GEPCE_predict(xtest,GEPCE_model); % Making prediction
MSE = mean((Mean-y1).^2)/var(y1)

%% 
PCE_model = PCE_fit(x,y,par);                     % Training PCE
[Mean1 Variance1] = PCE_predict(xtest,PCE_model); % Making prediction
MSE1 = mean((Mean1-y1).^2)/var(y1)

%% Response surface of true function and surrogate model

nn = 200;
xx = lb(1):(ub(1)-lb(1))/(nn-1):ub(1);
yy = lb(2):(ub(2)-lb(2))/(nn-1):ub(2);

[X,Y] = meshgrid(xx,yy);
xnod  = cat(2,reshape(X',nn^2,1),reshape(Y',nn^2,1));

[Mean Variance] = GEPCE_predict(xnod,GEPCE_model);
[Mean1 Variance1] = PCE_predict(xnod,PCE_model);

Z = reshape(Mean,nn,nn); Z1 = reshape(Variance,nn,nn); 
Z2 = reshape(Mean1,nn,nn); Z3 = reshape(Variance1,nn,nn); 

figure
subplot(1,2,1)
mesh(X,Y,Z');hold on
title('GE-PCE mean');
xlabel('x1'); ylabel('x2'); 
grid on;
hold on;

subplot(1,2,2)
contourf(X,Y,Z',20); hold on
plot3(x(:,1),x(:,2),y,'ro','linewidth',1.5);
xlabel('x1'); ylabel('x2'); 

figure
subplot(1,2,1)
mesh(X,Y,Z1');
title('GE-PCE variance');
xlabel('x1'); ylabel('x2'); 

subplot(1,2,2)
contourf(X,Y,Z1',20); hold on
plot3(x(:,1),x(:,2),y,'ro','linewidth',1.5);
xlabel('x1'); ylabel('x2'); 

%% Response surface of true function

yy= g(xnod);

figure
subplot(1,2,1)
Z=reshape(yy,nn,nn); 
mesh(X,Y,Z');
title('True response');
xlabel('x1'); ylabel('x2'); 
grid on;
xlabel('x1'); ylabel('x2'); 
subplot(1,2,2)
contourf(X,Y,Z',20); hold on
