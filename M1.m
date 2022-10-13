clc;  clear;

format long;

syms x1 x2 x3 x4 x5 x6 x7 x8

% Performance function
g = @(x)2.*pi.*x(:,3).*(x(:,4)-x(:,6))./(log(x(:,2)./x(:,1)).*(1+2.*x(:,7).*x(:,3)./(log(x(:,2)./x(:,1)).*x(:,1).^2.*x(:,8))+x(:,3)./x(:,5)));

% Symbolic performance function
G = 2.*pi.*x3.*(x4-x6)./(log(x2./x1).*(1+2.*x7.*x3./(log(x2./x1).*x1.^2.*x8)+x3./x5));

grad_f = [diff(G,x1),diff(G,x2),diff(G,x3),diff(G,x4),diff(G,x5),diff(G,x6),diff(G,x7),diff(G,x8)];

% Partial derivative function
Pd1 = matlabFunction(grad_f(1));
Pd2 = matlabFunction(grad_f(2));
Pd3 = matlabFunction(grad_f(3));
Pd4 = matlabFunction(grad_f(4));
Pd5 = matlabFunction(grad_f(5));
Pd6 = matlabFunction(grad_f(6));
Pd7 = matlabFunction(grad_f(7));
Pd8 = matlabFunction(grad_f(8));

lb = [0.05  100   63070  990  63.1 700 1120  9855 ]; % Lower bound of input parameter
ub = [0.15 50000 115600  1110 116  820 1680 12045 ]; % Upper bound of input parameter

%% Sampling

N = 50; N1 = 3000; n=8;   % Training sample size ; test sample size ; input dimension 

% Generate samples 
pp = sobolset(n,'Skip',3); u = net(pp,N);    
pp1 = sobolset(n,'Skip',1000,'Leap',N1); u1 = net(pp1,N1);  

% Transform samples to orginal space
for i = 1:n
  x(:,i) = u(:,i)*(ub(i)-lb(i))+lb(i);
  xtest(:,i) = u1(:,i)*(ub(i)-lb(i))+lb(i);
end

y = g(x);   y1 = g(xtest);   % model response

% Gradient information
Grad_y = [];
for i = 1:N
    
  Par_output_1(i) = Pd1(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
  Par_output_2(i) = Pd2(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
  Par_output_3(i) = Pd3(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
  Par_output_4(i) = Pd4(x(i,1),x(i,2),x(i,3),x(i,5),x(i,7),x(i,8));
  Par_output_5(i) = Pd5(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
  Par_output_6(i) = Pd6(x(i,1),x(i,2),x(i,3),x(i,5),x(i,7),x(i,8));
  Par_output_7(i) = Pd7(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
  Par_output_8(i) = Pd8(x(i,1),x(i,2),x(i,3),x(i,4),x(i,5),x(i,6),x(i,7),x(i,8));
  grad_y(i,:) = [Par_output_1(i) Par_output_2(i)  Par_output_3(i)  Par_output_4(i) Par_output_5(i) Par_output_6(i) Par_output_7(i) Par_output_8(i)];

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

GEPCE_model = GEPCE_fit(x,y,grad_y,par);             % Training GE-PCE 
[Mean Variance] = GEPCE_predict(xtest,GEPCE_model);  % Making prediction
MSE = mean((Mean-y1).^2)/var(y1)

PCE_model = PCE_fit(x,y,par);                    % Training PCE
[Mean Variance] = PCE_predict(xtest,PCE_model);  % Making prediction
MSE1 = mean((Mean-y1).^2)/var(y1)