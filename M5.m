clc;  clear;

format long;

n = 30; % Input dimension

% Performance function
g = @(x)(3-0.25.*sum((x.*(1:n))')+0.05.*sum((x.^3.*(1:n))')+log(1/60.*sum(((x.^2+x.^4).*(1:n))')))';

% Partial derivative function
for i = 1:n
  Pd{i} = @(x) -0.25*i+0.15.*i*x(:,i).^2+(1/60.*i*(2.*x(:,i)+4.*x(:,i).^3))./(1/60.*sum(((x.^2+x.^4).*(1:n))')');
end

%% Sampling

Samplesize = 30: 30: 150;

for k = 1:5

x = [];

lb = -2.*ones(1,n);  ub = 2.*ones(1,n); % Lower bound and upper bound of input parameter

N = Samplesize(k); N1 = 3000;   % Training sample size ; test sample size ;

% Generate samples 
pp = sobolset(n,'Skip',3); u=net(pp,N);  
pp1 = sobolset(n,'Skip',100,'Leap',N1); u1=net(pp1,N1);  

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
    Par_output(i) = Pd{j}(x(i,:));
    Par = [Par Par_output(i)];
  end
  grad_y(i,:) = Par;
end

%% Training GE-PCE and PCE model
for i = 1:n
   par.polytype{i} = 'Legendre';
end
par.pceorder = 3;
par.q_truncation = 0.5;
par.dim = n;
par.lb = lb;
par.ub = ub;

t1=clock;
  GEPCE_model = GEPCE_fit(x,y,grad_y,par);          % Training GE-PCE 
t2=clock;
Time(k) = etime(t2,t1)
[Mean Variance] = GEPCE_predict(xtest,GEPCE_model); % Making prediction
MSE(k) = mean((Mean-y1).^2)/var(y1)

t1=clock;
  PCE_model = PCE_fit(x,y,par);            % Training PCE
t2=clock;
Time1(k) = etime(t2,t1)
[Mean1 Variance1] = PCE_predict(xtest,PCE_model); % Making prediction
MSE1(k) = mean((Mean1-y1).^2)/var(y1)

end


subplot(1,2,1)
ii = 1: 5;
plot (ii,(Time),'-o','LineWidth',2); hold on;
plot (ii,(Time1),'-*','LineWidth',2); hold on;
legend('GE-PCE','PCE')
xlabel('Model evaluations'); ylabel('Training time (s)')

subplot(1,2,2)
plot (ii,MSE,'-o','LineWidth',2); hold on;
plot (ii,MSE1,'-*','LineWidth',2); hold on;
legend('GE-PCE','PCE')
xlabel('Model evaluations'); ylabel('RMSE')

