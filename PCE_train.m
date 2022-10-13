function model = PCE_train(model)

% Training GE-PCE 

%% Preparation

m1 = model.sample_size; n = model.dim; measure_mat = model.measure_mat;

total_output = model.output; % Model responses

index = model.basisindex;    % retained basis functions in PCE 

order = model.order ;        % Maximum order of PCE model

if n == 1
   Order = index;
else
   Order = sum(index');
end
 
for i = 1 : order
   P(i) = length(find(Order<=i));  % the number of the terms in kernel function 
end

Bh0 = 1:P(2); % Training GE-PCE model with basis functions of maximum order 2

%% Adaptive procedure for training GE-PCE model

for i = 1 :order-1

 if i == 1
   Bh = Bh0;
 else
   Bh = [Bh0 P(i)+1:P(i+1)];
 end

 current_mat = measure_mat(:,Bh);      % Initial measurement matrix
 num = size(current_mat,2);
 w = ones(num,1);    W = diag(w);         
 sigma2 = 1;  Omega = ones(num,1);
 k = 0; I = eye(m1);   

while (1)  

 k = k +1; Omega_old = Omega;  w_old=w;
  
 Cov = sigma2.*I+current_mat*W*current_mat';    % Covariance matrix

 w_inv = diag(w.^-1);   
   
 C_mat = sigma2^-1*current_mat'*current_mat + w_inv;

 if num < m1

   [R1 rd] = chol(C_mat+10^-7.*eye(num));    % Cholesky decomposition 

   C_inv = R1\(R1'\eye(num)); % Matrix inversion

   detC = prod(diag(R1).^2);  % Determinant 

   Cov_inv = sigma2^-1.*I-sigma2^-1*current_mat*C_inv*current_mat'*sigma2^-1; % Woodbury formula

   Omega = W*current_mat'*(Cov_inv*total_output);  % GE-PCE coefficients

   Loglikelihood(k) = log(detC) + m1*log(sigma2) - log(det(w_inv)); % likelihood function

 else  

%   [R1 rd] = chol(Cov +10^-7.*I); 

   [R1 rd] = chol(Cov+10^-7.*I);  % Cholesky decomposition 

   C_inv = W - W*current_mat'*(R1\(R1'\(current_mat*W)));  % Woodbury formula

   Omega = W*current_mat'*(R1\(R1'\total_output)); % GE-PCE coefficients

   detCov = prod(diag(R1).^2); % Determinant 
   
   Loglikelihood(k) = log(detCov) + total_output'*(R1\(R1'\total_output)); % likelihood function

 end

 if (k>=1000) break;  end

 DiagC = diag(C_inv); 

 w = Omega.^2+DiagC;    % Update of weight

 ind = find(w>10^-20);   % Delete the nonsignificant terms

 Num(k) = length(ind);
 num = Num(k);
 w = w(ind);  W=diag(w);  % Delete the small weights
 Bh = Bh(ind);        
 Omega = Omega(ind);
 current_mat = current_mat(:,ind);
 
 DiagC = DiagC(ind);

 gamma=(1-DiagC./w_old(ind));

 sigma2 = ((total_output-current_mat*Omega)'*(total_output-current_mat*Omega))/(m1-sum(gamma)); % Update of noise variance parameters
 
 if (size(Omega)==size(Omega_old))
      dmu = max(max(abs(Omega_old-Omega))); % Convergence threshold
      if (dmu <10^-15)  break;   end
 end
    
end 

 Bh0 = Bh;

end

 model.sigma2 = sigma2;
 model.basisindex = Bh0;
 model.likelihood = Loglikelihood;
 model.Iteration = k;
 model.weight = W; 
 model.Num=Num;
 model.covcoef=C_inv;
 model.covmat=Cov;
 model.coef=Omega;  
 model.measure_mat = current_mat;

end

 

