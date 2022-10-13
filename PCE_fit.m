function PCEmodel = PCE_fit(input,output,par)

% Training a PCE model with Gaussian process regression

% Based on paper: Gradient-enhanced polynomial chaos expansion for
% high-dimensional function approximation, The 13th International Conference on Structural Safety and Reliability (ICOSSAR 2021)At: Shanghai, P.R. China


%  Model parameter
polytype = par.polytype;
order = par.pceorder;
q = par.q_truncation;
dim = par.dim;
lb = par.lb ;
ub = par.ub;

[m n]  =  size(input); 

u = 2.*((input-repmat(lb,m,1))./(repmat(ub,m,1)-repmat(lb,m,1)))-1;   % Normalization of input data

mean_output = mean(output); std_output = std(output);
output = (output-repmat(mean_output,m,1))./repmat(std_output,m,1);   % Normalization of output data

PCEmodel.order = order;
PCEmodel.polytype = polytype;
PCEmodel.truncation = q;

PCEmodel.lb_input = lb;
PCEmodel.ub_input = ub;
PCEmodel.tran_input = u;
PCEmodel.mean_output = mean_output;
PCEmodel.std_output = std_output;
PCEmodel.output = output;

PCEmodel.sample_size = m;
PCEmodel.dim = n;

[measure_mat num trunc_index] = Measurement(u,order,polytype,q); % Generate measurement matrix of GE-PCE basis function

PCEmodel.measure_mat = measure_mat;
PCEmodel.basisnumber = num;
PCEmodel.basisindex = trunc_index;

Sparsemodel = PCE_train(PCEmodel);  % Training sparse GE-PCE 

weight = diag(Sparsemodel.weight);
sigma2 = Sparsemodel.sigma2;

PCEmodel.basisnumber = Sparsemodel.basisnumber;
PCEmodel.basisindex = Sparsemodel.basisindex;

PCEmodel.pcelikelihood = Sparsemodel.likelihood;
PCEmodel.basiscoef = Sparsemodel.coef;
PCEmodel.measure_mat = measure_mat;
PCEmodel.sigma2 = sigma2;
PCEmodel.covmat = Sparsemodel.covmat ;
PCEmodel.coef = Sparsemodel.coef ;
PCEmodel.weight = Sparsemodel.weight;
PCEmodel.covcoef = Sparsemodel.covcoef;
PCEmodel.index = trunc_index(Sparsemodel.basisindex,:);

end
 


