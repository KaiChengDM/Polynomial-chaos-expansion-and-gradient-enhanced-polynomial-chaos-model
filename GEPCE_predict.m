function [Y_pre V_pre]=GEPCE_predict(x_pre,model)

% GE-PCE prediction

% Model parameter
order = model.order;
polytype = model.polytype;
truncation = model.truncation;
measure_mat = model.measure_mat;
mean_output = model.mean_output;
std_output = model.std_output;
sparse_basis = model.basisindex;
lb = model.lb_input;
ub= model.ub_input;

m1 = size(x_pre,1); 

u_pre = 2.*(x_pre-repmat(lb,m1,1))./(repmat(ub,m1,1)-repmat(lb,m1,1))-1;   % Normalization of input data
  
num = size(measure_mat,2);
  
measure_pre = GE_measurement(u_pre,order,polytype,truncation);  % Generate measurement matrix of GE-PCE basis function

measure_pre = measure_pre(:,sparse_basis); 

covcoef = model.covcoef;
 
coef = model.coef;

Y_pre = measure_pre*coef;  % Making prediction 
  
Y_pre = Y_pre.*std_output+mean_output;
   
for i = 1: m1
   V_pre (i,1) =  diag (measure_pre(i,:) * covcoef * measure_pre(i,:)').*std_output^2 ; % Prediction variance
end

Y_pre=Y_pre(1:m1);

V_pre=V_pre(1:m1);

end