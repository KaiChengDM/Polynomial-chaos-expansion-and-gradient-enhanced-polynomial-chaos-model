function  [measure_mat Num trunc_index]=GE_measurement(x,order,polytype,q)

% Compute measurement matrix of orthogonal polynomials 

[m n] = size(x); syms u;

for k = 1 : n
  switch polytype{k}    
    case 'Legendre' 
       
      p(:,1,k) = ones(m,1);            %  Zero order Legendre polynomials 
      pd(:,1,k) = zeros(m,1);          %  Partial derivative of zero order Legendre polynomials 
      p(:,2,k) = x(:,k);               %  First order Legendre polynomials 
      pd(:,2,k) = ones(m,1);           %  Partial derivative of first order Legendre polynomials 
      p(:,3,k) = (3.*x(:,k).^2-1)./2;  %  Second order Legendre polynomials    
      pd(:,3,k) = 3.*x(:,k);           %  Partial derivative of second order Legendre polynomials 

      for i = 3:order 
        p(:,i+1,k) = (2*i-1)./i.*x(:,k).*p(:,i,k)-(i-1)./i.*p(:,i-1,k);               % Recurrence relationship of Legendre polynomials
        pd(:,i+1,k) = (2*i-1)./i.*(p(:,i,k)+x(:,k).*pd(:,i,k))-(i-1)./i.*pd(:,i-1,k); % Recurrence relationship of partial derivative of Legendre polynomials
      end 

    case 'Hermite'
        
      p(:,1,k) = ones(m,1);         % Zero order Hermite polynomials  
      pd(:,1,k) = zeros(m,1);       % Partial derivative of zero order Hermite polynomials 
      p(:,2,k) = 2.*x(:,k);         % First order Hermite polynomials  
      pd(:,1,k) = 2.*ones(m,1);     % Partial derivative of first order Hermite polynomials 
      p(:,3,k) = 4.*x(:,k).^2-2;    % Second order Hermite polynomials
      pd(:,1,k) = 8.*x(:,k);     % Partial derivative of second order Hermite polynomials 

     for i = 3 : order
        p(:,i+1,k) = 2.*x(:,k).*p(:,i,k)-2*i.*p(:,i-1,k);  % Recurrence relationship
        pd(:,i+1,k) = 2.*(x(:,k).*pd(:,i,k)+p(:,i,k))-2*i.*pd(:,i-1,k);  % Recurrence relationship
     end


  end
end 
 
  P = factorial(order+n)/(factorial(order)*factorial(n)); % The number of the terms in kernel function 
       
  Seq = pcegetseq_h(order,n);
  index = zeros(1,n);
  
  for i = 1 : order
     index = [index; Seq{i}];
  end
  
  index = double(index);
  
%   for i = 1 : P  
%     inter(i) = length(find(index(i,:)~=0));   % Compute the interaction order 
%   end
%   
%   ind = find(inter <= truncation);  % Truncation of basis functions
%   trunc_index = index(ind,:);
  
  ind = find((sum(index.^q')).^(1/q) <= order);  % q-norm truncation
  
  trunc_index = index(ind,:);

  Num = length(ind);
  
  for i = 1 : Num       % Compute the basis function value 
     M = ones(m,1);  
    for ii = 1 : n
       M = M.*p(:,trunc_index(i,ii)+1,ii);
    end 
     Measurement(:,i) = M;
  end
  
  trunc_index1 = trunc_index+1;  
  Grad_measurement = zeros(m,Num);
  Grad_Measurement = [];

  for k = 1 : n       % Compute the partial derivative value of basis function 
    for i = 1 : Num  
         M = ones(m,1);
        for ii = 1 : n  
           if ii == k
             M = M.*pd(:,trunc_index1(i,ii),ii); 
           else
             M = M.*p(:,trunc_index1(i,ii),ii);  
           end
        end  
         Grad_measurement(:,i) = M;
    end
    Grad_Measurement = [Grad_Measurement; Grad_measurement]; 
  end

  measure_mat = [Measurement; Grad_Measurement];  % GE-PCE measurement matrix

end

 

