function  [measure_mat Num trunc_index] = Measurement(x,order,polytype,q)

% Compute measurement matrix of orthogonal polynomials 

[m n] = size(x); syms u;

for k = 1 : n
  switch polytype{k}    
    case 'Legendre' 
       
      p(:,1,k) = ones(m,1);            %  Zero order Legendre polynomials 
      p(:,2,k) = x(:,k);               %  First order Legendre polynomials 
      p(:,3,k) = (3.*x(:,k).^2-1)./2;  %  Second order Legendre polynomials    

      for i = 3:order 
        p(:,i+1,k) = (2*i-1)./i.*x(:,k).*p(:,i,k)-(i-1)./i.*p(:,i-1,k);               % Recurrence relationship of Legendre polynomials
      end   

    case 'Hermite'

     p(:,1,k) = ones(m,1);         % Constant term
     p(:,2,k) = 2.*x(:,k);         % First order Hermite polynomials  
     p(:,3,k) = 4.*x(:,k).^2-2;    % Second order Hermite polynomials

     for i = 3 : order
       p(:,i+1,k) = 2.*x(:,k).*p(:,i,k)-2*(i).*p(:,i-1,k);  % Recurrence relationship
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

  ind = find((sum(index.^q')).^(1/q) <= order);
  
  trunc_index = index(ind,:);

  Num = length(ind);
  
  for i = 1 : Num      % Compute the value of the orthogonal polynomials 
     M = ones(m,1);  
    for ii = 1 : n
       M = M.*p(:,trunc_index(i,ii)+1,ii);
    end 
     Measurement(:,i) = M;
  end
  
  measure_mat = Measurement;  % PCE measurement matrix

end

 

