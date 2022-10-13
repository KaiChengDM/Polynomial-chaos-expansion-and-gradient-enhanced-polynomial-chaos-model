function Seq = pcegetseq_h(order,d)

% Compute basis function index 

    Seq = cell(order,1);
    for i = 1:order
        seq = pcegetseq(i,d);
        Seq{i} = seq;
    end
end