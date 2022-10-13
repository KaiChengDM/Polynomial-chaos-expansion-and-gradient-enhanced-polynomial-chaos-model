function seq = pcegetseq(n,d)

% SPGETSEQ  Compute the sets of indices 

n = uint8(n);
d = uint16(d);

nlevels = uint32(nchoosek(double(n)+double(d)-1,double(d)-1));
seq = zeros(nlevels,d,'uint8');

seq(1,1) = n;
max = n;
for k = uint32(2):nlevels
	if seq(k-1,1) > uint8(0)
		seq(k,1) = seq(k-1,1) - 1;
		for l = uint16(2):d
			if seq(k-1,l) < max
				seq(k,l) = seq(k-1,l) + 1;
				for m = l+1:d
					seq(k,m) = seq(k-1,m);
				end
				break;
			end
		end
	else
		sum = uint8(0);
		for l = uint16(2):d
			if seq(k-1,l) < max
				seq(k,l) = seq(k-1,l) + 1;
				sum = sum + seq(k,l);
				for m = l+1:d
					seq(k,m) = seq(k-1,m);
					sum = sum + seq(k,m);
				end
				break;
			else
				temp = uint8(0);
				for m = l+2:d
					temp = temp + seq(k-1,m);
				end
				max = n-temp;
				seq(k,l) = 0;
			end
		end
		seq(k,1) = n - sum;
		max = n - sum;
	end
end
