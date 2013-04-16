function myconvolve(a,b)

N = size(a,2);
M = size(b,2);

c = zeros(1,N);


for k=0:(N-1)
    for i=0:(M-1)
        c(k+1) = c(k+1) + a(mod(k-i,N)+1) * b(i+1) ;
    end;
end;

%for k=0:(p-1)
%    for i=0:(p-1)
%	    c(k+1) = c(k+1) + a(mod(k-i,p)+1) * b(mod(i,p)+1);
%    end;
%end;

%for k=0:(N-1)
%    for i=(k-M+1):k
%        c(k+1) = c(k+1) + a(mod(i,N)+1) * b(k-i+1);
%    end;
%end;

c

