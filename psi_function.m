function psi = psi_function(x, c)
% Psi function c = 4.68
% derivative of rho function 
leng = length(x);
psi = zeros(leng,1);

for i = 1:leng
    
    psi(i) = (x(i)*(1-(x(i)^2/c^2))^2)*(abs(x(i)) <= c);

end

