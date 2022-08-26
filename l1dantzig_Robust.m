%% This script is modified from l1dantzig_pd written by Justin Romberg

function xp = l1dantzig_Robust(x0, A, At, b, epsilon, pdtol, pdmaxiter, cgtol, cgmaxiter)
% x0 is beta0 in this function

psi_c = 4.68;

if (nargin < 6), pdtol = 1e-3;  end
if (nargin < 7), pdmaxiter = 50;  end
if (nargin < 8), cgtol = 1e-8;  end
if (nargin < 9), cgmaxiter = 200;  end

N = length(x0);

alpha = 0.01;
beta = 0.5;
mu = 10;

gradf0 = [zeros(N,1); ones(N,1)];


% % % % Estimate S estimator
% % % % Two different rho functions are used for S and MM
% % % rhofuncS='hyperbolic';
% % % rhofuncMM='hampel';
% % % [outS]=Sreg(b,A,'rhofunc',rhofuncS,'nsamp',0);

% Residuals for the initialbeta
[betaRobRidge resid edf lamin]= RobRidge(A,b) ;
disp("edf = "+edf)

lambda_optimal = lamin;
[betaSE betaMM residSE residMM edfSE edfMM mseMM]=RidSEMM(A,b,lambda_optimal);

ini_scale = mscale(residSE);

% starting point --- make sure that it is feasible

  if (max( abs(A'*psi_function(((A*x0 - b)/ini_scale), psi_c)) - epsilon ) > 0)
    disp('Starting point infeasible; using x0 = At*inv(AAt)*y.');
    opts.POSDEF = true; opts.SYM = true;
    [w, hcond] = linsolve(A*A', b, opts);
    if (hcond < 1e-14)
      disp('A*At is ill-conditioned: cannot find starting point');
      xp = x0;
      return;
    end
    x0 = A'*w;
  end  

x = x0;
u = (0.95)*abs(x0) + (0.10)*max(abs(x0));

% set up for the first iteration

  Atr = A'*psi_function(((A*x - b)/ini_scale),psi_c); 

fu1 = x - u;
fu2 = -x - u;
fe1 = Atr - epsilon;
fe2 = -Atr - epsilon;
lamu1 = -(1./fu1);
lamu2 = -(1./fu2);
lame1 = -(1./fe1);
lame2 = -(1./fe2);
% add two more constraints

  AtAv = A'*psi_function((A*(lame1-lame2)/ini_scale), psi_c); %???????????


% sdg = surrogate duality gap
sdg = -[fu1; fu2; fe1; fe2]'*[lamu1; lamu2; lame1; lame2];
tau = mu*(4*N)/sdg;

% residuals
rdual = gradf0 + [lamu1-lamu2 + AtAv; -lamu1-lamu2];
rcent = -[lamu1.*fu1; lamu2.*fu2; lame1.*fe1; lame2.*fe2] - (1/tau);
resnorm = norm([rdual; rcent]);

% iterations
pditer = 0;
done = (sdg < pdtol) | (pditer >= pdmaxiter);
while (~done)

  % solve for step direction
  w2 = - 1 - (1/tau)*(1./fu1 + 1./fu2);
  
  sig11 = -lamu1./fu1 - lamu2./fu2;
  sig12 = lamu1./fu1 - lamu2./fu2;
  siga = -(lame1./fe1 + lame2./fe2);
  sigx = sig11 - sig12.^2./sig11;
  

    w1 = -(1/tau)*( A'*psi_function(((A*(1./fe2-1./fe1))/ini_scale),psi_c) + 1./fu2 - 1./fu1 );
    w1p = w1 - (sig12./sig11).*w2;
    Hp = A'*(A*sparse(diag(siga))*A')*A + diag(sigx);
    opts.POSDEF = true; opts.SYM = true;
    [dx, hcond] = linsolve(Hp, w1p,opts);
    if (hcond < 1e-14)
      disp('Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)');
      xp = x;
      return
    end
    AtAdx = A'*(A*dx);

  du = w2./sig11 - (sig12./sig11).*dx;
  
  dlamu1 = -(lamu1./fu1).*(dx-du) - lamu1 - (1/tau)*1./fu1;
  dlamu2 = -(lamu2./fu2).*(-dx-du) - lamu2 - (1/tau)*1./fu2;
  dlame1 = -(lame1./fe1).*(AtAdx) - lame1 - (1/tau)*1./fe1;
  dlame2 = -(lame2./fe2).*(-AtAdx) - lame2 - (1/tau)*1./fe2;

    % originally AtAdv = A'*(A*(dlame1-dlame2)); 

    AtAdv = A'*(A*(dlame1-dlame2));  

    AtAdv = A'*psi_function( (A*(dlame1-dlame2)/ini_scale), psi_c); 
    %AtAv = A'*psi_function((A*(lame1-lame2)/ini_scale), psi_c); %???????????

	
  
  % find minimal step size that keeps ineq functions < 0, dual vars > 0
  iu1 = find(dlamu1 < 0); iu2 = find(dlamu2 < 0); 
  ie1 = find(dlame1 < 0); ie2 = find(dlame2 < 0);
  ifu1 = find((dx-du) > 0); ifu2 = find((-dx-du) > 0); 
  ife1 = find(AtAdx > 0); ife2 = find(-AtAdx > 0); 
  smax = min(1,min([...
    -lamu1(iu1)./dlamu1(iu1); -lamu2(iu2)./dlamu2(iu2); ...
    -lame1(ie1)./dlame1(ie1); -lame2(ie2)./dlame2(ie2); ...
    -fu1(ifu1)./(dx(ifu1)-du(ifu1)); -fu2(ifu2)./(-dx(ifu2)-du(ifu2)); ...
    -fe1(ife1)./AtAdx(ife1); -fe2(ife2)./(-AtAdx(ife2)) ]));
  s = 0.99*smax;
  
  % backtracking line search
  suffdec = 0;
  backiter = 0;
  while (~suffdec)
  
    xp = x + s*dx;  up = u + s*du;
    Atrp = Atr + s*AtAdx;  AtAvp = AtAv + s*AtAdv;
    fu1p = fu1 + s*(dx-du);  fu2p = fu2 + s*(-dx-du);
    fe1p = fe1 + s*AtAdx;  fe2p = fe2 + s*(-AtAdx);
    lamu1p = lamu1 + s*dlamu1;  lamu2p = lamu2 + s*dlamu2;
    lame1p = lame1 + s*dlame1; lame2p = lame2 + s*dlame2;
    rdp = gradf0 + [lamu1p-lamu2p + AtAvp; -lamu1p-lamu2p];
    rcp = -[lamu1p.*fu1p; lamu2p.*fu2p; lame1p.*fe1p; lame2p.*fe2p] - (1/tau);
    suffdec = (norm([rdp; rcp]) <= (1-alpha*s)*resnorm);
    s = beta*s;
    backiter = backiter+1;
    if (backiter > 32)
      disp('Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)')
      xp = x;
      return
    end
  end
  disp("set up for next iteration")
  % setup for next iteration
  x = xp;  u = up;
  Atr = Atrp;  AtAv = AtAvp;
  fu1 = fu1p; fu2 = fu2p; 
  fe1 = fe1p; fe2 = fe2p;
  lamu1 = lamu1p; lamu2 = lamu2p; 
  lame1 = lame1p; lame2 = lame2p;
  
  sdg = -[fu1; fu2; fe1; fe2]'*[lamu1; lamu2; lame1; lame2];
  tau = mu*(4*N)/sdg;

  rdual = rdp;
  rcent = -[lamu1.*fu1; lamu2.*fu2; lame1.*fe1; lame2.*fe2] - (1/tau);
  resnorm = norm([rdual; rcent]);
  
  pditer = pditer+1;
  done = (sdg < pdtol) | (pditer >= pdmaxiter);
  
  disp(sprintf('Iteration = %d, tau = %8.3e, Primal = %8.3e, PDGap = %8.3e, Dual res = %8.3e',...
    pditer, tau, sum(u), sdg, norm(rdual)));

    disp(sprintf('                  H11p condition number = %8.3e', hcond));
  
end
