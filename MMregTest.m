clear;
n=30;
p=3;
randn('state', 16); % this is an old way of seeding
X=randn(n,p); 
% Uncontaminated data
y=randn(n,1);
% Contaminated data
ycont=y;
ycont(1:5)=ycont(1:5)+6;


% Two different rho functions are used for S and MM
rhofuncS='hyperbolic';
rhofuncMM='hampel';
% Direct call to MMreg
[out]=MMreg(ycont,X,'Srhofunc',rhofuncS,'rhofunc',rhofuncMM,'Snsamp',0);
% Call to Sreg and then to MMregcore
[outS]=Sreg(ycont,X,'rhofunc',rhofuncS,'nsamp',0);
outMM=MMregcore(ycont,X,outS.beta,outS.scale,'rhofunc',rhofuncMM);
disp('Difference between direct call to S and the calls to Sreg and MMregcore')
max(abs([out.beta-outMM.beta]))
