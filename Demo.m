%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%                                                                                                 %
%                  This script is a demo for simulation                                           %
%                                                                                                 %
%                                                                                                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
clc;
%% set a seed 
rng(1006);  
%% Define parameters
eps = 0.1;                  % contamination percentage
N = 72;                     % number of observations  
P = 256;                    % number of parameters   
S = 8;                      % non-zero parameters 
sigma = sqrt(S/N)/3;        % noise level
psi_c = 4.68; 
n_outliers = round(eps*N);  % number of outliers
outlier_magnitude = 5;      % controls how big outliers are
epsilon_robust = 1.25;      % regularization paramter for MM Dantzig selector


   
%% Generate Data
% create true beta
beta = zeros(P,1);
q = randperm(P);  
beta(q(1:S)) = 2*randn(S,1);
g1 = q(1:floor(S/2));   % group 1 index
g2 = q(floor(S/2)+1:S); % group 2 index
beta(g1) = 2 + 0.3*randn(length(g1),1);
beta(g2) = -2 + 0.3*randn(length(g2),1);

% generate a random matrix X
X = randn(N,P);
X= normc(X);                % normalize X
vecnorm(X);
% add noise
e = sigma*randn(N,1);
y = X*beta + e;             % this is the clean y 

% Monte Carlo simulation for lambda_p
    n_mcs = 50;
    lambda_vec = zeros(1,n_mcs);
    for k = 1:n_mcs
        e = randn(N,1); 
        lambda_vec(k) = max(abs(X'*e));
    end
    lambda_p = max(lambda_vec);
    epsilon = lambda_p*sigma;
        
    %% For clean data
    % initial guess for DS = min energy 
    beta0 = X'*y; 

    % (1) DS estiamte:
    beta_DS = l1dantzig_pd(beta0, X, [], y, epsilon, 5e-2);
    % add threshold
    beta_DS(abs(beta_DS) <= epsilon)=0;
    
    % Gauss-Dantzig (two-stage)
    I_hat = find(beta_DS);
    X_submatrix = X(:,I_hat);
    beta_I_hat = inv(X_submatrix'*X_submatrix)*X_submatrix'*y;
    beta_2stage = zeros(P,1);
    % (2) two-stage estiamte
    beta_2stage(I_hat) = beta_I_hat;
    
    % (3) MM DS 
    % for clean data, epsilon is the same
    beta_robust = l1dantzig_Robust( beta0, X, [], y, epsilon, 5e-2); 
    % add threshold
    beta_robust(abs(beta_robust) <= epsilon)=0;
    
   % (4) MM DS with IR (scaled)
   % (5) MM 2 stage
   [beta_scaledIR, beta_2stage_Robust] = MMDantzig_scaledIR(beta0,X, y, epsilon);

  
    
    %% For contaminated data
    % randomly choose n_outliers observations to contaminate
    q2 = randperm(N);
    y_contaminated = y;
    y_contaminated(q2(1:n_outliers)) = y(q2(1:n_outliers)) + outlier_magnitude*sign(y(q2(1:n_outliers)));
    
    % initial guess for DS: Robust Ridge estimates
    [betaRobRidge_cont resid_cont edf_cont lamin_cont]= RobRidge(X,y_contaminated);
    
    % Dantzig selection initial 
    beta0_cont = betaRobRidge_cont(1:end-1);
    
    % (1) DS estimate for contaminated data
    beta_DS_cont = l1dantzig_pd(beta0_cont, X, [], y_contaminated, epsilon, 5e-2);
    % add threshold
    beta_DS_cont(abs(beta_DS_cont) <= epsilon)=0;
    
    % (2) Gauss-Dantzig (two-stage) DS
    I_hat_cont = find(beta_DS_cont);
    X_submatrix = X(:,I_hat_cont);
    beta_I_hat_cont = inv(X_submatrix'*X_submatrix)*X_submatrix'*y_contaminated;
    beta_2stage_cont = zeros(P,1);
    % two-stage estiamte
    beta_2stage_cont(I_hat_cont) = beta_I_hat_cont;

    
    % MM DS for contaminated data (3)
    beta_robust_cont = l1dantzig_Robust(beta0_cont, X, [], y_contaminated, epsilon_robust, 5e-2); 
    % add threshold
    beta_robust_cont(abs(beta_robust_cont) <= epsilon_robust)=0;
 
   
   % (5) MM DS with IR two-stage contaminated
   [beta_scaledIR_cont, beta_2stage_Robust_cont] = MMDantzig_scaledIR(beta0_cont,X, y_contaminated, epsilon);

%% Figures 
figure(1)
subplot(1,2,1)
plot(beta, "+", color = 'blue', LineWidth= 2.5)
ylim([-3.8,4.3])
hold on
plot(beta_DS, "*", color = 'magenta', LineWidth= 2.5)
plot(beta_2stage, "o", color = 'red', LineWidth= 2.5)
plot(beta_robust,"s", color = 'black', LineWidth= 2.5)
plot(beta_robust, "d", color = 'green', LineWidth= 2.5)% IR scaled weight
plot(beta_2stage_Robust, "x", color = 'cyan', LineWidth = 2.5)
legend("True \beta","Dantzig estimates","Two-stage estimates", "Original MM Dantzig estimates", ...
    "MM Dantzig estimates with IR (scaled)", "Two-stage MM estimates")
title('Estimates and the true values of \beta (clean data)')

subplot(1,2,2)
plot(beta, "+", color = 'blue', LineWidth= 2.5)
ylim([-3.8,4.3])
hold on
plot(beta_DS_cont, "*", color = 'magenta', LineWidth= 2.5)
plot(beta_2stage_cont, "o", color = 'red', LineWidth= 2.5)
plot(beta_robust_cont,"s", color = 'black', LineWidth= 2.5)
plot(beta_scaledIR_cont, "d", color = 'green', LineWidth= 2.5) % IR scaled weight
plot(beta_2stage_Robust_cont, "x", color = 'cyan', LineWidth = 2.5)
legend("True \beta","Dantzig estimates","Two-stage estimates", "Original MM Dantzig estimates", ...
    "MM Dantzig estimates with IR (scaled)", "Two-stage MM estimates")
title('Estimates and the true values of \beta (contaminated data)')



figure(3)
subplot(1,2,1)
plot(beta, "+", color = 'blue', LineWidth= 2.5)
ylim([-3.8,4.5])
hold on
plot(beta_DS_cont, "*", color = 'magenta', LineWidth= 2.5)
plot(beta_2stage_cont, "o", color = 'red', LineWidth= 2.5)
plot(beta_robust_cont, "d", color = 'green', LineWidth= 2.5) % IR scaled weight
plot(beta_2stage_Robust_cont, "x", color = 'cyan', LineWidth = 2.5)
legend("True \beta","Dantzig estimates","Two-stage estimates", "Original MM Dantzig estimates", ...
     "MM Dantzig estimates with IR (scaled)", "Two-stage MM estimates",'fontsize', 9)
title('Estimates and the true values of \beta (contaminated data)')
subplot(1,2,2)
plot(beta, "+", color = 'blue', LineWidth= 3)
ylim([-3.1,3.5])
hold on
plot(beta_robust_cont, "d", color = 'green', LineWidth= 3) % IR scaled weight
plot(beta_2stage_Robust_cont, "x", color = 'cyan', LineWidth = 3)
legend("True \beta", "Original MM Dantzig estimates", ...
     "MM Dantzig estimates with IR (scaled)",  "Two-stage MM estimates", 'fontsize', 9)
title('MM estimates and the true values of \beta (contaminated data)')