function [beta_scaledIR, beta_2stage_Robust] = MMDantzig_scaledIR(beta0,X, y, epsilon)
%% This function returns the estimated beta vector using the proposed algorithm with scaled weights
    psi_c = 4.68; 
    MaxIter = 25;               % for IR
    tolerance = 1e-4;           % tolerance level for IR
    epsilon_robust = 1.25;      % regularization paramter for MM Dantzig selector
 
    % First compute Robust DS 
    beta_robust = l1dantzig_Robust( beta0, X, [], y, epsilon_robust, 5e-2); 
    % add threshold
    beta_robust(abs(beta_robust) <= epsilon_robust)=0;
    % Iterative reweighting procedure
    iter = 0;
    % initial residuals and weights
    residuals = y - X*beta_robust;
    scale = mscale(residuals);
    Weight = max(0.0001, (psi_function(residuals/scale, psi_c)./residuals)); % this weight function has scale: psi(x/s)/x 
    WeightMatrix = diag(Weight);
    WeightedX = WeightMatrix*X;
    firstWeight = WeightedX; % save as weighted one time
    gof = false;
    while ~gof && iter < MaxIter
        residuals_prev = residuals;
        beta_robust = l1dantzig_pd(beta_robust, WeightedX, [], y, epsilon, 5e-2); 
        beta_robust(abs(beta_robust) <= epsilon)=0;
    
        residuals = y - X*beta_robust;
        scale = mscale(residuals);
        Weight = max(0.0001, (psi_function(residuals/scale, psi_c)./residuals)); % this weight function has scale: psi(x/s)/x
        WeightedX = diag(Weight)*X;
        % convergence check
        iter = iter+1;
        gof = (max(abs(residuals - residuals_prev)) < tolerance);
       disp(['Reweighting iteration: ', num2str(iter)]);
    end

    beta_scaledIR = beta_robust;

    % MM DS with IR two-stage clean data 
    I_hat_Robust = find(beta_robust);
    WeightedX = firstWeight;
    X_submatrix_Robust = WeightedX(:,I_hat_Robust);
    beta_I_hat_Robust = inv(X_submatrix_Robust'*X_submatrix_Robust)*X_submatrix_Robust'*y;
    [N, P] = size(X);
    beta_2stage_Robust = zeros(P,1);
    % two-stage estimate
    beta_2stage_Robust(I_hat_Robust) = beta_I_hat_Robust;

end