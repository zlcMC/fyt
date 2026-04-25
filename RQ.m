%% FC2 UKF parameter sensitivity analysis: Q and R scaling (元胞数组终极防爆版)
clear; clc; close all;

%% ==================== 0. Load TS-LSTM model ====================
disp('0. Loading trained TS-LSTM model...');
model_file = 'FC1_TS_LSTM_Attention_Model.mat';
if ~isfile(model_file)
    error('Cannot find FC1_TS_LSTM_Attention_Model.mat');
end
S_model = load(model_file);
ts_lstm_net = S_model.ts_lstm_net;
x_mean = S_model.x_mean;
x_std  = S_model.x_std;
y_mean = S_model.y_mean;
y_std  = S_model.y_std;

if canUseGPU
    exec_env = 'gpu';
else
    exec_env = 'auto';
end

%% ==================== 1. Load FC2 ripple dataset ====================
disp('1. Loading FC2 ripple dataset...');
fc2_folder = 'F:\FC1_FC2_Excel\FC1_FC2_Excel\FC2_With_Ripples_Excel\';
file_names = { [fc2_folder, 'FC2_Ageing_part1.xlsx'], ...
               [fc2_folder, 'FC2_Ageing_part2.csv'] };
data_full = table();

for i = 1:length(file_names)
    fprintf('   Reading: %s\n', file_names{i});
    temp_data = readtable(file_names{i}, 'VariableNamingRule', 'preserve');
    vars = temp_data.Properties.VariableNames;
    for j = 1:length(vars)
        col = temp_data.(vars{j});
        if iscell(col) || isstring(col) || ischar(col)
            col_str = strrep(string(col), ',', '.');
            temp_data.(vars{j}) = str2double(col_str);
        else
            temp_data.(vars{j}) = double(col);
        end
    end
    data_full = [data_full; temp_data];
end

try
    I_raw     = data_full.('I (A)');
    T_C_raw   = data_full.('ToutWAT (ｰC)');
    P_air_raw = data_full.('PoutAIR (mbara)');
    P_H2_raw  = data_full.('PoutH2 (mbara)');
    Utot_raw  = data_full.('Utot (V)');
    J_raw     = data_full.('J (A/cmｲ)');
catch
    I_raw     = data_full{:,9};
    J_raw     = data_full{:,8};
    T_C_raw   = data_full{:,15};
    P_air_raw = data_full{:,17};
    P_H2_raw  = data_full{:,18};
    Utot_raw  = data_full{:,7};
end

valid_mask_raw = ~isnan(I_raw)     & isfinite(I_raw) & ...
                 ~isnan(J_raw)     & isfinite(J_raw) & ...
                 ~isnan(T_C_raw)   & isfinite(T_C_raw) & ...
                 ~isnan(P_air_raw) & isfinite(P_air_raw) & ...
                 ~isnan(P_H2_raw)  & isfinite(P_H2_raw) & ...
                 ~isnan(Utot_raw)  & isfinite(Utot_raw);
I_raw     = I_raw(valid_mask_raw);
J_raw     = J_raw(valid_mask_raw);
T_C_raw   = T_C_raw(valid_mask_raw);
P_air_raw = P_air_raw(valid_mask_raw);
P_H2_raw  = P_H2_raw(valid_mask_raw);
Utot_raw  = Utot_raw(valid_mask_raw);

n_cells = 5;
valid_area_idx = find(abs(J_raw) > 1e-6);
if isempty(valid_area_idx)
    error('J_raw is invalid, cannot estimate A_area.');
end
idx_use = valid_area_idx(1:min(100, length(valid_area_idx)));
A_area = median(I_raw(idx_use) ./ J_raw(idx_use));

if isnan(A_area) || ~isfinite(A_area) || A_area < 10
    warning('A_area is abnormal, fallback to 100.0 cm^2');
    A_area = 100.0;
end
l_thick = 0.00508;
V_actual_raw = Utot_raw / n_cells;
T_K_raw      = T_C_raw + 273.15;
P_air_atm    = max(P_air_raw / 1013.25, 1e-3);
P_H2_atm     = max(P_H2_raw  / 1013.25, 1e-3);

%% ==================== 2. Downsampling and normalization ====================
disp('2. Downsampling and normalization...');
window_size = 60;
N_windows = floor(length(I_raw) / window_size);
I_test    = zeros(N_windows,1);
T_test    = zeros(N_windows,1);
Pair_test = zeros(N_windows,1);
PH2_test  = zeros(N_windows,1);
V_test    = zeros(N_windows,1);

for i = 1:N_windows
    idx = (i-1)*window_size+1 : i*window_size;
    I_test(i)    = mean(I_raw(idx));
    T_test(i)    = mean(T_K_raw(idx));
    Pair_test(i) = mean(P_air_atm(idx));
    PH2_test(i)  = mean(P_H2_atm(idx));
    V_test(i)    = mean(V_actual_raw(idx));
end

X_test_raw = [I_test, T_test, Pair_test, PH2_test];
x_mean_rep = repmat(x_mean, size(X_test_raw,1), 1);
x_std_rep  = repmat(x_std,  size(X_test_raw,1), 1);
x_std_rep(abs(x_std_rep) < 1e-12) = 1.0;
X_test_norm = (X_test_raw - x_mean_rep) ./ x_std_rep;

%% ==================== 3. Prepare TS-LSTM prior prediction ====================
disp('3. Running batch TS-LSTM prior prediction...');
seq_len = 80;
valid_len = N_windows - seq_len + 1;
if valid_len <= 0
    error('N_windows is smaller than seq_len.');
end

X_all_cells = cell(valid_len, 1);
for k = seq_len:N_windows
    X_all_cells{k-seq_len+1} = X_test_norm(k-seq_len+1:k, :)';
end
y_pred_all_norm = predict(ts_lstm_net, X_all_cells, ...
    'ExecutionEnvironment', exec_env, ...
    'MiniBatchSize', 256);

if iscell(y_pred_all_norm)
    temp_pred_mat = zeros(length(y_pred_all_norm), 2);
    for c_idx = 1:length(y_pred_all_norm)
        temp_pred_mat(c_idx, :) = y_pred_all_norm{c_idx}(:)';
    end
    y_pred_all_norm = temp_pred_mat;
end

y_mean_rep = repmat(y_mean, size(y_pred_all_norm,1), 1);
y_std_rep  = repmat(y_std,  size(y_pred_all_norm,1), 1);
y_std_rep(abs(y_std_rep) < 1e-12) = 1.0;
y_pred_all = y_pred_all_norm .* y_std_rep + y_mean_rep;
y_pred_all(:,1) = min(max(y_pred_all(:,1), 3.0), 14.0);
y_pred_all(:,2) = min(max(y_pred_all(:,2), -1.5), -0.5);

%% ==================== 4. Sensitivity settings ====================
Q_base = diag([5e-4, 5e-6]);
R_base = 0.008;
P0_base = diag([0.08, 0.008]);
scale_list = [0.1, 0.5, 1, 2, 10];
lambda_step_max = 0.03;
xi1_step_max    = 0.002;

out_dir = fullfile('消融实验', 'UKF_Sensitivity_QR');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ==================== 5. Q sensitivity: scale Q, fix R ====================
disp('4. Running Q sensitivity analysis...');
Q_metrics = [];
% 【绝对防爆修改 1】：放弃极其危险的结构体数组，改用元胞数组 (Cell Array) 存储
Q_results = cell(length(scale_list), 1); 

for s = 1:length(scale_list)
    q_scale = scale_list(s);
    Q_process = q_scale * Q_base;
    R_fixed = R_base;
    
    [result_now] = run_ukf_once(...
        N_windows, seq_len, y_pred_all, ...
        I_test, T_test, Pair_test, PH2_test, V_test, ...
        A_area, l_thick, ...
        P0_base, Q_process, R_fixed, ...
        lambda_step_max, xi1_step_max);
        
    result_now.study_type = 'Q_sensitivity';
    result_now.scale = q_scale;
    result_now.Q_process = Q_process;
    result_now.R_used = R_fixed;
    
    % 【绝对防爆修改 2】：使用大括号存入集装箱，MATLAB 绝不会再检查内部字段一致性
    Q_results{s} = result_now; 
    
    Q_metrics = [Q_metrics; ...
        table(q_scale, result_now.rmse_v_prior, result_now.rmse_v_post, ...
              result_now.mae_v_prior, result_now.mae_v_post, ...
              result_now.maxae_v_prior, result_now.maxae_v_post, ...
              'VariableNames', {'Q_scale','RMSE_prior','RMSE_post','MAE_prior','MAE_post','MaxAE_prior','MaxAE_post'})]; %#ok<AGROW>
              
    save(fullfile(out_dir, sprintf('RESULT_Qscale_%g.mat', q_scale)), 'result_now');
end
writetable(Q_metrics, fullfile(out_dir, 'Q_sensitivity_metrics.csv'));
save(fullfile(out_dir, 'Q_sensitivity_all_results.mat'), 'Q_results', 'Q_metrics');

%% ==================== 6. R sensitivity: scale R, fix Q ====================
disp('5. Running R sensitivity analysis...');
R_metrics = [];
% 【绝对防爆修改 3】：同样改用元胞数组
R_results = cell(length(scale_list), 1); 

for s = 1:length(scale_list)
    r_scale = scale_list(s);
    Q_process = Q_base;
    R_fixed = r_scale * R_base;
    
    [result_now] = run_ukf_once(...
        N_windows, seq_len, y_pred_all, ...
        I_test, T_test, Pair_test, PH2_test, V_test, ...
        A_area, l_thick, ...
        P0_base, Q_process, R_fixed, ...
        lambda_step_max, xi1_step_max);
        
    result_now.study_type = 'R_sensitivity';
    result_now.scale = r_scale;
    result_now.Q_process = Q_process;
    result_now.R_used = R_fixed;
    
    % 【绝对防爆修改 4】：大括号存入
    R_results{s} = result_now; 
    
    R_metrics = [R_metrics; ...
        table(r_scale, result_now.rmse_v_prior, result_now.rmse_v_post, ...
              result_now.mae_v_prior, result_now.mae_v_post, ...
              result_now.maxae_v_prior, result_now.maxae_v_post, ...
              'VariableNames', {'R_scale','RMSE_prior','RMSE_post','MAE_prior','MAE_post','MaxAE_prior','MaxAE_post'})]; %#ok<AGROW>
              
    save(fullfile(out_dir, sprintf('RESULT_Rscale_%g.mat', r_scale)), 'result_now');
end
writetable(R_metrics, fullfile(out_dir, 'R_sensitivity_metrics.csv'));
save(fullfile(out_dir, 'R_sensitivity_all_results.mat'), 'R_results', 'R_metrics');

disp('>>> UKF sensitivity analysis for Q and R has been completed and saved.');

%% ==================== Auxiliary functions ====================
function result_now = run_ukf_once(...
    N_windows, seq_len, y_pred_all, ...
    I_test, T_test, Pair_test, PH2_test, V_test, ...
    A_area, l_thick, ...
    P0_base, Q_process, R_fixed, ...
    lambda_step_max, xi1_step_max)

    x_est = [10.0; -0.85];
    P_cov = P0_base;
    
    X_prior_history    = nan(N_windows, 2);
    X_est_history      = nan(N_windows, 2);
    V_prior_history    = nan(N_windows, 1);
    V_post_history     = nan(N_windows, 1);
    R_history          = nan(N_windows, 1);
    innovation_history = nan(N_windows, 1);

    for k = 1:N_windows
        if k < seq_len
            x_prior = x_est;
        else
            x_prior = y_pred_all(k-seq_len+1, :)';
        end
        x_prior(1) = min(max(x_prior(1), 3.0), 14.0);
        x_prior(2) = min(max(x_prior(2), -1.5), -0.5);
        
        X_prior_history(k,:) = x_prior';
        P_prior = P_cov + Q_process;
        
        [x_sigma, Wm, Wc] = generateSigmaPoints_local(x_prior, P_prior);
        z_sigma = zeros(1, size(x_sigma,2));
        for j = 1:size(x_sigma,2)
            z_sigma(j) = fuelCellModel_local(x_sigma(:,j), I_test(k), T_test(k), ...
                                             Pair_test(k), PH2_test(k), A_area, l_thick);
        end
        
        bad_idx = ~isfinite(z_sigma) | abs(z_sigma) > 5;
        if any(bad_idx)
            valid_idx = ~bad_idx & isfinite(z_sigma);
            if any(valid_idx)
                z_sigma(bad_idx) = mean(z_sigma(valid_idx));
            else
                z_sigma(:) = V_test(k);
            end
        end
        
        z_pred = z_sigma * Wm';
        V_prior_history(k) = z_pred;
        
        innovation = V_test(k) - z_pred;
        innovation_history(k) = innovation;
        R_history(k) = R_fixed;
        
        [x_est, P_cov] = ukfUpdate_local(x_sigma, z_sigma, x_prior, z_pred, V_test(k), R_fixed, Wm, Wc, P_prior);
        
        x_est(1) = min(max(x_est(1), 3.0), 14.0);
        x_est(2) = min(max(x_est(2), -1.5), -0.5);
        
        if k > 1
            prev_lambda = X_est_history(k-1,1);
            prev_xi1    = X_est_history(k-1,2);
            if ~isnan(prev_lambda)
                x_est(1) = min(max(x_est(1), prev_lambda - lambda_step_max), ...
                                   prev_lambda + lambda_step_max);
            end
            if ~isnan(prev_xi1)
                x_est(2) = min(max(x_est(2), prev_xi1 - xi1_step_max), ...
                                   prev_xi1 + xi1_step_max);
            end
        end
        
        X_est_history(k,:) = x_est';
        
        V_post_history(k) = fuelCellModel_local(x_est, I_test(k), T_test(k), ...
                                                Pair_test(k), PH2_test(k), A_area, l_thick);
        if ~isfinite(V_post_history(k)) || abs(V_post_history(k)) > 5
            V_post_history(k) = z_pred;
        end
    end
    
    rmse_v_prior = sqrt(mean((V_test - V_prior_history).^2, 'omitnan'));
    rmse_v_post  = sqrt(mean((V_test - V_post_history).^2, 'omitnan'));
    mae_v_prior  = mean(abs(V_test - V_prior_history), 'omitnan');
    mae_v_post   = mean(abs(V_test - V_post_history), 'omitnan');
    maxae_v_prior = max(abs(V_test - V_prior_history));
    maxae_v_post  = max(abs(V_test - V_post_history));
    
    result_now = struct();
    result_now.V_test = V_test;
    result_now.V_prior = V_prior_history;
    result_now.V_post  = V_post_history;
    result_now.lambda_prior = X_prior_history(:,1);
    result_now.lambda_post  = X_est_history(:,1);
    result_now.xi1_prior = X_prior_history(:,2);
    result_now.xi1_post  = X_est_history(:,2);
    result_now.innovation_history = innovation_history;
    result_now.R_history = R_history;
    
    result_now.rmse_v_prior = rmse_v_prior;
    result_now.rmse_v_post  = rmse_v_post;
    result_now.mae_v_prior  = mae_v_prior;
    result_now.mae_v_post   = mae_v_post;
    result_now.maxae_v_prior = maxae_v_prior;
    result_now.maxae_v_post  = maxae_v_post;
end

function [X, Wm, Wc] = generateSigmaPoints_local(x, P)
    n = length(x);
    alpha = 1e-3; kappa = 0; beta = 2;
    lambda_ukf = alpha^2 * (n + kappa) - n;
    c = n + lambda_ukf;
    Wm = zeros(1, 2*n+1);
    Wc = zeros(1, 2*n+1);
    Wm(1) = lambda_ukf / c;
    Wc(1) = lambda_ukf / c + (1 - alpha^2 + beta);
    for i = 2:2*n+1
        Wm(i) = 1 / (2*c);
        Wc(i) = 1 / (2*c);
    end
    P = (P + P') / 2;
    [sP, p_err] = chol(c * P, 'lower');
    if p_err > 0
        P = P + eye(n) * 1e-6;
        P = (P + P') / 2;
        [sP, p_err] = chol(c * P, 'lower');
        if p_err > 0
            sP = chol(c * (P + eye(n)*1e-4), 'lower');
        end
    end
    X = [x, repmat(x,1,n) + sP, repmat(x,1,n) - sP];
end

function [x, P] = ukfUpdate_local(x_sigma, z_sigma, x_prior, z_pred, z_actual, R, Wm, Wc, P_prior)
    n = size(x_sigma, 1);
    P_zz = R;
    P_xz = zeros(n, 1);
    for i = 1:size(x_sigma, 2)
        dz = z_sigma(i) - z_pred;
        dx = x_sigma(:,i) - x_prior;
        P_zz = P_zz + Wc(i) * (dz * dz');
        P_xz = P_xz + Wc(i) * (dx * dz');
    end
    P_zz = max(P_zz, 1e-8);
    K = P_xz / P_zz;
    x = x_prior + K * (z_actual - z_pred);
    P = P_prior - K * P_zz * K';
    P = (P + P') / 2;
    [V,D] = eig(P);
    D = diag(max(diag(D), 1e-10));
    P = V * D * V';
    P = (P + P') / 2;
end

function V = fuelCellModel_local(x, I, T, Pair, PH2, A, l)
    lambda = x(1);
    xi1    = x(2);
    T    = max(T, 273.15);
    Pair = max(Pair, 1e-6);
    PH2  = max(PH2, 1e-6);
    A    = max(A, 1e-3);
    I    = max(I, 0);
    Po2 = max(0.21 * Pair, 1e-6);
    log_arg_nernst = max(PH2 * sqrt(Po2), 1e-12);
    E_nernst = 1.229 - 0.85e-3*(T - 298.15) + 4.3085e-5*T*log(log_arg_nernst);
    xi2 = 0.00312; xi3 = 7.4e-5; xi4 = -1.87e-4;
    C_O2 = Po2 / (5.08e6 * exp(-498/T));
    C_O2 = max(C_O2, 1e-12);
    I_safe = max(I, 1e-6);
    V_act = -(xi1 + xi2*T + xi3*T*log(C_O2) + xi4*T*log(I_safe));
    J = max(I / A, 1e-6);
    num = 181.6 * (1 + 0.03*J + 0.062*(T/303)^2 * J^2.5);
    den_raw = (lambda - 0.634 - 3*J) * exp(4.18*(T-303)/T);
    den = max(den_raw, 1e-3);
    V_ohm = I * (num / den * l) / A;
    j_ratio = min(J / 1.5, 0.999);
    conc_arg = max(1 - j_ratio, 1e-6);
    V_conc = -0.016 * log(conc_arg);
    V = E_nernst - V_act - V_ohm - V_conc;
    if ~isfinite(V) || abs(V) > 5
        V = nan;
    end
end