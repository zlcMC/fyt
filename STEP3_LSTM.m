%% FC2 Ablation Test: plain LSTM only
clear; clc; close all;

%% ==================== 0. Load plain LSTM model ====================
disp('0. Loading plain LSTM model...');
model_file = 'FC1_TS_LSTM_Attention_Model.mat';
if ~isfile(model_file)
    error('Cannot find');
end

S_model = load(model_file);

if isfield(S_model, 'ts_lstm_net')
    state_net = S_model.ts_lstm_net;
else
    error('ts_lstm_net not found in model file.');
end

x_mean = S_model.x_mean;
x_std  = S_model.x_std;
y_mean = S_model.y_mean;
y_std  = S_model.y_std;

if canUseGPU
    exec_env = 'gpu';
else
    exec_env = 'auto';
end

%% ==================== 1. Load FC2 data ====================
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

valid_mask_raw = ~isnan(I_raw) & isfinite(I_raw) & ...
                 ~isnan(J_raw) & isfinite(J_raw) & ...
                 ~isnan(T_C_raw) & isfinite(T_C_raw) & ...
                 ~isnan(P_air_raw) & isfinite(P_air_raw) & ...
                 ~isnan(P_H2_raw) & isfinite(P_H2_raw) & ...
                 ~isnan(Utot_raw) & isfinite(Utot_raw);

I_raw     = I_raw(valid_mask_raw);
J_raw     = J_raw(valid_mask_raw);
T_C_raw   = T_C_raw(valid_mask_raw);
P_air_raw = P_air_raw(valid_mask_raw);
P_H2_raw  = P_H2_raw(valid_mask_raw);
Utot_raw  = Utot_raw(valid_mask_raw);

n_cells = 5;
valid_area_idx = find(abs(J_raw) > 1e-6);
idx_use = valid_area_idx(1:min(100, length(valid_area_idx)));
A_area = median(I_raw(idx_use) ./ J_raw(idx_use));
if isnan(A_area) || ~isfinite(A_area) || A_area < 10
    A_area = 100.0;
end

l_thick = 0.00508;
V_actual_raw = Utot_raw / n_cells;
T_K_raw = T_C_raw + 273.15;
P_air_atm = max(P_air_raw / 1013.25, 1e-3);
P_H2_atm  = max(P_H2_raw  / 1013.25, 1e-3);

%% ==================== 2. Downsample and normalize ====================
disp('2. Downsampling and normalization...');
window_size = 60;
N_windows = floor(length(I_raw) / window_size);

I_test = zeros(N_windows,1);
T_test = zeros(N_windows,1);
Pair_test = zeros(N_windows,1);
PH2_test = zeros(N_windows,1);
V_test = zeros(N_windows,1);

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
x_std_rep  = repmat(x_std, size(X_test_raw,1), 1);
x_std_rep(abs(x_std_rep) < 1e-12) = 1.0;
X_test_norm = (X_test_raw - x_mean_rep) ./ x_std_rep;

%% ==================== 3. State prediction only ====================
disp('3. Running TS LSTM only...');
seq_len = 80;
valid_len = N_windows - seq_len + 1;

X_all_cells = cell(valid_len, 1);
for k = seq_len:N_windows
    X_all_cells{k-seq_len+1} = X_test_norm(k-seq_len+1:k, :)';
end

y_pred_all_norm = predict(state_net, X_all_cells, ...
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
y_std_rep  = repmat(y_std, size(y_pred_all_norm,1), 1);
y_std_rep(abs(y_std_rep) < 1e-12) = 1.0;
y_pred_all = y_pred_all_norm .* y_std_rep + y_mean_rep;

y_pred_all(:,1) = min(max(y_pred_all(:,1), 3.0), 14.0);
y_pred_all(:,2) = min(max(y_pred_all(:,2), -1.5), -0.5);

X_prior_history = nan(N_windows, 2);
V_prior_history = nan(N_windows, 1);

for k = 1:N_windows
    if k < seq_len
        x_prior = [10.0; -0.85];
    else
        x_prior = y_pred_all(k-seq_len+1, :)';
    end

    X_prior_history(k,:) = x_prior';

    V_prior_history(k) = fuelCellModel(x_prior, I_test(k), T_test(k), ...
        Pair_test(k), PH2_test(k), A_area, l_thick);
end

rmse_v_prior = sqrt(mean((V_test - V_prior_history).^2, 'omitnan'));
mae_v_prior  = mean(abs(V_test - V_prior_history), 'omitnan');
maxae_v_prior = max(abs(V_test - V_prior_history));

fprintf('\n===== TS LSTM only on FC2 =====\n');
fprintf('Voltage RMSE = %.4f V\n', rmse_v_prior);
fprintf('Voltage MAE  = %.4f V\n', mae_v_prior);
fprintf('Voltage MaxAE = %.4f V\n', maxae_v_prior);

%% ==================== 4. Save results ====================
out_dir = fullfile('消融实验', 'TS_LSTM_only');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

RESULT = struct();
RESULT.method = 'TS-LSTM only';
RESULT.V_test = V_test;
RESULT.V_prior = V_prior_history;
RESULT.lambda_prior = X_prior_history(:,1);
RESULT.xi1_prior    = X_prior_history(:,2);
RESULT.rmse_v_prior = rmse_v_prior;
RESULT.mae_v_prior  = mae_v_prior;
RESULT.maxae_v_prior = maxae_v_prior;

save(fullfile(out_dir, 'RESULT_TS_LSTM_only_FC2.mat'), 'RESULT');

T = table(rmse_v_prior, mae_v_prior, maxae_v_prior, ...
    'VariableNames', {'RMSE','MAE','MaxAE'});
writetable(T, fullfile(out_dir, 'metrics_TS_LSTM_only_FC2.csv'));

disp('>>> plain TSLSTM only FC2 results saved.');

%% ==================== Auxiliary functions ====================
function V = fuelCellModel(x, I, T, Pair, PH2, A, l)
    lambda = x(1); xi1 = x(2);

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