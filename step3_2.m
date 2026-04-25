%% FC1+FC2 第三步：双尺度 AUKF 状态观测器 (稳健改良版)
% 核心逻辑：
% 1. 载入模型：加载 STEP2 训练好的 Attention-TS-LSTM
% 2. 载入数据：加载 FC2 (带高频逆变器纹波 Ripple) 恶劣数据集
% 3. 算法融合：LSTM 提供平滑先验，AUKF 结合实时电压做后验修正
% 4. 智能抗扰：在纹波/异常点出现时，自适应增大 R，但避免过度敏感
clear; clc; close all;

%% ==================== 0. 环境加载 ====================
disp('0. 加载训练好的 TS-LSTM 模型与归一化统计量...');
model_file = 'FC1_TS_LSTM_Attention_Model.mat';
if ~isfile(model_file)
    error('找不到模型文件，请确保 STEP2 运行成功并在当前目录');
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

%% ==================== 1. 加载 FC2 测试数据 (带纹波) ====================
disp('1. 正在加载并拼接 FC2 全生命周期数据 (With Ripples)...');
fc2_folder = 'F:\FC1_FC2_Excel\FC1_FC2_Excel\FC2_With_Ripples_Excel\';
file_names = { [fc2_folder, 'FC2_Ageing_part1.xlsx'], ...
               [fc2_folder, 'FC2_Ageing_part2.csv'] };

data_full = table();

for i = 1:length(file_names)
    fprintf('   读取测试集: %s\n', file_names{i});
    temp_data = readtable(file_names{i}, 'VariableNamingRule', 'preserve');
    vars = temp_data.Properties.VariableNames;

    for j = 1:length(vars)
        col = temp_data.(vars{j});
        if iscell(col) || isstring(col) || ischar(col)
            % 兼容欧洲小数点逗号格式
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

% -------------------- 有效面积估计 --------------------
n_cells = 5;
valid_area_idx = find(abs(J_raw) > 1e-6);
if isempty(valid_area_idx)
    error('J_raw 全部接近 0，无法估计有效面积 A_area');
end

idx_use = valid_area_idx(1:min(100, length(valid_area_idx)));
A_area = median(I_raw(idx_use) ./ J_raw(idx_use));

% 防御性兜底
if isnan(A_area) || ~isfinite(A_area) || A_area < 10
    warning('A_area 异常，已回退到默认值 100.0 cm^2');
    A_area = 100.0;
end

l_thick = 0.00508;

V_actual_raw = Utot_raw / n_cells;
T_K_raw      = T_C_raw + 273.15;
P_air_atm    = max(P_air_raw / 1013.25, 1e-3);
P_H2_atm     = max(P_H2_raw  / 1013.25, 1e-3);

fprintf('   A_area = %.4f cm^2\n', A_area);
fprintf('   原始数据长度 = %d\n', length(I_raw));

%% ==================== 2. 测试集降采样与标准化 ====================
disp('2. 正在进行 FC2 测试集降采样与标准化...');

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

% 避免除零
x_std_rep(abs(x_std_rep) < 1e-12) = 1.0;
X_test_norm = (X_test_raw - x_mean_rep) ./ x_std_rep;

%% ==================== 3. AUKF 状态观测循环 ====================
disp('3. 初始化 AUKF 参数与状态变量...');

% -------------------- 初值 --------------------
x_est = [10.0; -0.85];
P_cov = diag([0.08, 0.008]);

% -------------------- 改良过程噪声 --------------------
% 比原版稍大，让状态更有可调性，但又不至于过抖
Q_process = diag([5e-4, 5e-6]);

% -------------------- 改良观测噪声自适应参数 --------------------
R_base = 0.008;
R_min  = R_base;
R_max  = 20 * R_base;

ripple_win  = 20;    % 变长，减弱局部波动敏感性
ripple_gain = 0.30;  % 降低增益，防止 R 过激增大
R_smooth_alpha = 0.85; % 对 R 做指数平滑，越接近1越平稳

seq_len = 80;

% -------------------- 状态变化率限制 --------------------
lambda_step_max = 0.03;
xi1_step_max    = 0.002;

% -------------------- 历史记录 --------------------
X_prior_history      = nan(N_windows, 2);
X_est_history        = nan(N_windows, 2);
V_prior_history      = nan(N_windows, 1);
V_post_history       = nan(N_windows, 1);
R_adaptive_history   = nan(N_windows, 1);
innovation_history   = nan(N_windows, 1);
innovation_ratio_hist = nan(N_windows,1);

disp('   -> 正在执行 TS-LSTM 全序列先验预测 (Batch Prediction)...');

valid_len = N_windows - seq_len + 1;
if valid_len <= 0
    error('N_windows 小于 seq_len，无法构造测试序列');
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

% 给先验也加物理范围裁剪
y_pred_all(:,1) = min(max(y_pred_all(:,1), 3.0), 14.0);
y_pred_all(:,2) = min(max(y_pred_all(:,2), -1.5), -0.5);

disp('   -> 启动 AUKF 逐点滤波更新...');

R_prev = R_base;

for k = 1:N_windows
    % ---------- 1) 先验 ----------
    if k < seq_len
        x_prior = x_est;
    else
        x_prior = y_pred_all(k-seq_len+1, :)';
    end

    % 先验范围保护
    x_prior(1) = min(max(x_prior(1), 3.0), 14.0);
    x_prior(2) = min(max(x_prior(2), -1.5), -0.5);

    X_prior_history(k, :) = x_prior';
    P_prior = P_cov + Q_process;

    % ---------- 2) UKF sigma 点预测 ----------
    [x_sigma, Wm, Wc] = generateSigmaPoints(x_prior, P_prior);

    z_sigma = zeros(1, size(x_sigma,2));
    for j = 1:size(x_sigma,2)
        z_sigma(j) = fuelCellModel(x_sigma(:,j), I_test(k), T_test(k), ...
                                   Pair_test(k), PH2_test(k), A_area, l_thick);
    end

    % 若有异常值则进行保护
    bad_idx = ~isfinite(z_sigma) | abs(z_sigma) > 5;
    if any(bad_idx)
        z_sigma(bad_idx) = nanmedian(z_sigma(~bad_idx));
        if all(~isfinite(z_sigma))
            z_sigma(:) = V_test(k);
        end
    end

    z_pred = z_sigma * Wm';
    V_prior_history(k) = z_pred;

    % ---------- 3) 创新 ----------
    innovation = V_test(k) - z_pred;
    innovation_history(k) = innovation;

    if k == 1
        local_std = max(abs(innovation), 1e-4);
    else
        win_idx = max(1, k-ripple_win+1):k;
        local_std = std(innovation_history(win_idx), 'omitnan');
        local_std = max(local_std, 1e-4);
    end

    innovation_ratio = abs(innovation) / local_std;
    innovation_ratio = min(innovation_ratio, 4.0);   % 限幅，防止 R 爆跳
    innovation_ratio_hist(k) = innovation_ratio;

    % 原始自适应 R
    R_raw = R_base * (1 + ripple_gain * max(0, innovation_ratio - 1)^2);
    R_raw = min(max(R_raw, R_min), R_max);

    % 平滑后的自适应 R
    if k == 1
        R_k = R_raw;
    else
        R_k = R_smooth_alpha * R_prev + (1 - R_smooth_alpha) * R_raw;
    end
    R_k = min(max(R_k, R_min), R_max);
    R_prev = R_k;
    R_adaptive_history(k) = R_k;

    % ---------- 4) UKF 更新 ----------
    [x_est, P_cov] = ukfUpdate(x_sigma, z_sigma, x_prior, z_pred, V_test(k), R_k, Wm, Wc, P_prior);

    % ---------- 5) 状态约束 ----------
    x_est(1) = min(max(x_est(1), 3.0), 14.0);
    x_est(2) = min(max(x_est(2), -1.5), -0.5);

    % ---------- 6) 状态变化率限制 ----------
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

    % ---------- 7) 后验输出 ----------
    V_post_history(k) = fuelCellModel(x_est, I_test(k), T_test(k), ...
                                      Pair_test(k), PH2_test(k), A_area, l_thick);

    if ~isfinite(V_post_history(k)) || abs(V_post_history(k)) > 5
        V_post_history(k) = z_pred;
    end
end

%% ==================== 4. 绘图与评估 ====================
disp('4. 绘图与评估...');

rmse_v_prior = sqrt(mean((V_test - V_prior_history).^2, 'omitnan'));
rmse_v_post  = sqrt(mean((V_test - V_post_history).^2,  'omitnan'));

mae_v_prior = mean(abs(V_test - V_prior_history), 'omitnan');
mae_v_post  = mean(abs(V_test - V_post_history),  'omitnan');

fprintf('\n========= 结果评估 =========\n');
fprintf('TS-LSTM 先验: RMSE = %.6f V, MAE = %.6f V\n', rmse_v_prior, mae_v_prior);
fprintf('AUKF 后验   : RMSE = %.6f V, MAE = %.6f V\n', rmse_v_post,  mae_v_post);

figure('Name', '方向三：TS-LSTM + AUKF 恶劣纹波工况测试 (FC2, 稳健改良版)', ...
       'Position', [100 60 1400 980]);

subplot(5,1,1);
plot(V_test, 'k', 'LineWidth', 1); hold on;
plot(V_prior_history, 'b--', 'LineWidth', 1.0);
plot(V_post_history, 'r-', 'LineWidth', 1.2);
title(sprintf('电压跟踪性能 (FC2 带高频纹波) | TS-LSTM先验 RMSE=%.4f | AUKF后验 RMSE=%.4f', ...
      rmse_v_prior, rmse_v_post));
ylabel('Voltage (V)');
legend('实际传感器', 'TS-LSTM 先验', 'AUKF 后验', 'Location', 'best');
grid on;

subplot(5,1,2);
plot(X_prior_history(:,1), 'b--', 'LineWidth', 1.0); hold on;
plot(X_est_history(:,1), 'r-', 'LineWidth', 1.2);
title('估算的膜含水量 \lambda (快变状态)');
ylabel('\lambda');
legend('LSTM先验', 'AUKF修正', 'Location', 'best');
grid on;

subplot(5,1,3);
plot(X_prior_history(:,2), 'm--', 'LineWidth', 1.0); hold on;
plot(X_est_history(:,2), 'k-', 'LineWidth', 1.2);
title('估算的催化剂老化参数 \xi_1 (慢变状态)');
ylabel('\xi_1');
legend('LSTM先验', 'AUKF修正', 'Location', 'best');
grid on;

subplot(5,1,4);
yyaxis left;
plot(abs(innovation_history), 'b-', 'LineWidth', 1.0);
ylabel('|电压残差 (V)|');

yyaxis right;
plot(R_adaptive_history, 'g-', 'LineWidth', 1.2);
ylabel('自适应 R');

title('AUKF 抗扰机制：残差与平滑后的自适应 R');
xlabel('时间窗口');
grid on;

subplot(5,1,5);
plot(innovation_ratio_hist, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.0); hold on;
yline(1.0, '--k', '阈值=1');
title('创新比值 |innovation| / local\_std');
xlabel('时间窗口');
ylabel('ratio');
grid on;

%% ==================== 辅助函数集 ====================

function [X, Wm, Wc] = generateSigmaPoints(x, P)
    n = length(x);

    alpha = 1e-3;
    kappa = 0;
    beta  = 2;

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

function [x, P] = ukfUpdate(x_sigma, z_sigma, x_prior, z_pred, z_actual, R, Wm, Wc, P_prior)
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
    [V, D] = eig(P);
    D = diag(max(diag(D), 1e-10));
    P = V * D * V';
    P = (P + P') / 2;
end

function V = fuelCellModel(x, I, T, Pair, PH2, A, l)
    lambda = x(1);
    xi1    = x(2);

    % -------------------- 数值保护 --------------------
    T    = max(T, 273.15);
    Pair = max(Pair, 1e-6);
    PH2  = max(PH2, 1e-6);
    A    = max(A, 1e-3);
    I    = max(I, 0);

    Po2 = max(0.21 * Pair, 1e-6);

    % -------------------- Nernst 电压 --------------------
    log_arg_nernst = max(PH2 * sqrt(Po2), 1e-12);
    E_nernst = 1.229 - 0.85e-3*(T - 298.15) + 4.3085e-5*T*log(log_arg_nernst);

    % -------------------- 活化极化 --------------------
    xi2 = 0.00312;
    xi3 = 7.4e-5;
    xi4 = -1.87e-4;

    C_O2 = Po2 / (5.08e6 * exp(-498/T));
    C_O2 = max(C_O2, 1e-12);

    I_safe = max(I, 1e-6);
    V_act = -(xi1 + xi2*T + xi3*T*log(C_O2) + xi4*T*log(I_safe));

    % -------------------- 欧姆极化 --------------------
    J = max(I / A, 1e-6);

    num = 181.6 * (1 + 0.03*J + 0.062*(T/303)^2 * J^2.5);

    den_raw = (lambda - 0.634 - 3*J) * exp(4.18*(T-303)/T);
    den = max(den_raw, 1e-3);   % 比 1e-8 更保守稳定

    V_ohm = I * (num / den * l) / A;

    % -------------------- 浓差极化 --------------------
    j_ratio = min(J / 1.5, 0.999);
    conc_arg = max(1 - j_ratio, 1e-6);
    V_conc = -0.016 * log(conc_arg);

    % -------------------- 总电压 --------------------
    V = E_nernst - V_act - V_ohm - V_conc;

    % 防止异常值污染 UKF
    if ~isfinite(V) || abs(V) > 5
        V = nan;
    end
end
%% ==================== 5. Journal-quality visualization and data saving ====================
out_dir = 'results_step3';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultLineLineWidth', 1.5);

t = (1:N_windows)';

rmse_v_prior = sqrt(mean((V_test - V_prior_history).^2, 'omitnan'));
rmse_v_post  = sqrt(mean((V_test - V_post_history).^2, 'omitnan'));
mae_v_prior  = mean(abs(V_test - V_prior_history), 'omitnan');
mae_v_post   = mean(abs(V_test - V_post_history), 'omitnan');

%% -------- Fig. STEP3-1: Overall tracking performance --------
fig1 = figure('Color','w','Position',[80 60 1100 800]);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t, V_test, 'k-', 'LineWidth', 1.2); hold on;
plot(t, V_prior_history, '--', 'Color', [0 0.45 0.74], 'LineWidth', 1.2);
plot(t, V_post_history, '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);
ylabel('Voltage (V)');
title(sprintf('Voltage tracking under ripple disturbance (Prior RMSE = %.4f V, Posterior RMSE = %.4f V)', rmse_v_prior, rmse_v_post));
legend({'Measured voltage','TS-LSTM prior','AUKF posterior'}, 'Location','best', 'Box','off');
grid on; box on; xlim([1 N_windows]);

nexttile;
plot(t, X_prior_history(:,1), '--', 'Color', [0 0.45 0.74], 'LineWidth', 1.2); hold on;
plot(t, X_est_history(:,1), '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);
ylabel('\lambda');
title('Estimated fast state');
legend({'Prior','Posterior'}, 'Location','best', 'Box','off');
grid on; box on; xlim([1 N_windows]);

nexttile;
plot(t, X_prior_history(:,2), '--', 'Color', [0.49 0.18 0.56], 'LineWidth', 1.2); hold on;
plot(t, X_est_history(:,2), 'k-', 'LineWidth', 1.4);
ylabel('\xi_1');
xlabel('Time window');
title('Estimated slow state');
legend({'Prior','Posterior'}, 'Location','best', 'Box','off');
grid on; box on; xlim([1 N_windows]);

exportgraphics(fig1, fullfile(out_dir, 'Fig_STEP3_Main_Tracking.png'), 'Resolution', 600);
exportgraphics(fig1, fullfile(out_dir, 'Fig_STEP3_Main_Tracking.pdf'), 'ContentType', 'vector');

%% -------- Fig. STEP3-2: Adaptive filtering mechanism --------
fig2 = figure('Color','w','Position',[100 80 1100 700]);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

nexttile;
plot(t, abs(innovation_history), 'Color', [0 0.45 0.74], 'LineWidth', 1.2);
ylabel('|Innovation| (V)');
title('Innovation magnitude');
grid on; box on; xlim([1 N_windows]);

nexttile;
plot(t, R_adaptive_history, 'Color', [0.47 0.67 0.19], 'LineWidth', 1.5);
ylabel('Adaptive R');
title('Adaptive measurement-noise covariance');
grid on; box on; xlim([1 N_windows]);

nexttile;
plot(t, innovation_ratio_hist, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2); hold on;
yline(1, 'k--', 'Threshold = 1', 'LineWidth', 1.0, 'LabelHorizontalAlignment', 'left');
ylabel('Ratio');
xlabel('Time window');
title('Normalized innovation ratio');
grid on; box on; xlim([1 N_windows]);

exportgraphics(fig2, fullfile(out_dir, 'Fig_STEP3_Adaptive_Mechanism.png'), 'Resolution', 600);
exportgraphics(fig2, fullfile(out_dir, 'Fig_STEP3_Adaptive_Mechanism.pdf'), 'ContentType', 'vector');

%% -------- Fig. STEP3-3: Error distribution --------
err_prior = V_prior_history - V_test;
err_post  = V_post_history - V_test;

fig3 = figure('Color','w','Position',[100 80 1000 420]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
histogram(err_prior, 50, 'Normalization', 'pdf', 'FaceColor', [0 0.45 0.74], 'FaceAlpha', 0.45, 'EdgeColor', 'none'); hold on;
histogram(err_post,  50, 'Normalization', 'pdf', 'FaceColor', [0.85 0.33 0.10], 'FaceAlpha', 0.45, 'EdgeColor', 'none');
xlabel('Voltage error (V)');
ylabel('PDF');
title('Error distribution');
legend({'Prior error','Posterior error'}, 'Location','best', 'Box','off');
grid on; box on;

nexttile;
boxplot([abs(err_prior), abs(err_post)], 'Labels', {'Prior','Posterior'});
ylabel('Absolute voltage error (V)');
title('Absolute error comparison');
grid on; box on;

exportgraphics(fig3, fullfile(out_dir, 'Fig_STEP3_Error_Statistics.png'), 'Resolution', 600);
exportgraphics(fig3, fullfile(out_dir, 'Fig_STEP3_Error_Statistics.pdf'), 'ContentType', 'vector');

%% -------- Fig. STEP3-4: Zoomed-in transient windows --------
% Pick several representative regions manually or automatically
zoom_sets = {
    max(1,1100-50):min(N_windows,1100+50), ...
    max(1,1420-50):min(N_windows,1420+50), ...
    max(1,1750-50):min(N_windows,1750+50)
};

fig4 = figure('Color','w','Position',[80 60 1200 900]);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

for i = 1:3
    zr = zoom_sets{i};
    nexttile;
    plot(zr, V_test(zr), 'k-', 'LineWidth', 1.2); hold on;
    plot(zr, V_prior_history(zr), '--', 'Color', [0 0.45 0.74], 'LineWidth', 1.2);
    plot(zr, V_post_history(zr), '-', 'Color', [0.85 0.33 0.10], 'LineWidth', 1.5);
    ylabel('Voltage (V)');
    title(sprintf('Zoomed-in transient region %d', i));
    legend({'Measured','Prior','Posterior'}, 'Location','best', 'Box','off');
    grid on; box on;
end
xlabel('Time window');

exportgraphics(fig4, fullfile(out_dir, 'Fig_STEP3_Zoomed_Transients.png'), 'Resolution', 600);
exportgraphics(fig4, fullfile(out_dir, 'Fig_STEP3_Zoomed_Transients.pdf'), 'ContentType', 'vector');

%% -------- Save structured data for paper --------
STEP3_RESULTS = struct();
STEP3_RESULTS.time_window = t;
STEP3_RESULTS.V_measured  = V_test;
STEP3_RESULTS.V_prior     = V_prior_history;
STEP3_RESULTS.V_post      = V_post_history;
STEP3_RESULTS.lambda_prior = X_prior_history(:,1);
STEP3_RESULTS.lambda_post  = X_est_history(:,1);
STEP3_RESULTS.xi1_prior    = X_prior_history(:,2);
STEP3_RESULTS.xi1_post     = X_est_history(:,2);
STEP3_RESULTS.innovation   = innovation_history;
STEP3_RESULTS.adaptive_R   = R_adaptive_history;
STEP3_RESULTS.innovation_ratio = innovation_ratio_hist;
STEP3_RESULTS.rmse_v_prior = rmse_v_prior;
STEP3_RESULTS.rmse_v_post  = rmse_v_post;
STEP3_RESULTS.mae_v_prior  = mae_v_prior;
STEP3_RESULTS.mae_v_post   = mae_v_post;

save(fullfile(out_dir, 'STEP3_RESULTS_FOR_PAPER.mat'), 'STEP3_RESULTS');