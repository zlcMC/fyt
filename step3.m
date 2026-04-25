%% FC1+FC2 第三步：双尺度 AUKF 状态观测器 (方向三论文最终稳定版)
% 核心逻辑：
% 1. 载入模型：加载 STEP2 训练好的 Attention-TS-LSTM
% 2. 载入数据：加载 FC2 (带高频逆变器纹波 Ripple) 恶劣数据集
% 3. 算法融合：LSTM 负责提供粗略的、不带毛刺的“物理发展趋势”（先验）
%             AUKF 负责结合实时的电压读数，把细节给纠正回来（后验）
% 4. 智能抗扰：遇到极其恶劣的纹波时，AUKF 会自动闭上眼睛（增大 R），死死抱住 LSTM 不松手。
clear; clc; close all;

%% ==================== 0. 环境加载 ====================
disp('0. 加载训练好的 TS-LSTM 模型与归一化统计量...');
model_file = 'FC1_TS_LSTM_Attention_Model.mat';
if ~isfile(model_file)
    error('找不到模型文件，请确保 STEP2 运行成功并在当前目录');
end
S_model = load(model_file);
ts_lstm_net = S_model.ts_lstm_net;
x_mean = S_model.x_mean; x_std = S_model.x_std;
y_mean = S_model.y_mean; y_std = S_model.y_std;

if canUseGPU; exec_env = 'gpu'; else; exec_env = 'auto'; end

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
            % 强行替换欧洲 CSV 的逗号小数点，彻底解决 NaN 问题
            col_str = strrep(string(col), ',', '.'); 
            temp_data.(vars{j}) = str2double(col_str);  
        else
            temp_data.(vars{j}) = double(col);    
        end
    end
    data_full = [data_full; temp_data];
end

try
    I_raw = data_full.('I (A)'); T_C_raw = data_full.('ToutWAT (ｰC)');
    P_air_raw = data_full.('PoutAIR (mbara)'); P_H2_raw = data_full.('PoutH2 (mbara)');
    Utot_raw = data_full.('Utot (V)'); J_raw = data_full.('J (A/cmｲ)');
catch
    I_raw = data_full{:,9}; J_raw = data_full{:,8}; T_C_raw = data_full{:,15};
    P_air_raw = data_full{:,17}; P_H2_raw = data_full{:,18}; Utot_raw = data_full{:,7};
end

valid_mask_raw = ~isnan(I_raw) & isfinite(I_raw) & ~isnan(J_raw) & isfinite(J_raw) & ...
                 ~isnan(T_C_raw) & isfinite(T_C_raw) & ~isnan(P_air_raw) & isfinite(P_air_raw) & ...
                 ~isnan(P_H2_raw) & isfinite(P_H2_raw) & ~isnan(Utot_raw) & isfinite(Utot_raw);
I_raw = I_raw(valid_mask_raw); J_raw = J_raw(valid_mask_raw); T_C_raw = T_C_raw(valid_mask_raw);
P_air_raw = P_air_raw(valid_mask_raw); P_H2_raw = P_H2_raw(valid_mask_raw); Utot_raw = Utot_raw(valid_mask_raw);

% -------------------------------------------------------------------------
% 【严格保留原逻辑】：使用中位数去寻找真实有效面积 A
% -------------------------------------------------------------------------
n_cells = 5;
valid_area_idx = find(abs(J_raw) > 1e-6); % 提取非零电流区间
idx_use = valid_area_idx(1:min(100, length(valid_area_idx))); % 补回你遗失的索引定义
A_area = median(I_raw(idx_use) ./ J_raw(idx_use));

% 【防爆炸补丁】：由于FC2有动态停车，若算出的面积等于0或NaN，强制兜底防止10^21爆炸
if isnan(A_area) || A_area < 10
    A_area = 100.0;
end

l_thick = 0.00508;
V_actual_raw = Utot_raw / n_cells;
T_K_raw = T_C_raw + 273.15;
P_air_atm = max(P_air_raw / 1013.25, 1e-3); 
P_H2_atm  = max(P_H2_raw  / 1013.25, 1e-3);

%% ==================== 2. 测试集降采样与标准化 ====================
disp('2. 正在进行 FC2 测试集降采样与标准化...');
window_size = 60;
N_windows = floor(length(I_raw) / window_size);
I_test = zeros(N_windows,1); T_test = zeros(N_windows,1);
Pair_test = zeros(N_windows,1); PH2_test = zeros(N_windows,1); V_test = zeros(N_windows,1);

for i = 1:N_windows
    idx = (i-1)*window_size+1 : i*window_size;
    I_test(i) = mean(I_raw(idx)); T_test(i) = mean(T_K_raw(idx));
    Pair_test(i) = mean(P_air_atm(idx)); PH2_test(i) = mean(P_H2_atm(idx)); V_test(i) = mean(V_actual_raw(idx));
end

X_test_raw = [I_test, T_test, Pair_test, PH2_test];
x_mean_rep = repmat(x_mean, size(X_test_raw, 1), 1);
x_std_rep  = repmat(x_std, size(X_test_raw, 1), 1);
X_test_norm = (X_test_raw - x_mean_rep) ./ x_std_rep;

%% ==================== 3. AUKF 状态观测循环 ====================
disp('3. 初始化 AUKF 参数与状态变量...');
x_est = [10.0; -0.85]; 
P_cov = diag([0.1, 0.01]); 
Q_process = diag([1e-4, 1e-6]); 

R_base = 0.01; R_min = R_base; R_max = 50 * R_base;
ripple_win = 10; ripple_gain = 1.0; 
seq_len = 80;

X_prior_history = nan(N_windows, 2); X_est_history = nan(N_windows, 2);
V_prior_history = nan(N_windows, 1); V_post_history = nan(N_windows, 1);
R_adaptive_history = nan(N_windows, 1); innovation_history = nan(N_windows, 1);

disp('   -> 正在执行 TS-LSTM 全序列先验预测 (Batch Prediction)...');
valid_len = N_windows - seq_len + 1;
X_all_cells = cell(valid_len, 1);
for k = seq_len:N_windows
    X_all_cells{k-seq_len+1} = X_test_norm(k-seq_len+1:k, :)';
end
y_pred_all_norm = predict(ts_lstm_net, X_all_cells, 'ExecutionEnvironment', exec_env, 'MiniBatchSize', 256);

if iscell(y_pred_all_norm)
    temp_pred_mat = zeros(length(y_pred_all_norm), 2);
    for c_idx = 1:length(y_pred_all_norm)
        temp_pred_mat(c_idx, :) = y_pred_all_norm{c_idx}(:)';
    end
    y_pred_all_norm = temp_pred_mat;
end

y_mean_rep = repmat(y_mean, size(y_pred_all_norm, 1), 1);
y_std_rep  = repmat(y_std, size(y_pred_all_norm, 1), 1);
y_pred_all = y_pred_all_norm .* y_std_rep + y_mean_rep; 

disp('   -> 启动 AUKF 逐点滤波更新...');
for k = 1:N_windows
    if k < seq_len
        x_prior = x_est;
    else
        x_prior = y_pred_all(k-seq_len+1, :)';
    end
    X_prior_history(k, :) = x_prior';
    P_prior = P_cov + Q_process;
    
    [x_sigma, Wm, Wc] = generateSigmaPoints(x_prior, P_prior);
    z_sigma = zeros(1, size(x_sigma, 2));
    for j = 1:size(x_sigma, 2)
        z_sigma(j) = fuelCellModel(x_sigma(:,j), I_test(k), T_test(k), Pair_test(k), PH2_test(k), A_area, l_thick);
    end
    z_pred = z_sigma * Wm'; 
    V_prior_history(k) = z_pred;
    
    innovation = V_test(k) - z_pred; 
    innovation_history(k) = innovation;
    
    if k == 1
        local_std = abs(innovation) + 1e-6;
    else
        win_idx = max(1, k-ripple_win+1):k;
        local_std = max(std(innovation_history(win_idx)), 1e-6);
    end
    
    innovation_ratio = abs(innovation) / local_std;
    R_k = R_base * (1 + ripple_gain * max(0, innovation_ratio - 1)^2);
    R_k = min(max(R_k, R_min), R_max); 
    R_adaptive_history(k) = R_k;
    
    [x_est, P_cov] = ukfUpdate(x_sigma, z_sigma, x_prior, z_pred, V_test(k), R_k, Wm, Wc, P_prior);
    
    x_est(1) = max(3.0, min(14.0, x_est(1)));    
    x_est(2) = max(-1.5, min(-0.5, x_est(2)));   
    X_est_history(k, :) = x_est';
    
    V_post_history(k) = fuelCellModel(x_est, I_test(k), T_test(k), Pair_test(k), PH2_test(k), A_area, l_thick);
end

%% ==================== 4. 绘图与评估 ====================
disp('4. 绘图与评估...');
rmse_v_prior = sqrt(mean((V_test - V_prior_history).^2, 'omitnan'));
rmse_v_post  = sqrt(mean((V_test - V_post_history).^2, 'omitnan'));

figure('Name', '方向三：TS-LSTM + AUKF 恶劣纹波工况测试 (FC2)', 'Position', [100 60 1350 950]);
subplot(4,1,1);
plot(V_test, 'k', 'LineWidth', 1); hold on;
plot(V_prior_history, 'b--', 'LineWidth', 1.0); plot(V_post_history, 'r-', 'LineWidth', 1.2);
title(sprintf('电压跟踪性能 (FC2 带高频纹波) | TS-LSTM先验 RMSE=%.4f | AUKF后验 RMSE=%.4f', rmse_v_prior, rmse_v_post));
ylabel('Voltage (V)'); legend('实际传感器', 'TS-LSTM 先验', 'AUKF 后验', 'Location', 'best'); grid on;

subplot(4,1,2);
plot(X_prior_history(:,1), 'b--', 'LineWidth', 1.0); hold on;
plot(X_est_history(:,1), 'r-', 'LineWidth', 1.2);
title('估算的膜含水量 \lambda (快变状态)'); ylabel('\lambda'); legend('LSTM先验', 'AUKF修正', 'Location', 'best'); grid on;

subplot(4,1,3);
plot(X_prior_history(:,2), 'm--', 'LineWidth', 1.0); hold on;
plot(X_est_history(:,2), 'k-', 'LineWidth', 1.2);
title('估算的催化剂老化参数 \xi_1 (慢变状态)'); ylabel('\xi_1'); legend('LSTM先验', 'AUKF修正', 'Location', 'best'); grid on;

subplot(4,1,4);
yyaxis left; plot(abs(innovation_history), 'b-', 'LineWidth', 1.0); ylabel('|电压残差 (V)|');
yyaxis right; plot(R_adaptive_history, 'g-', 'LineWidth', 1.2); ylabel('自适应 R 矩阵权重');
title('AUKF 智能抗扰机制：残差一旦飙升，R 权重立刻自动调大拒绝干扰'); xlabel('时间窗口'); grid on;

%% ==================== 辅助函数集 ====================
function [X, Wm, Wc] = generateSigmaPoints(x, P)
    n = length(x); alpha = 1e-3; kappa = 0; beta = 2;
    lambda = alpha^2 * (n + kappa) - n; c = n + lambda;
    Wm = zeros(1, 2*n+1); Wc = zeros(1, 2*n+1);
    Wm(1) = lambda / c; Wc(1) = lambda / c + (1 - alpha^2 + beta);
    for i = 2:2*n+1; Wm(i) = 1 / (2*c); Wc(i) = 1 / (2*c); end
    P = (P + P') / 2; 
    [sP, p_err] = chol(c * P, 'lower');
    if p_err > 0 
        P = P + eye(n) * 1e-6; sP = chol(c * P, 'lower');
    end
    X = [x, repmat(x, 1, n) + sP, repmat(x, 1, n) - sP];
end

function [x, P] = ukfUpdate(x_sigma, z_sigma, x_prior, z_pred, z_actual, R, Wm, Wc, P_prior)
    n = size(x_sigma, 1); P_zz = R; P_xz = zeros(n, 1);
    for i = 1:size(x_sigma, 2)
        dz = z_sigma(i) - z_pred; dx = x_sigma(:,i) - x_prior;
        P_zz = P_zz + Wc(i) * (dz * dz'); P_xz = P_xz + Wc(i) * (dx * dz');
    end
    P_zz = max(P_zz, 1e-10); K = P_xz / P_zz;
    x = x_prior + K * (z_actual - z_pred);
    P = P_prior - K * P_zz * K';
    P = (P + P') / 2; [V,D] = eig(P); D = diag(max(diag(D), 1e-10)); P = V * D * V';
end

function V = fuelCellModel(x, I, T, Pair, PH2, A, l)
    lambda = x(1); xi1 = x(2); Po2 = 0.21 * Pair;
    E_nernst = 1.229 - 0.85e-3*(T - 298.15) + 4.3085e-5*T*log(PH2 * sqrt(Po2));
    xi2 = 0.00312; xi3 = 7.4e-5; xi4 = -1.87e-4; C_O2 = Po2 / (5.08e6 * exp(-498/T));
    I_safe = max(I, 1e-6); V_act = -(xi1 + xi2*T + xi3*T*log(C_O2) + xi4*T*log(I_safe));
    J = max(I/A, 1e-6);
    num = 181.6*(1 + 0.03*J + 0.062*(T/303)^2*J^2.5); den = max((lambda - 0.634 - 3*J)*exp(4.18*(T-303)/T), 1e-8);
    V_ohm = I * (num/den * l)/A;
    V_conc = -0.016 * log(1 - min(J, 1.5 - 1e-3)/1.5);
    V = E_nernst - V_act - V_ohm - V_conc;
end