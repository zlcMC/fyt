% %% Plot UKF vs AUKF comparison on FC2 (step2-style final version)
% clear; clc; close all;
% 
% %% ==================== 0. Load result files ====================
% base_dir = '消融实验';
% 
% S1 = load(fullfile(base_dir, 'TS_LSTM_UKF',  'RESULT_TS_LSTM_UKF_FC2.mat'));
% S2 = load(fullfile(base_dir, 'TS_LSTM_AUKF', 'RESULT_TS_LSTM_AUKF_FC2.mat'));
% 
% R_ukf  = S1.RESULT;
% R_aukf = S2.RESULT;
% 
% out_dir = fullfile(base_dir, 'UKF_vs_AUKF');
% if ~exist(out_dir, 'dir')
%     mkdir(out_dir);
% end
% 
% %% ==================== 1. Global plotting style ====================
% set(groot, 'defaultAxesFontName', 'Times New Roman');
% set(groot, 'defaultTextFontName', 'Times New Roman');
% set(groot, 'defaultAxesFontSize', 11);
% set(groot, 'defaultLineLineWidth', 1.5);
% 
% %% ==================== Unified color palette (step2-style) ====================
% c_meas = [0.00 0.00 0.00];   % black
% c_ukf  = [0.05 0.42 0.62];   % deeper blue
% c_aukf = [0.58 0.42 0.74];   % muted purple
% 
% V_test = R_ukf.V_test;
% t = (1:length(V_test))';
% 
% %% ==================== 2. Collect metrics ====================
% methods = {'UKF', 'AUKF'};
% rmse_vals  = [R_ukf.rmse_v_post,  R_aukf.rmse_v_post];
% mae_vals   = [R_ukf.mae_v_post,   R_aukf.mae_v_post];
% maxae_vals = [R_ukf.maxae_v_post, R_aukf.maxae_v_post];
% 
% T = table(methods', rmse_vals', mae_vals', maxae_vals', ...
%     'VariableNames', {'Method','RMSE','MAE','MaxAE'});
% writetable(T, fullfile(out_dir, 'UKF_vs_AUKF_Table.csv'));
% 
% %% ==================== 3. Figure 1: Metric comparison ====================
% fig1 = figure('Color','w','Position',[100 80 1000 420]);
% tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
% 
% metric_colors = [c_ukf; c_aukf];
% 
% nexttile;
% b1 = bar(rmse_vals, 'FaceColor', 'flat', 'BarWidth', 0.65);
% for i = 1:2
%     b1.CData(i,:) = metric_colors(i,:);
% end
% set(gca, 'XTick', 1:2, 'XTickLabel', methods);
% ylabel('RMSE (V)');
% title('Posterior RMSE');
% grid on; box on;
% ylim([0, max(rmse_vals)*1.15]);
% for i = 1:2
%     text(i, rmse_vals(i)+0.0001, sprintf('%.4f', rmse_vals(i)), ...
%         'HorizontalAlignment','center', 'FontSize',10);
% end
% 
% nexttile;
% b2 = bar(mae_vals, 'FaceColor', 'flat', 'BarWidth', 0.65);
% for i = 1:2
%     b2.CData(i,:) = metric_colors(i,:);
% end
% set(gca, 'XTick', 1:2, 'XTickLabel', methods);
% ylabel('MAE (V)');
% title('Posterior MAE');
% grid on; box on;
% ylim([0, max(mae_vals)*1.15]);
% for i = 1:2
%     text(i, mae_vals(i)+0.0001, sprintf('%.4f', mae_vals(i)), ...
%         'HorizontalAlignment','center', 'FontSize',10);
% end
% 
% nexttile;
% b3 = bar(maxae_vals, 'FaceColor', 'flat', 'BarWidth', 0.65);
% for i = 1:2
%     b3.CData(i,:) = metric_colors(i,:);
% end
% set(gca, 'XTick', 1:2, 'XTickLabel', methods);
% ylabel('MaxAE (V)');
% title('Posterior MaxAE');
% grid on; box on;
% ylim([0, max(maxae_vals)*1.15]);
% for i = 1:2
%     text(i, maxae_vals(i)+0.001, sprintf('%.4f', maxae_vals(i)), ...
%         'HorizontalAlignment','center', 'FontSize',10);
% end
% 
% drawnow;
% exportgraphics(fig1, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Metrics.png'), 'Resolution', 600);
% print(fig1, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Metrics'), '-dpdf', '-painters');
% 
% %% ==================== 4. Figure 2: Overall tracking comparison ====================
% fig2 = figure('Color','w','Position',[100 80 1200 520]);
% plot(t, V_test, 'Color', c_meas, 'LineWidth', 1.3); hold on;
% plot(t, R_ukf.V_post,  '-', 'Color', c_ukf,  'LineWidth', 1.6);
% plot(t, R_aukf.V_post, '-', 'Color', c_aukf, 'LineWidth', 1.4);
% 
% xlabel('Time window');
% ylabel('Voltage (V)');
% title('Overall comparison between UKF and AUKF on the FC2 ripple dataset');
% legend({'Measured voltage','TS-LSTM + UKF','TS-LSTM + AUKF'}, ...
%        'Location','northwest', 'Box','off');
% grid on; box on;
% xlim([1 length(V_test)]);
% ylim([0.61 0.98]);
% 
% drawnow;
% exportgraphics(fig2, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Overall.png'), 'Resolution', 600);
% print(fig2, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Overall'), '-dpdf', '-painters');
% 
% %% ==================== 5. Figure 3: Zoomed comparison ====================
% zoom_sets = {
%     max(1,1100-50):min(length(V_test),1100+50), ...
%     max(1,1420-50):min(length(V_test),1420+50), ...
%     max(1,1750-50):min(length(V_test),1750+50)
% };
% 
% fig3 = figure('Color','w','Position',[80 60 1200 900]);
% tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
% 
% for i = 1:3
%     zr = zoom_sets{i};
%     nexttile;
%     plot(zr, V_test(zr), 'Color', c_meas, 'LineWidth', 1.3); hold on;
%     plot(zr, R_ukf.V_post(zr),  '-', 'Color', c_ukf,  'LineWidth', 1.6);
%     plot(zr, R_aukf.V_post(zr), '-', 'Color', c_aukf, 'LineWidth', 1.4);
% 
%     ylabel('Voltage (V)');
%     title(sprintf('Zoomed comparison between UKF and AUKF in transient region %d', i));
%     legend({'Measured','UKF','AUKF'}, 'Location','best', 'Box','off');
%     grid on; box on;
% end
% xlabel('Time window');
% 
% drawnow;
% exportgraphics(fig3, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Zoomed.png'), 'Resolution', 600);
% print(fig3, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Zoomed'), '-dpdf', '-painters');
% 
% %% ==================== 6. Figure 4: Absolute error comparison ====================
% err_ukf  = abs(R_ukf.V_post  - V_test);
% err_aukf = abs(R_aukf.V_post - V_test);
% 
% fig4 = figure('Color','w','Position',[120 100 900 430]);
% boxplot([err_ukf(:), err_aukf(:)], ...
%     'Labels', {'UKF', 'AUKF'}, ...
%     'Symbol', 'r+', 'Whisker', 1.5);
% ylabel('Absolute voltage error (V)');
% title('Absolute error comparison between UKF and AUKF');
% grid on; box on;
% 
% drawnow;
% exportgraphics(fig4, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Boxplot.png'), 'Resolution', 600);
% print(fig4, fullfile(out_dir, 'Fig_UKF_vs_AUKF_Boxplot'), '-dpdf', '-painters');
% 
% disp('>>> UKF vs AUKF figures and table have been saved.');

%% Plot final journal-style UKF sensitivity figure (2x3 layout)
% clear; clc; close all;
% 
% base_dir = fullfile('消融实验', 'UKF_Sensitivity_QR');
% 
% Qtab = readtable(fullfile(base_dir, 'Q_sensitivity_metrics.csv'));
% Rtab = readtable(fullfile(base_dir, 'R_sensitivity_metrics.csv'));
% 
% %% ==================== Global style ====================
% set(groot, 'defaultAxesFontName', 'Times New Roman');
% set(groot, 'defaultTextFontName', 'Times New Roman');
% set(groot, 'defaultAxesFontSize', 11);
% set(groot, 'defaultLineLineWidth', 1.6);
% 
% % Unified color palette
% c_rmse = [0.12 0.47 0.71];   % blue
% c_mae  = [0.00 0.40 0.52];   % blue-teal
% c_max  = [0.58 0.42 0.74];   % muted purple
% c_base = [0.20 0.20 0.20];   % dark gray
% 
% %% ==================== Final figure ====================
% fig = figure('Color','w','Position',[80 60 1250 700]);
% tiledlayout(2,3,'TileSpacing','compact','Padding','compact');
% 
% % -------------------- Row 1: Q sensitivity --------------------
% nexttile;
% plot(Qtab.Q_scale, Qtab.RMSE_post, '-o', ...
%     'Color', c_rmse, 'MarkerFaceColor', c_rmse, 'MarkerSize', 6); hold on;
% xline(1, '--', 'Base', 'Color', c_base, 'LineWidth', 1.0, ...
%     'LabelVerticalAlignment','middle', 'LabelHorizontalAlignment','left');
% set(gca, 'XScale', 'log');
% xlabel('Q / Q_{base}');
% ylabel('Posterior RMSE (V)');
% title('Sensitivity of RMSE to Q');
% grid on; box on;
% xlim([min(Qtab.Q_scale)*0.9, max(Qtab.Q_scale)*1.1]);
% 
% nexttile;
% plot(Qtab.Q_scale, Qtab.MAE_post, '-s', ...
%     'Color', c_mae, 'MarkerFaceColor', c_mae, 'MarkerSize', 6); hold on;
% xline(1, '--', 'Base', 'Color', c_base, 'LineWidth', 1.0, ...
%     'LabelVerticalAlignment','middle', 'LabelHorizontalAlignment','left');
% set(gca, 'XScale', 'log');
% xlabel('Q / Q_{base}');
% ylabel('Posterior MAE (V)');
% title('Sensitivity of MAE to Q');
% grid on; box on;
% xlim([min(Qtab.Q_scale)*0.9, max(Qtab.Q_scale)*1.1]);
% 
% nexttile;
% plot(Qtab.Q_scale, Qtab.MaxAE_post, '-d', ...
%     'Color', c_max, 'MarkerFaceColor', c_max, 'MarkerSize', 6); hold on;
% xline(1, '--', 'Base', 'Color', c_base, 'LineWidth', 1.0, ...
%     'LabelVerticalAlignment','middle', 'LabelHorizontalAlignment','left');
% set(gca, 'XScale', 'log');
% xlabel('Q / Q_{base}');
% ylabel('Posterior MaxAE (V)');
% title('Sensitivity of MaxAE to Q');
% grid on; box on;
% xlim([min(Qtab.Q_scale)*0.9, max(Qtab.Q_scale)*1.1]);
% 
% % -------------------- Row 2: R sensitivity --------------------
% nexttile;
% plot(Rtab.R_scale, Rtab.RMSE_post, '-o', ...
%     'Color', c_rmse, 'MarkerFaceColor', c_rmse, 'MarkerSize', 6); hold on;
% xline(1, '--', 'Base', 'Color', c_base, 'LineWidth', 1.0, ...
%     'LabelVerticalAlignment','middle', 'LabelHorizontalAlignment','left');
% set(gca, 'XScale', 'log');
% xlabel('R / R_{base}');
% ylabel('Posterior RMSE (V)');
% title('Sensitivity of RMSE to R');
% grid on; box on;
% xlim([min(Rtab.R_scale)*0.9, max(Rtab.R_scale)*1.1]);
% 
% nexttile;
% plot(Rtab.R_scale, Rtab.MAE_post, '-s', ...
%     'Color', c_mae, 'MarkerFaceColor', c_mae, 'MarkerSize', 6); hold on;
% xline(1, '--', 'Base', 'Color', c_base, 'LineWidth', 1.0, ...
%     'LabelVerticalAlignment','middle', 'LabelHorizontalAlignment','left');
% set(gca, 'XScale', 'log');
% xlabel('R / R_{base}');
% ylabel('Posterior MAE (V)');
% title('Sensitivity of MAE to R');
% grid on; box on;
% xlim([min(Rtab.R_scale)*0.9, max(Rtab.R_scale)*1.1]);
% 
% nexttile;
% plot(Rtab.R_scale, Rtab.MaxAE_post, '-d', ...
%     'Color', c_max, 'MarkerFaceColor', c_max, 'MarkerSize', 6); hold on;
% xline(1, '--', 'Base', 'Color', c_base, 'LineWidth', 1.0, ...
%     'LabelVerticalAlignment','middle', 'LabelHorizontalAlignment','left');
% set(gca, 'XScale', 'log');
% xlabel('R / R_{base}');
% ylabel('Posterior MaxAE (V)');
% title('Sensitivity of MaxAE to R');
% grid on; box on;
% xlim([min(Rtab.R_scale)*0.9, max(Rtab.R_scale)*1.1]);
% 
% %% ==================== Export ====================
% drawnow;
% exportgraphics(fig, fullfile(base_dir, 'Fig_UKF_Sensitivity_QR_Final.png'), 'Resolution', 600);
% exportgraphics(fig, fullfile(base_dir, 'Fig_UKF_Sensitivity_QR_Final.pdf'), 'ContentType', 'vector');
% 
% disp('>>> Final UKF sensitivity figure has been saved.');


%% FC2 fault robustness study: TS-LSTM + UKF vs TS-LSTM + AUKF
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

%% ==================== 4. Fault case settings ====================
rng(42);  % fixed seed for reproducibility

fault_cases = {};

% Case 1: burst noise
case1.name = 'burst_noise';
case1.display_name = 'Burst noise fault';
case1.fault_idx = 1200:1260;
case1.type = 'noise';
case1.sigma = 0.08;
fault_cases{end+1} = case1;

% Case 2: bias fault
case2.name = 'bias_fault';
case2.display_name = 'Bias fault';
case2.fault_idx = 1420:1480;
case2.type = 'bias';
case2.bias = 0.08;
fault_cases{end+1} = case2;

out_dir = fullfile('消融实验', 'Fault_Robustness_UKF_AUKF');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

summary_rows = [];

%% ==================== 5. Run fault robustness cases ====================
for c = 1:length(fault_cases)
    fc = fault_cases{c};
    fprintf('\n==============================\n');
    fprintf('Running case: %s\n', fc.display_name);
    fprintf('==============================\n');

    V_meas_faulty = V_test;

    switch fc.type
        case 'noise'
            V_meas_faulty(fc.fault_idx) = V_meas_faulty(fc.fault_idx) + ...
                fc.sigma * randn(length(fc.fault_idx),1);
        case 'bias'
            V_meas_faulty(fc.fault_idx) = V_meas_faulty(fc.fault_idx) + fc.bias;
        otherwise
            error('Unknown fault type.');
    end

    % ---------- UKF ----------
    result_ukf = run_ukf_fault_case( ...
        N_windows, seq_len, y_pred_all, ...
        I_test, T_test, Pair_test, PH2_test, ...
        V_test, V_meas_faulty, ...
        A_area, l_thick);

    % ---------- AUKF ----------
    result_aukf = run_aukf_fault_case( ...
        N_windows, seq_len, y_pred_all, ...
        I_test, T_test, Pair_test, PH2_test, ...
        V_test, V_meas_faulty, ...
        A_area, l_thick);

    % ---------- Fault-window metrics ----------
    fi = fc.fault_idx;

    result_ukf.fault_rmse = sqrt(mean((V_test(fi) - result_ukf.V_post(fi)).^2, 'omitnan'));
    result_ukf.fault_mae  = mean(abs(V_test(fi) - result_ukf.V_post(fi)), 'omitnan');
    result_ukf.fault_maxae = max(abs(V_test(fi) - result_ukf.V_post(fi)));

    result_aukf.fault_rmse = sqrt(mean((V_test(fi) - result_aukf.V_post(fi)).^2, 'omitnan'));
    result_aukf.fault_mae  = mean(abs(V_test(fi) - result_aukf.V_post(fi)), 'omitnan');
    result_aukf.fault_maxae = max(abs(V_test(fi) - result_aukf.V_post(fi)));

    save(fullfile(out_dir, sprintf('RESULT_%s_UKF.mat', fc.name)), 'result_ukf');
    save(fullfile(out_dir, sprintf('RESULT_%s_AUKF.mat', fc.name)), 'result_aukf');

    summary_rows = [summary_rows; ...
        table(string(fc.name), "UKF", result_ukf.rmse_v_post, result_ukf.mae_v_post, result_ukf.maxae_v_post, ...
              result_ukf.fault_rmse, result_ukf.fault_mae, result_ukf.fault_maxae, ...
              'VariableNames', {'Case','Method','Global_RMSE','Global_MAE','Global_MaxAE','Fault_RMSE','Fault_MAE','Fault_MaxAE'})]; %#ok<AGROW>

    summary_rows = [summary_rows; ...
        table(string(fc.name), "AUKF", result_aukf.rmse_v_post, result_aukf.mae_v_post, result_aukf.maxae_v_post, ...
              result_aukf.fault_rmse, result_aukf.fault_mae, result_aukf.fault_maxae, ...
              'VariableNames', {'Case','Method','Global_RMSE','Global_MAE','Global_MaxAE','Fault_RMSE','Fault_MAE','Fault_MaxAE'})]; %#ok<AGROW>

    %% ---------- Plot comparison figure ----------
    set(groot, 'defaultAxesFontName', 'Times New Roman');
    set(groot, 'defaultTextFontName', 'Times New Roman');
    set(groot, 'defaultAxesFontSize', 11);
    set(groot, 'defaultLineLineWidth', 1.5);

    c_clean = [0.00 0.00 0.00];   % black
    c_fault = [0.75 0.25 0.25];   % muted red
    c_ukf   = [0.05 0.42 0.62];   % deep blue
    c_aukf  = [0.58 0.42 0.74];   % muted purple

    zr = max(1, fi(1)-40):min(N_windows, fi(end)+40);

    fig = figure('Color','w','Position',[90 70 1200 760]);
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    nexttile;
    plot(zr, V_test(zr), '-', 'Color', c_clean, 'LineWidth', 1.4); hold on;
    plot(zr, V_meas_faulty(zr), '--', 'Color', c_fault, 'LineWidth', 1.2);
    plot(zr, result_ukf.V_post(zr), '-', 'Color', c_ukf, 'LineWidth', 1.5);
    plot(zr, result_aukf.V_post(zr), '-', 'Color', c_aukf, 'LineWidth', 1.5);

    xline(fi(1), '--k', 'Fault start', 'LineWidth', 1.0, ...
        'LabelVerticalAlignment','bottom');
    xline(fi(end), '--k', 'Fault end', 'LineWidth', 1.0, ...
        'LabelVerticalAlignment','bottom');

    ylabel('Voltage (V)');
    title(sprintf('%s: UKF vs AUKF under faulty measurement', fc.display_name));
    legend({'Clean reference','Faulty measurement','UKF posterior','AUKF posterior'}, ...
        'Location','best', 'Box','off');
    grid on; box on;

    nexttile;
    plot(zr, result_ukf.R_history(zr), '-', 'Color', c_ukf, 'LineWidth', 1.4); hold on;
    plot(zr, result_aukf.R_history(zr), '-', 'Color', c_aukf, 'LineWidth', 1.4);

    xline(fi(1), '--k', 'Fault start', 'LineWidth', 1.0, ...
        'LabelVerticalAlignment','bottom');
    xline(fi(end), '--k', 'Fault end', 'LineWidth', 1.0, ...
        'LabelVerticalAlignment','bottom');

    ylabel('R');
    xlabel('Time window');
    title('Measurement-noise covariance used by UKF and AUKF');
    legend({'UKF R','AUKF R'}, 'Location','best', 'Box','off');
    grid on; box on;

    drawnow;
    exportgraphics(fig, fullfile(out_dir, sprintf('Fig_%s_UKF_vs_AUKF.png', fc.name)), 'Resolution', 600);
    exportgraphics(fig, fullfile(out_dir, sprintf('Fig_%s_UKF_vs_AUKF.pdf', fc.name)), 'ContentType', 'vector');
end

writetable(summary_rows, fullfile(out_dir, 'Fault_Robustness_Summary.csv'));
save(fullfile(out_dir, 'Fault_Robustness_All_Summary.mat'), 'summary_rows', 'fault_cases');

disp('>>> Fault robustness study has been completed and saved.');
%% ==================== Auxiliary functions ====================
function result_now = run_ukf_fault_case( ...
    N_windows, seq_len, y_pred_all, ...
    I_test, T_test, Pair_test, PH2_test, ...
    V_test_clean, V_meas_faulty, ...
    A_area, l_thick)

    x_est = [10.0; -0.85];
    P_cov = diag([0.08, 0.008]);
    Q_process = diag([5e-4, 5e-6]);
    R_base = 0.008;

    lambda_step_max = 0.03;
    xi1_step_max    = 0.002;

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
                z_sigma(:) = V_meas_faulty(k);
            end
        end

        z_pred = z_sigma * Wm';
        V_prior_history(k) = z_pred;

        innovation = V_meas_faulty(k) - z_pred;
        innovation_history(k) = innovation;

        R_k = R_base;
        R_history(k) = R_k;

        [x_est, P_cov] = ukfUpdate_local(x_sigma, z_sigma, x_prior, z_pred, V_meas_faulty(k), R_k, Wm, Wc, P_prior);

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

    result_now = pack_result(V_test_clean, V_meas_faulty, X_prior_history, X_est_history, ...
        V_prior_history, V_post_history, innovation_history, R_history);
end

function result_now = run_aukf_fault_case( ...
    N_windows, seq_len, y_pred_all, ...
    I_test, T_test, Pair_test, PH2_test, ...
    V_test_clean, V_meas_faulty, ...
    A_area, l_thick)

    x_est = [10.0; -0.85];
    P_cov = diag([0.08, 0.008]);
    Q_process = diag([5e-4, 5e-6]);

    R_base = 0.008;
    R_min  = R_base;
    R_max  = 20 * R_base;
    ripple_win  = 20;
    ripple_gain = 0.30;
    R_smooth_alpha = 0.85;

    lambda_step_max = 0.03;
    xi1_step_max    = 0.002;

    X_prior_history       = nan(N_windows, 2);
    X_est_history         = nan(N_windows, 2);
    V_prior_history       = nan(N_windows, 1);
    V_post_history        = nan(N_windows, 1);
    R_adaptive_history    = nan(N_windows, 1);
    innovation_history    = nan(N_windows, 1);
    innovation_ratio_hist = nan(N_windows, 1);

    R_prev = R_base;

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
                z_sigma(:) = V_meas_faulty(k);
            end
        end

        z_pred = z_sigma * Wm';
        V_prior_history(k) = z_pred;

        innovation = V_meas_faulty(k) - z_pred;
        innovation_history(k) = innovation;

        if k == 1
            local_std = max(abs(innovation), 1e-4);
        else
            win_idx = max(1, k-ripple_win+1):k;
            local_std = std(innovation_history(win_idx), 'omitnan');
            local_std = max(local_std, 1e-4);
        end

        innovation_ratio = abs(innovation) / local_std;
        innovation_ratio = min(innovation_ratio, 4.0);
        innovation_ratio_hist(k) = innovation_ratio;

        R_raw = R_base * (1 + ripple_gain * max(0, innovation_ratio - 1)^2);
        R_raw = min(max(R_raw, R_min), R_max);

        if k == 1
            R_k = R_raw;
        else
            R_k = R_smooth_alpha * R_prev + (1 - R_smooth_alpha) * R_raw;
        end
        R_k = min(max(R_k, R_min), R_max);
        R_prev = R_k;
        R_adaptive_history(k) = R_k;

        [x_est, P_cov] = ukfUpdate_local(x_sigma, z_sigma, x_prior, z_pred, V_meas_faulty(k), R_k, Wm, Wc, P_prior);

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

    result_now = pack_result(V_test_clean, V_meas_faulty, X_prior_history, X_est_history, ...
        V_prior_history, V_post_history, innovation_history, R_adaptive_history);
    result_now.innovation_ratio_hist = innovation_ratio_hist;
end

function result_now = pack_result(V_test_clean, V_meas_faulty, X_prior_history, X_est_history, ...
                                  V_prior_history, V_post_history, innovation_history, R_history)

    rmse_v_prior = sqrt(mean((V_test_clean - V_prior_history).^2, 'omitnan'));
    rmse_v_post  = sqrt(mean((V_test_clean - V_post_history).^2, 'omitnan'));
    mae_v_prior  = mean(abs(V_test_clean - V_prior_history), 'omitnan');
    mae_v_post   = mean(abs(V_test_clean - V_post_history), 'omitnan');
    maxae_v_prior = max(abs(V_test_clean - V_prior_history));
    maxae_v_post  = max(abs(V_test_clean - V_post_history));

    result_now = struct();
    result_now.V_test_clean = V_test_clean;
    result_now.V_meas_faulty = V_meas_faulty;
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