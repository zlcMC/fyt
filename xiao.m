%% Plot main ablation results for FC2 (3 methods)
clear; clc; close all;

%% ==================== 0. Load result files ====================
base_dir = '消融实验';

S1 = load(fullfile(base_dir, 'plain_LSTM_only', 'RESULT_plain_LSTM_only_FC2.mat'));
S2 = load(fullfile(base_dir, 'TS_LSTM_only',    'RESULT_TS_LSTM_only_FC2.mat'));
S3 = load(fullfile(base_dir, 'TS_LSTM_UKF',     'RESULT_TS_LSTM_UKF_FC2.mat'));

R1 = S1.RESULT;
R2 = S2.RESULT;
R3 = S3.RESULT;

out_dir = fullfile(base_dir, 'Ablation_Main_3Methods');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ==================== 1. Global plotting style ====================
set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultAxesFontSize', 11);
set(groot, 'defaultLineLineWidth', 1.5);

%% ==================== Unified color palette (step2-style) ====================
c_plain  = [0.72 0.72 0.72];   % light gray
c_tslstm = [0.12 0.47 0.71];   % step2-like blue
c_ukf    = [0.00 0.40 0.52];   % deeper blue-teal
c_meas   = [0.00 0.00 0.00];   % black

colors = [c_plain; c_tslstm; c_ukf];
methods = {'Plain LSTM', 'TS-LSTM', 'TS-LSTM + UKF'};

%% ==================== 2. Collect metrics ====================
rmse_vals  = [R1.rmse_v_prior,  R2.rmse_v_prior,  R3.rmse_v_post];
mae_vals   = [R1.mae_v_prior,   R2.mae_v_prior,   R3.mae_v_post];
maxae_vals = [R1.maxae_v_prior, R2.maxae_v_prior, R3.maxae_v_post];

T = table(methods', rmse_vals', mae_vals', maxae_vals', ...
    'VariableNames', {'Method','RMSE','MAE','MaxAE'});
writetable(T, fullfile(out_dir, 'Ablation_Main_3Methods_Table.csv'));

%% ==================== 3. Figure 1: Bar chart of metrics ====================
fig1 = figure('Color','w','Position',[100 80 1180 420]);
tiledlayout(1,3,'TileSpacing','compact','Padding','compact');

nexttile;
b1 = bar(rmse_vals, 'FaceColor', 'flat', 'BarWidth', 0.72);
for i = 1:length(rmse_vals)
    b1.CData(i,:) = colors(i,:);
end
set(gca, 'XTick', 1:3, 'XTickLabel', methods, 'XTickLabelRotation', 18);
ylabel('RMSE (V)');
title('Voltage reconstruction RMSE');
grid on; box on;
ylim([0, max(rmse_vals)*1.15]);
for i = 1:length(rmse_vals)
    text(i, rmse_vals(i)+0.00015, sprintf('%.4f', rmse_vals(i)), ...
        'HorizontalAlignment','center', 'FontSize',10);
end

nexttile;
b2 = bar(mae_vals, 'FaceColor', 'flat', 'BarWidth', 0.72);
for i = 1:length(mae_vals)
    b2.CData(i,:) = colors(i,:);
end
set(gca, 'XTick', 1:3, 'XTickLabel', methods, 'XTickLabelRotation', 18);
ylabel('MAE (V)');
title('Voltage reconstruction MAE');
grid on; box on;
ylim([0, max(mae_vals)*1.15]);
for i = 1:length(mae_vals)
    text(i, mae_vals(i)+0.00015, sprintf('%.4f', mae_vals(i)), ...
        'HorizontalAlignment','center', 'FontSize',10);
end

nexttile;
b3 = bar(maxae_vals, 'FaceColor', 'flat', 'BarWidth', 0.72);
for i = 1:length(maxae_vals)
    b3.CData(i,:) = colors(i,:);
end
set(gca, 'XTick', 1:3, 'XTickLabel', methods, 'XTickLabelRotation', 18);
ylabel('MaxAE (V)');
title('Maximum absolute error');
grid on; box on;
ylim([0, max(maxae_vals)*1.15]);
for i = 1:length(maxae_vals)
    text(i, maxae_vals(i)+0.001, sprintf('%.4f', maxae_vals(i)), ...
        'HorizontalAlignment','center', 'FontSize',10);
end

drawnow;
exportgraphics(fig1, fullfile(out_dir, 'Fig_Ablation3_Bar_Metrics.png'), 'Resolution', 600);
print(fig1, fullfile(out_dir, 'Fig_Ablation3_Bar_Metrics'), '-dpdf', '-painters');

%% ==================== 4. Figure 2: Overall comparison ====================
V_test = R3.V_test;
t = (1:length(V_test))';

fig2 = figure('Color','w','Position',[100 80 1250 520]);
plot(t, V_test, 'Color', c_meas, 'LineWidth', 1.3); hold on;
plot(t, R1.V_prior, '-', 'Color', c_plain,  'LineWidth', 1.1);
plot(t, R2.V_prior, '-', 'Color', c_tslstm, 'LineWidth', 1.2);
plot(t, R3.V_post,  '-', 'Color', c_ukf,    'LineWidth', 1.6);

xlabel('Time window');
ylabel('Voltage (V)');
title('Overall comparison of voltage reconstruction on the FC2 ripple dataset');
legend({'Measured voltage','Plain LSTM','TS-LSTM','TS-LSTM + UKF'}, ...
       'Location','northwest', 'Box','off');
grid on; box on;
xlim([1 length(V_test)]);
ylim([0.61 0.98]);

drawnow;
exportgraphics(fig2, fullfile(out_dir, 'Fig_Ablation3_Overall_Comparison.png'), 'Resolution', 600);
print(fig2, fullfile(out_dir, 'Fig_Ablation3_Overall_Comparison'), '-dpdf', '-painters');

%% ==================== 5. Figure 3: Zoomed comparison ====================
zoom_sets = {
    max(1,1100-50):min(length(V_test),1100+50), ...
    max(1,1420-50):min(length(V_test),1420+50), ...
    max(1,1750-50):min(length(V_test),1750+50)
};

fig3 = figure('Color','w','Position',[80 60 1250 920]);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');

for i = 1:3
    zr = zoom_sets{i};
    nexttile;
    plot(zr, V_test(zr), 'Color', c_meas, 'LineWidth', 1.3); hold on;
    plot(zr, R1.V_prior(zr), '-', 'Color', c_plain,  'LineWidth', 1.0);
    plot(zr, R2.V_prior(zr), '-', 'Color', c_tslstm, 'LineWidth', 1.1);
    plot(zr, R3.V_post(zr),  '-', 'Color', c_ukf,    'LineWidth', 1.6);

    ylabel('Voltage (V)');
    title(sprintf('Zoomed comparison in transient region %d', i));
    legend({'Measured','Plain LSTM','TS-LSTM','TS-LSTM + UKF'}, ...
           'Location','best', 'Box','off');
    grid on; box on;
end
xlabel('Time window');

drawnow;
exportgraphics(fig3, fullfile(out_dir, 'Fig_Ablation3_Zoomed_Comparison.png'), 'Resolution', 600);
print(fig3, fullfile(out_dir, 'Fig_Ablation3_Zoomed_Comparison'), '-dpdf', '-painters');

%% ==================== 6. Figure 4: Absolute error boxplot ====================
err1 = abs(R1.V_prior - V_test);
err2 = abs(R2.V_prior - V_test);
err3 = abs(R3.V_post  - V_test);

fig4 = figure('Color','w','Position',[120 100 900 460]);
boxplot([err1(:), err2(:), err3(:)], ...
    'Labels', {'Plain LSTM', 'TS-LSTM', 'TS-LSTM + UKF'}, ...
    'Symbol', 'r+', 'Whisker', 1.5);
ylabel('Absolute voltage error (V)');
title('Absolute error comparison across the main ablation methods');
grid on; box on;

drawnow;
exportgraphics(fig4, fullfile(out_dir, 'Fig_Ablation3_Boxplot_Error.png'), 'Resolution', 600);
print(fig4, fullfile(out_dir, 'Fig_Ablation3_Boxplot_Error'), '-dpdf', '-painters');

disp('>>> Main ablation figures and table have been saved.');