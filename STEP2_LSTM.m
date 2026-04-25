%% FC1 Step 2 Baseline: Plain LSTM internal-state learner
% Baseline model for ablation:
% - No self-attention
% - No shared-bottom dual-head structure
% - Single LSTM backbone with 2-output regression
clear; clc; close all;

%% ==================== 0. Basic configuration ====================
disp('0. Configuration...');
pseudo_file = 'FC1_Full_PseudoLabels.mat';

seq_len = 80;
stride = 1;
val_ratio = 0.15;
test_ratio = 0.15;

max_epochs = 60;
mini_batch_size = 128;
initial_lr = 5e-4;
drop_period = 20;
drop_factor = 0.5;

rng(42);

%% ==================== 1. Load and clean data ====================
disp('1. Loading pseudo-label data...');
S = load(pseudo_file);

I_down       = S.I_down(:);
T_K_down     = S.T_K_down(:);
Pair_down    = S.Pair_down(:);
PH2_down     = S.PH2_down(:);
lambda_label = S.lambda_label(:);
xi1_label    = S.xi1_label(:);
norm_stats   = S.norm_stats;

N = min([length(I_down), length(T_K_down), length(Pair_down), ...
         length(PH2_down), length(lambda_label), length(xi1_label)]);

valid_mask = ...
    ~isnan(I_down(1:N)) & isfinite(I_down(1:N)) & ...
    ~isnan(T_K_down(1:N)) & isfinite(T_K_down(1:N)) & ...
    ~isnan(Pair_down(1:N)) & isfinite(Pair_down(1:N)) & ...
    ~isnan(PH2_down(1:N)) & isfinite(PH2_down(1:N)) & ...
    ~isnan(lambda_label(1:N)) & isfinite(lambda_label(1:N)) & ...
    ~isnan(xi1_label(1:N)) & isfinite(xi1_label(1:N));

I_down       = I_down(valid_mask);
T_K_down     = T_K_down(valid_mask);
Pair_down    = Pair_down(valid_mask);
PH2_down     = PH2_down(valid_mask);
lambda_label = lambda_label(valid_mask);
xi1_label    = xi1_label(valid_mask);

N = length(I_down);
fprintf('Valid sample length N = %d\n', N);

%% ==================== 2. Assemble and normalize ====================
disp('2. Assembling and normalizing data...');

X_all = [I_down, T_K_down, Pair_down, PH2_down];

x_mean = [norm_stats.I(1), norm_stats.T(1), norm_stats.Pair(1), norm_stats.PH2(1)];
x_std  = [norm_stats.I(2), norm_stats.T(2), norm_stats.Pair(2), norm_stats.PH2(2)];

Y_all = [lambda_label, xi1_label];
y_mean = [norm_stats.lam(1), norm_stats.xi1(1)];
y_std  = [norm_stats.lam(2), norm_stats.xi1(2)];

x_std(x_std == 0) = 1;
y_std(y_std == 0) = 1;

X_all_norm = (X_all - x_mean) ./ x_std;
Y_all_norm = (Y_all - y_mean) ./ y_std;

num_features = size(X_all, 2);
num_outputs  = size(Y_all, 2);

%% ==================== 3. Split and build sequences ====================
disp('3. Splitting dataset and building sequences...');

n_train = floor((1 - val_ratio - test_ratio) * N);
n_val   = floor(val_ratio * N);

X_train_norm = X_all_norm(1:n_train, :);
Y_train_norm = Y_all_norm(1:n_train, :);

X_val_norm   = X_all_norm(n_train+1:n_train+n_val, :);
Y_val_norm   = Y_all_norm(n_train+1:n_train+n_val, :);

X_test_norm  = X_all_norm(n_train+n_val+1:end, :);
Y_test_norm  = Y_all_norm(n_train+n_val+1:end, :);

[XTrain, YTrain] = buildSequenceDataset(X_train_norm, Y_train_norm, seq_len, stride);
[XVal,   YVal]   = buildSequenceDataset(X_val_norm,   Y_val_norm,   seq_len, 1);
[XTest,  YTest]  = buildSequenceDataset(X_test_norm,  Y_test_norm,  seq_len, 1);

%% ==================== 4. Build plain LSTM baseline ====================
disp('4. Building plain LSTM baseline...');

layers = [
    sequenceInputLayer(num_features, 'Name', 'input')
    lstmLayer(128, 'OutputMode', 'last', 'Name', 'lstm')
    dropoutLayer(0.2, 'Name', 'dropout')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(num_outputs, 'Name', 'fc_out')
    regressionLayer('Name', 'reg_output')
];

%% ==================== 5. Training ====================
disp('5. Training plain LSTM baseline...');

if canUseGPU
    exec_env = 'gpu';
else
    exec_env = 'auto';
end

options = trainingOptions('adam', ...
    'ExecutionEnvironment', exec_env, ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_lr, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', drop_factor, ...
    'LearnRateDropPeriod', drop_period, ...
    'L2Regularization', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

[plain_lstm_net, info] = trainNetwork(XTrain, YTrain, layers, options);

%% ==================== 6. Test evaluation ====================
disp('6. Test evaluation...');
YPred_norm = predict(plain_lstm_net, XTest);

YPred = YPred_norm .* y_std + y_mean;
YTrue = YTest .* y_std + y_mean;

rmse_lambda = sqrt(mean((YTrue(:,1) - YPred(:,1)).^2));
rmse_xi1    = sqrt(mean((YTrue(:,2) - YPred(:,2)).^2));

mae_lambda = mean(abs(YTrue(:,1) - YPred(:,1)));
mae_xi1    = mean(abs(YTrue(:,2) - YPred(:,2)));

ss_res_lambda = sum((YTrue(:,1) - YPred(:,1)).^2);
ss_tot_lambda = sum((YTrue(:,1) - mean(YTrue(:,1))).^2);
if ss_tot_lambda < 1e-12
    r2_lambda = NaN;
else
    r2_lambda = 1 - ss_res_lambda / ss_tot_lambda;
end

ss_res_xi1 = sum((YTrue(:,2) - YPred(:,2)).^2);
ss_tot_xi1 = sum((YTrue(:,2) - mean(YTrue(:,2))).^2);
if ss_tot_xi1 < 1e-12
    r2_xi1 = NaN;
else
    r2_xi1 = 1 - ss_res_xi1 / ss_tot_xi1;
end

fprintf('\n===== Plain LSTM Baseline Results =====\n');
fprintf('lambda: RMSE = %.4f, MAE = %.4f, R2 = %.4f\n', rmse_lambda, mae_lambda, r2_lambda);
fprintf('xi_1  : RMSE = %.4f, MAE = %.4f, R2 = %.4f\n', rmse_xi1, mae_xi1, r2_xi1);

%% ==================== 7. Save model and paper-ready results ====================
save('FC1_Plain_LSTM_Model.mat', 'plain_lstm_net', 'x_mean', 'x_std', 'y_mean', 'y_std');
disp('>>> Plain LSTM model saved to FC1_Plain_LSTM_Model.mat');

out_dir = 'results_step2_lstm';
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

LSTM_BASELINE_RESULTS = struct();
LSTM_BASELINE_RESULTS.YTrue = YTrue;
LSTM_BASELINE_RESULTS.YPred = YPred;
LSTM_BASELINE_RESULTS.rmse_lambda = rmse_lambda;
LSTM_BASELINE_RESULTS.rmse_xi1    = rmse_xi1;
LSTM_BASELINE_RESULTS.mae_lambda  = mae_lambda;
LSTM_BASELINE_RESULTS.mae_xi1     = mae_xi1;
LSTM_BASELINE_RESULTS.r2_lambda   = r2_lambda;
LSTM_BASELINE_RESULTS.r2_xi1      = r2_xi1;
LSTM_BASELINE_RESULTS.info        = info;
LSTM_BASELINE_RESULTS.x_mean      = x_mean;
LSTM_BASELINE_RESULTS.x_std       = x_std;
LSTM_BASELINE_RESULTS.y_mean      = y_mean;
LSTM_BASELINE_RESULTS.y_std       = y_std;
LSTM_BASELINE_RESULTS.seq_len     = seq_len;

save(fullfile(out_dir, 'STEP2_LSTM_BASELINE_RESULTS.mat'), 'LSTM_BASELINE_RESULTS');

T = table({'lambda'; 'xi1'}, ...
          [rmse_lambda; rmse_xi1], ...
          [mae_lambda; mae_xi1], ...
          [r2_lambda; r2_xi1], ...
          'VariableNames', {'State','RMSE','MAE','R2'});
writetable(T, fullfile(out_dir, 'Table_STEP2_LSTM_BASELINE.csv'));

%% ==================== Appendix ====================
function [XCell, YMat] = buildSequenceDataset(X, Y, seq_len, stride)
    N = size(X, 1); XCell = {}; YMat = [];
    if N < seq_len
        return;
    end
    for i = 1:stride:(N - seq_len + 1)
        XCell{end+1,1} = X(i:i+seq_len-1, :)';
        YMat(end+1,:) = Y(i+seq_len-1, :);
    end
end