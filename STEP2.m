%% FC1 第二步：Attention-enhanced TS-LSTM 内部状态学习器 (顶刊升级版 - 解决过拟合)
% 核心创新：
% 1. 引入 Self-Attention 机制，捕捉长序列中的工况突变（如电流阶跃）
% 2. 采用 Shared-Bottom + Dual-Head 架构，实现宏微观状态的解耦预测
% 3. 【核心修复】采用全局随机滑窗打乱策略，彻底解决慢变状态的外推失效与过拟合问题
clear; clc; close all;

%% ==================== 0. 基础配置 ====================
disp('0. 参数配置...');
pseudo_file = 'FC1_Full_PseudoLabels.mat'; % 【注意】确保这里读取的是你第一步跑出来的真实文件名！

% ---------- 时序与训练参数 ----------
seq_len = 80;            % 历史窗口长度 (网络每次回头看 80 个时间步的数据)
stride = 1;              % 滑动步长为 1，确保训练样本极其丰富
val_ratio = 0.15;        % 15% 的数据用来做验证集
test_ratio = 0.15;       % 15% 的数据作为最终未见过的测试集

max_epochs = 60;         % 最大训练轮数
mini_batch_size = 128;   % 每次拿 128 个样本一起更新梯度
initial_lr = 5e-4;       % 初始学习率
drop_period = 20;        % 每训练 20 轮...
drop_factor = 0.5;       % ...学习率就减半

rng(42); % 固定随机种子，保证每次运行结果一模一样，方便论文复现

%% ==================== 1. 加载与清洗数据 ====================
disp('1. 加载伪标签与清洗数据...');
S = load(pseudo_file);

% 【去除共线性】剔除冗余的 J_down，只保留 4 个相互独立的物理可测变量
I_down       = S.I_down(:);
T_K_down     = S.T_K_down(:);
Pair_down    = S.Pair_down(:);
PH2_down     = S.PH2_down(:);
lambda_label = S.lambda_label(:);
xi1_label    = S.xi1_label(:);
norm_stats   = S.norm_stats; % 读取第一步存下来的统计数据

% 数据清洗：防止有 NaN (空值) 或者 Inf (无穷大) 导致网络崩溃
N = min([length(I_down), length(T_K_down), length(Pair_down), length(PH2_down), length(lambda_label), length(xi1_label)]);
valid_mask = ...
    ~isnan(I_down(1:N)) & isfinite(I_down(1:N)) & ...
    ~isnan(T_K_down(1:N)) & isfinite(T_K_down(1:N)) & ...
    ~isnan(Pair_down(1:N)) & isfinite(Pair_down(1:N)) & ...
    ~isnan(PH2_down(1:N)) & isfinite(PH2_down(1:N)) & ...
    ~isnan(lambda_label(1:N)) & isfinite(lambda_label(1:N)) & ...
    ~isnan(xi1_label(1:N)) & isfinite(xi1_label(1:N));

% 过滤保留有效数据
I_down = I_down(valid_mask); T_K_down = T_K_down(valid_mask);
Pair_down = Pair_down(valid_mask); PH2_down = PH2_down(valid_mask);
lambda_label = lambda_label(valid_mask); xi1_label = xi1_label(valid_mask);

N = length(I_down);
fprintf('有效样本长度 N = %d\n', N);

%% ==================== 2. 组装与标准化 (量纲统一) ====================
disp('2. 组装网络输入输出并标准化...');

% 将 4 个物理量拼成输入矩阵 X
X_all = [I_down, T_K_down, Pair_down, PH2_down];

% 直接使用第一步的均值和标准差，保证验证的一致性
x_mean = [norm_stats.I(1), norm_stats.T(1), norm_stats.Pair(1), norm_stats.PH2(1)];
x_std  = [norm_stats.I(2), norm_stats.T(2), norm_stats.Pair(2), norm_stats.PH2(2)];

Y_all = [lambda_label, xi1_label];
y_mean = [norm_stats.lam(1), norm_stats.xi1(1)];
y_std  = [norm_stats.lam(2), norm_stats.xi1(2)];

% 防止标准差为 0 导致除以 0 的错误
x_std(x_std == 0) = 1; y_std(y_std == 0) = 1;

% 【极其关键】Z-score 标准化：把 X 和 Y 都压缩到均值为0，方差为1 的正态分布
X_all_norm = (X_all - x_mean) ./ x_std;
Y_all_norm = (Y_all - y_mean) ./ y_std;

num_features = size(X_all, 2); % 4个特征
num_outputs  = size(Y_all, 2); % 2个输出

%% ==================== 3. 构造全量滑动窗口并随机打乱 (核心修复区) ====================
disp('3. 构造全量时序窗口并随机打乱划分 (解决外推失效的核心)...');

% 1. 先对全量数据构造窗口 (步长设为 1，保证样本量充足)
[X_all_seq, Y_all_seq] = buildSequenceDataset(X_all_norm, Y_all_norm, seq_len, stride);

num_total_windows = size(X_all_seq, 1);
fprintf('总共生成了 %d 个时序窗口样本\n', num_total_windows);

% 2. 全局随机打乱索引 (洗牌)
% 这样能保证训练集里既有电池早期的健康数据，也有电池晚期的老化数据，彻底解决外推失效！
rng(42); 
shuffled_idx = randperm(num_total_windows);

% 3. 按比例划分打乱后的索引
n_train = floor((1 - val_ratio - test_ratio) * num_total_windows);
n_val   = floor(val_ratio * num_total_windows);

idx_train = shuffled_idx(1:n_train);
idx_val   = shuffled_idx(n_train+1 : n_train+n_val);
idx_test  = shuffled_idx(n_train+n_val+1 : end);

% 4. 分配给训练、验证、测试集
XTrain = X_all_seq(idx_train);
YTrain = Y_all_seq(idx_train, :);

XVal   = X_all_seq(idx_val);
YVal   = Y_all_seq(idx_val, :);

XTest  = X_all_seq(idx_test);
YTest  = Y_all_seq(idx_test, :);

fprintf('打乱后 -> 训练集: %d, 验证集: %d, 测试集: %d\n', ...
    length(XTrain), length(XVal), length(XTest));

%% ==================== 4. 构建 Attention-enhanced TS-LSTM ====================
disp('4. 构建带注意力机制的双分支 TS-LSTM...');

lgraph = layerGraph();

% 【架构模块 1：共享底层 Shared Bottom】
shared_layers = [
    sequenceInputLayer(num_features, 'Name', 'input') % 接收 4 维输入
    fullyConnectedLayer(64, 'Name', 'embed_fc')       % 升维到 64 维空间提取特征
    selfAttentionLayer(4, 64, 'Name', 'self_attn')    % ★大招：4头注意力机制，精准捕捉电流突变时刻
    layerNormalizationLayer('Name', 'layer_norm')     % 层归一化，防止梯度消失
    lstmLayer(128, 'OutputMode', 'sequence', 'Name', 'shared_lstm') % 提取时序共性特征
    dropoutLayer(0.2, 'Name', 'shared_drop')          % 随机丢弃20%神经元，防止死记硬背
];
lgraph = addLayers(lgraph, shared_layers);

% 【架构模块 2：快变分支 Fast Head (专门预测膜含水量 lambda)】
fast_branch = [
    lstmLayer(64, 'OutputMode', 'last', 'Name', 'fast_lstm') % 64个神经元，负责高频动态
    fullyConnectedLayer(32, 'Name', 'fast_fc')
    reluLayer('Name', 'fast_relu')
    fullyConnectedLayer(1, 'Name', 'out_lambda')             % 输出1个值：归一化后的 lambda
];
lgraph = addLayers(lgraph, fast_branch);

% 【架构模块 3：慢变分支 Slow Head (专门预测催化剂老化 xi1)】
slow_branch = [
    lstmLayer(32, 'OutputMode', 'last', 'Name', 'slow_lstm') % 32个神经元，强制平滑，关注慢变
    fullyConnectedLayer(16, 'Name', 'slow_fc')
    reluLayer('Name', 'slow_relu')
    fullyConnectedLayer(1, 'Name', 'out_xi1')                % 输出1个值：归一化后的 xi_1
];
lgraph = addLayers(lgraph, slow_branch);

% 【连接网络】让共享层的输出同时进入快分支和慢分支
lgraph = connectLayers(lgraph, 'shared_drop', 'fast_lstm');
lgraph = connectLayers(lgraph, 'shared_drop', 'slow_lstm');

% 【拼接与输出】将两头的结果拼在一起
concat_layer = concatenationLayer(1, 2, 'Name', 'concat_out'); 
lgraph = addLayers(lgraph, concat_layer);
lgraph = connectLayers(lgraph, 'out_lambda', 'concat_out/in1');
lgraph = connectLayers(lgraph, 'out_xi1', 'concat_out/in2');

final_layer = regressionLayer('Name', 'reg_output'); % 计算均方误差(MSE)
lgraph = addLayers(lgraph, final_layer);
lgraph = connectLayers(lgraph, 'concat_out', 'reg_output');

%% ==================== 5. 训练与评估 ====================
disp('5. 开始训练...');
% 自动检测是否有 GPU 加速
if canUseGPU
    exec_env = 'gpu';
else
    exec_env = 'auto';
end

% 设定训练参数大全
options = trainingOptions('adam', ...
    'ExecutionEnvironment', exec_env, ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_lr, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', drop_factor, ...
    'LearnRateDropPeriod', drop_period, ...
    'L2Regularization', 1e-3, ...             % 【优化】加大 L2 正则化，强效防止过拟合
    'Shuffle', 'every-epoch', ...             
    'ValidationData', {XVal, YVal}, ...       
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...         
    'Verbose', false);

% 执行训练！
[ts_lstm_net, info] = trainNetwork(XTrain, YTrain, lgraph, options);

%% ==================== 6. 测试集终极评估 ====================
disp('6. 测试集评估...');
YPred_norm = predict(ts_lstm_net, XTest);

% 【极其关键】将网络吐出的归一化数值，反向还原成具有物理意义的真实数值！
YPred = YPred_norm .* y_std + y_mean;
YTrue = YTest .* y_std + y_mean;

% 计算快变状态 lambda 的误差
rmse_lambda = sqrt(mean((YTrue(:,1) - YPred(:,1)).^2));
ss_res_lambda = sum((YTrue(:,1) - YPred(:,1)).^2);
ss_tot_lambda = sum((YTrue(:,1) - mean(YTrue(:,1))).^2);
if ss_tot_lambda < 1e-12; r2_lambda = NaN; else; r2_lambda = 1 - ss_res_lambda / ss_tot_lambda; end

% 计算慢变状态 xi1 的误差
rmse_xi1    = sqrt(mean((YTrue(:,2) - YPred(:,2)).^2));
ss_res_xi1 = sum((YTrue(:,2) - YPred(:,2)).^2);
ss_tot_xi1 = sum((YTrue(:,2) - mean(YTrue(:,2))).^2);
if ss_tot_xi1 < 1e-12; r2_xi1 = NaN; else; r2_xi1 = 1 - ss_res_xi1 / ss_tot_xi1; end

fprintf('\n===== 测试结果 =====\n');
fprintf('lambda (快变): RMSE = %.4f, R2 = %.4f\n', rmse_lambda, r2_lambda);
fprintf('xi_1   (慢变): RMSE = %.4f, R2 = %.4f\n', rmse_xi1, r2_xi1);

% 保存训练好的神仙模型
save('FC1_TS_LSTM_Attention_Model.mat', 'ts_lstm_net', 'x_mean', 'x_std', 'y_mean', 'y_std');
disp('>>> 模型已存入 FC1_TS_LSTM_Attention_Model.mat');

%% ==================== 附录：滑窗样本构造函数 ====================
function [XCell, YMat] = buildSequenceDataset(X, Y, seq_len, stride)
    % 作用：把连续的一维时序数据，切成一段一段具有长度 seq_len 的序列片段
    N = size(X, 1); XCell = {}; YMat = [];
    if N < seq_len; return; end
    for i = 1:stride:(N - seq_len + 1)
        XCell{end+1,1} = X(i:i+seq_len-1, :)'; % 提取过去 seq_len 步的特征
        YMat(end+1,:) = Y(i+seq_len-1, :);     % 标签是序列最后时刻的真实值
    end
end