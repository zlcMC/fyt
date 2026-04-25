%% PEMFC 全生命周期伪标签提取器 v3.1
% 功能：
% 从 FC1 全寿命实验数据中，基于 PEMFC 电压机理模型反演出两个伪标签：
%   1) lambda：膜含水相关快变状态，反映中高频水热动态
%   2) xi_1：活化极化相关慢变退化状态，反映长期老化趋势
%
% 版本特点：
%   (1) 不再对 xi_1 强制单调下降，允许出现“可逆恢复”
%   (2) 使用较短窗口的 S-G 滤波，保留 lambda 的较快动态
%   (3) n_cells 参数化，便于适配不同电堆
%   (4) 保存归一化统计量，供后续 TS-LSTM 等模型直接复用
%   (5) 增加局部诊断图，验证 lambda 能否跟踪电流跃迁附近的动态变化

clear; clc; close all;

%% ==================== 1. 批量读取 ====================
disp('1. 加载并拼接 FC1 全生命周期数据...');

% 三段 Excel 文件路径（FC1 老化试验分段数据）
file_names = {'F:\FC1_FC2_Excel\FC1_FC2_Excel\FC1_Without_Ripples_Excel\FC1_Ageing_part1.xlsx', ...
              'F:\FC1_FC2_Excel\FC1_FC2_Excel\FC1_Without_Ripples_Excel\FC1_Ageing_part2.xlsx', ...
              'F:\FC1_FC2_Excel\FC1_FC2_Excel\FC1_Without_Ripples_Excel\FC1_Ageing_part3.xlsx'};

% 初始化总表
data_full = table();

% 检测导入选项，保留 Excel 原始列名
opts = detectImportOptions(file_names{1}, 'VariableNamingRule', 'preserve');

% 逐个文件读取并纵向拼接
for i = 1:length(file_names)
    if isfile(file_names{i})
        fprintf('   %s\n', file_names{i});
        data_full = [data_full; readtable(file_names{i}, opts)];
    end
end

%% ==================== 2. 变量提取 + 单位换算 ====================
% 从表中提取所需变量：
%   I        : 电流
%   J        : 电流密度
%   T_C      : 冷却水出口温度（摄氏度）
%   P_air    : 空气出口压力
%   P_H2     : 氢气出口压力
%   Utot     : 电堆总电压

try
    % 优先按列名提取（更稳妥）
    I_raw=data_full.('I (A)');
    J_raw=data_full.('J (A/cmｲ)');
    T_C_raw=data_full.('ToutWAT (ｰC)');
    P_air_raw=data_full.('PoutAIR (mbara)');
    P_H2_raw=data_full.('PoutH2 (mbara)');
    Utot_raw=data_full.('Utot (V)');
catch
    % 如果列名异常，则退化为按列索引提取
    I_raw=data_full{:,9};
    J_raw=data_full{:,8};
    T_C_raw=data_full{:,15};
    P_air_raw=data_full{:,17};
    P_H2_raw=data_full{:,18};
    Utot_raw=data_full{:,7};
end

% 电堆单体数（FC1 为 5 节）
n_cells = 5;

% 根据 I/J 估算有效反应面积 A = I / J
A_area = I_raw(1)/J_raw(1);

% 膜厚度，单位 cm（Nafion 117）
l_thickness = 0.00508;

% 将总电压转换为单体平均电压
V_actual_raw = Utot_raw / n_cells;

% 温度转换：摄氏度 -> 开尔文
T_K_raw = T_C_raw + 273.15;

% 压力转换：mbar(a) -> atm
P_air_atm = P_air_raw / 1013.25;
P_H2_atm  = P_H2_raw  / 1013.25;

fprintf('n_cells=%d, A=%.2f cm², l=%.4f cm\n', n_cells, A_area, l_thickness);

%% ==================== 3. 降采样 ====================
% 原始数据采样点很多，直接逐点反演计算量大且噪声较强
% 因此按固定窗口取均值，做时域降采样
window_size = 60;

N_raw = length(I_raw);                 % 原始长度
N_windows = floor(N_raw/window_size);  % 可形成多少个完整窗口

% 初始化降采样后的变量
I_down=zeros(N_windows,1);
J_down=zeros(N_windows,1);
T_K_down=zeros(N_windows,1);
V_actual_down=zeros(N_windows,1);
Pair_down=zeros(N_windows,1);
PH2_down=zeros(N_windows,1);

% 每 60 个点取一个均值
for i=1:N_windows
    idx=(i-1)*window_size+1:i*window_size;
    I_down(i)=mean(I_raw(idx));
    J_down(i)=mean(J_raw(idx));
    T_K_down(i)=mean(T_K_raw(idx));
    V_actual_down(i)=mean(V_actual_raw(idx));
    Pair_down(i)=mean(P_air_atm(idx));
    PH2_down(i)=mean(P_H2_atm(idx));
end

%% ==================== 4. 双向解耦 + 迭代 ====================
% 核心思想：
% 在 PEMFC 电压模型中，lambda 和 xi_1 都影响输出电压，
% 但二者时标不同：
%   - xi_1：慢变量，反映长期退化
%   - lambda：快变量，反映膜水合/热管理动态
%
% 因此采用“交替反演”的方式：
%   Stage 1：固定 lambda，估计 xi_1
%   Stage 2：固定 xi_1，估计 lambda
% 并迭代若干轮使结果收敛

% 非线性最小二乘求解器设置
options = optimoptions('lsqnonlin','Display','off','Algorithm','trust-region-reflective');

% 根据模型中的分母约束，给 lambda 设置全局下界
J_max_global = max(J_down);
lambda_lb_global = max(3.0, 0.634 + 3*J_max_global + 0.5);

% 交替反演轮数
n_iter = 2;

% 初始化 lambda 轨迹
lambda_traj = 10*ones(N_windows,1);

% 初始化结果变量
xi1_label = zeros(N_windows,1);
lambda_label = zeros(N_windows,1);
V_fit = zeros(N_windows,1);

% S-G 滤波窗口长度
sgolay_win = 201;

for it = 1:n_iter
    fprintf('\n=== 迭代 %d / %d ===\n', it, n_iter);
    
    %% ---- Stage 1：反演 ξ₁ ----
    fprintf('  [Stage 1] 反演 ξ₁ ...\n');
    
    xi1_raw = zeros(N_windows,1);
    x0_xi1  = -0.912;   % 初值，来源于经验参数范围
    
    for k = 1:N_windows
        % 取当前窗口的平均工况
        I_k=max(I_down(k),1e-3);
        J_k=max(J_down(k),1e-4);
        T_k=T_K_down(k);
        V_k=V_actual_down(k);
        PH2_k=max(PH2_down(k),1e-3);
        Pair_k=max(Pair_down(k),1e-3);
        
        % 固定当前 lambda
        lam_k = lambda_traj(k);
        
        % 构造目标函数：模型电压 - 实际电压
        fun = @(xi1) voltage_error_single(lam_k, xi1, I_k, J_k, T_k, PH2_k, Pair_k, V_k, A_area, l_thickness);
        
        % 非线性最小二乘反演 xi1
        xi1_raw(k) = lsqnonlin(fun, x0_xi1, -1.5, -0.5, options);
        
        % 用前一时刻结果作为下一时刻初值，提高连续性和收敛速度
        x0_xi1 = xi1_raw(k);
    end
    
    % 对逐点反演出的 xi1 做 S-G 平滑
    % 注意：这里不再使用 cummin 强制单调恶化
    % 因此允许出现短时恢复现象
    xi1_smooth = smoothdata(xi1_raw, 'sgolay', sgolay_win);
    xi1_label  = xi1_smooth;
    
    %% ---- Stage 2：反演 λ ----
    fprintf('  [Stage 2] 反演 λ ...\n');
    
    x0_lambda = 10.0;
    for k = 1:N_windows
        % 当前窗口工况
        I_k=max(I_down(k),1e-3);
        J_k=max(J_down(k),1e-4);
        T_k=T_K_down(k);
        V_k=V_actual_down(k);
        PH2_k=max(PH2_down(k),1e-3);
        Pair_k=max(Pair_down(k),1e-3);
        
        % 固定 Stage 1 估计得到的 xi1
        xi1_k = xi1_label(k);
        
        % 构造只关于 lambda 的误差函数
        fun = @(lam) voltage_error_single(lam, xi1_k, I_k, J_k, T_k, PH2_k, Pair_k, V_k, A_area, l_thickness);
        
        % 非线性最小二乘反演 lambda
        lambda_label(k) = lsqnonlin(fun, x0_lambda, lambda_lb_global, 14.0, options);
        
        % 当前结果作为下一窗口初值
        x0_lambda = lambda_label(k);
        
        % 根据反演值计算拟合电压
        V_fit(k) = V_k + fun(lambda_label(k));
    end
    
    % 更新下一轮迭代使用的 lambda 轨迹
    lambda_traj = lambda_label;
end

%% ==================== 5. 归一化统计量（供 TS-LSTM 复用） ====================
% 为后续数据驱动模型训练保存均值和标准差
% 常用于 z-score 标准化
norm_stats.I    = [mean(I_down),         std(I_down)];
norm_stats.T    = [mean(T_K_down),       std(T_K_down)];
norm_stats.PH2  = [mean(PH2_down),       std(PH2_down)];
norm_stats.Pair = [mean(Pair_down),      std(Pair_down)];
norm_stats.J    = [mean(J_down),         std(J_down)];
norm_stats.lam  = [mean(lambda_label),   std(lambda_label)];
norm_stats.xi1  = [mean(xi1_label),      std(xi1_label)];

fprintf('\n[归一化统计量]\n');
disp(struct2table(norm_stats, 'AsArray', true));

%% ==================== 6. 可视化 ====================
% 计算拟合残差
res = V_fit - V_actual_down;

figure('Name','FC1 伪标签 v3.1','Position',[50 40 1200 950]);

% (1) 实际电压 vs 拟合电压
subplot(4,2,[1 2]);
plot(V_actual_down,'k-','LineWidth',1.5); hold on;
plot(V_fit,'r--','LineWidth',1.0);
title(sprintf('电压拟合 (RMSE = %.3f mV)', 1000*rms(res)));
ylabel('Cell V');
legend('实际测量','反推拟合','Location','best');
grid on;

% (2) lambda 序列
subplot(4,2,[3 4]);
plot(lambda_label,'b-','LineWidth',1.0);
title('快变 \lambda (保留中频水热动态响应)');
ylabel('\lambda');
ylim([3 14]);
grid on;

% (3) xi1 序列
subplot(4,2,[5 6]);
plot(xi1_label,'m-','LineWidth',1.8);
title('慢变 \xi_1 (S-G平滑，允许刻画电堆可逆恢复特征)');
ylabel('\xi_1');
grid on;

% ---- 诊断子图：lambda 与 J 局部对比 ----
% 找出电流密度变化最大的时刻
dJ = abs(diff(J_down));
[~, k_jump] = max(dJ);

% 截取该处附近局部区间
zoom_range = max(1,k_jump-100) : min(N_windows, k_jump+100);

% (4) lambda 与 J 的局部联动图
subplot(4,2,7);
yyaxis left;
plot(zoom_range, lambda_label(zoom_range), 'b-', 'LineWidth',1.4);
ylabel('\lambda');

yyaxis right;
plot(zoom_range, J_down(zoom_range), 'k-', 'LineWidth',1.0);
ylabel('J (A/cm^2)');

title(sprintf('诊断: \\lambda vs J 局部动态响应 (窗口 k≈%d)', k_jump));
xlabel('Time Window');
grid on;

% (5) 拟合残差序列
subplot(4,2,8);
plot(res*1000,'Color',[0.3 0.3 0.3]);
title('拟合残差序列 V_{fit}-V_{actual} (mV)');
xlabel('Time Window');
ylabel('Error (mV)');
grid on;

%% 保存结果
save('FC1_Full_PseudoLabels.mat', ...
     'I_down','J_down','T_K_down','Pair_down','PH2_down','V_actual_down', ...
     'lambda_label','xi1_label','V_fit','norm_stats','n_cells','A_area','l_thickness');

disp('>>> v3.1 伪标签 + 归一化统计量已保存：FC1_Full_PseudoLabels.mat');

%% ==================== 附录：代价函数 ====================
function err = voltage_error_single(lambda, xi_1, I, J, T, P_H2, P_air, V_actual, A, l)

    % 氧气分压：空气中氧气摩尔分数近似取 21%
    P_O2 = 0.21 * P_air;

    % 1) Nernst 理论可逆电压
    E_nernst = 1.229 ...
             - 0.85e-3*(T - 298.15) ...
             + 4.3085e-5*T*log(P_H2 * sqrt(P_O2));
    
    % 2) 活化极化损失（Mann 2000 经验模型）
    xi_2 = 0.00312;
    xi_3 = 7.4e-5;
    xi_4 = -1.87e-4;
    C_O2 = P_O2 / (5.08e6 * exp(-498/T));
    V_act = -(xi_1 + xi_2*T + xi_3*T*log(C_O2) + xi_4*T*log(I));
    
    % 3) 欧姆极化损失
    % lambda 影响膜含水量，从而影响膜电阻
    num = 181.6*(1 + 0.03*J + 0.062*(T/303)^2*J^2.5);
    den = (lambda - 0.634 - 3*J)*exp(4.18*(T-303)/T);
    R_ohm = (num/den * l)/A;
    V_ohm = I*R_ohm;
    
    % 4) 浓差极化损失
    B = 0.016;
    J_max = 1.5;
    J_safe = min(J, J_max - 1e-3);
    V_conc = -B*log(1 - J_safe/J_max);
    
    % 5) 总理论电压
    V_theory = E_nernst - V_act - V_ohm - V_conc;
    
    % 6) 输出误差：理论电压 - 测量电压
    err = V_theory - V_actual;
end