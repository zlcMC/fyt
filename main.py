import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 孤立森林：无监督异常检测模型
# 作用：
#   根据窗口特征判断“哪些窗口更像离群点 / 异常点”
# 注意：
#   score_samples 输出的分数越低，表示越异常
from sklearn.ensemble import IsolationForest

# 标准化：
#   让不同特征处在相近数值尺度，避免量纲大的特征主导模型
from sklearn.preprocessing import StandardScaler

from adaptive_regime import (
    fit_adaptive_current_thresholds,
    print_adaptive_thresholds,
    is_rest_window_adaptive,
    is_cc_window_adaptive,
    has_extreme_cell_voltage
)


# =========================================================
# 0. 参数配置区
# =========================================================
# 这一段是整个脚本里最常需要看的地方。
#
# 参数大致分成这些类别：
#   1. 输入输出文件路径
#   2. 分块读取参数（防止大文件炸内存）
#   3. 重采样与时间切段参数
#   4. REST / CC 窗口定义参数
#   5. DQ（数据质量）过滤参数
#   6. Isolation Forest 参数
#   7. EWMA 动态阈值参数
#   8. 系统模式模板参数（sys1 / sys6）
#
# 建议：
#   真要调参，优先只改这里；
#   后面的函数逻辑尽量少动。
# =========================================================


# ---------------------------------------------------------
# 原始数据文件路径
# ---------------------------------------------------------
# 当前要处理的 CSV 文件。
# 跑不同系统时，一般只需要改这一行。
# 例如：
#   data_sys_1.csv
#   data_sys_6.csv
#   data_sys_12.csv
# ---------------------------------------------------------


# ---------------------------------------------------------
# 为了先调试，只读取前 NROWS 行
# ---------------------------------------------------------
# 在现在这个“分块版主流程”中：
#   NROWS 主要用于兼容小样本调试
# 真正大文件的全量处理主要靠 CHUNKSIZE。
#
# 设成 None 表示不限制。
# ---------------------------------------------------------
NROWS = None


# ---------------------------------------------------------
# 分块读取参数
# ---------------------------------------------------------
# CHUNKSIZE：
#   一次从 CSV 中读取多少行，避免一次性读入超大文件导致内存爆炸
#
# 常用经验：
#   30万 ~ 50万比较稳
#
# 如果内存小，可以减小；
# 如果内存富余，可以适当加大提高速度。
# ---------------------------------------------------------
CHUNKSIZE = 300000


# ---------------------------------------------------------
# 拟合自适应阈值时使用的样本行数
# ---------------------------------------------------------
# 不是全量读取，只用一部分前部样本去估计：
#   1. 当前系统的电流分布
#   2. 静置 / CC 自适应工况阈值
#   3. 当前系统更像 sys1 还是 sys6
#
# 太小：估计可能不稳
# 太大：启动阶段会变慢
# ---------------------------------------------------------
ADAPTIVE_FIT_NROWS = 500000


# ---------------------------------------------------------
# 重采样间隔
# ---------------------------------------------------------
# 所有连续片段最终会先被统一重采样到这个时间步长。
#
# 例如 "30s" 表示每 30 秒一个点。
#
# 为什么要重采样？
#   原始数据可能有 5s / 60s / 61s 等不同采样周期；
#   如果不统一，后面的窗口特征、dV/dt、EWMA 都会不稳。
# ---------------------------------------------------------
RESAMPLE_RULE = "30s"


# ---------------------------------------------------------
# 时间断点阈值（基础下限）
# ---------------------------------------------------------
# 如果相邻两条记录时间差太大，就认为数据在这里“断开了”，
# 需要切成不同连续片段。
#
# 注意：
#   这个值现在只是最低保护值；
#   真正切段时还会结合当前 chunk 的主采样间隔，自适应放宽。
# ---------------------------------------------------------
TIME_GAP_THRESHOLD_SECONDS = 30


# ---------------------------------------------------------
# 自适应时间断点阈值倍率
# ---------------------------------------------------------
# 实际断点阈值：
#   max(TIME_GAP_THRESHOLD_SECONDS, TIME_GAP_MULTIPLIER * dominant_dt)
#
# 例如：
#   dominant_dt = 5s  -> 阈值 = max(30, 3*5)  = 30s
#   dominant_dt = 61s -> 阈值 = max(30, 3*61) = 183s
#
# 作用：
#   同时兼容 5s / 60s / 61s 等不同采样节奏，
#   避免把 61 秒采样误切成一堆“断片”。
# ---------------------------------------------------------
TIME_GAP_MULTIPLIER = 3.0


# ---------------------------------------------------------
# 微短路方向判定阈值
# ---------------------------------------------------------
# end_dev_signed_min：
#   表示某个窗口结束时，最低单体相对同一时刻中位数电压的偏差
#
# 如果它低于这个阈值，才更偏向：
#   "low_voltage_like_micro_short"
#
# 当前是 -10mV，表示判得比较严格。
# ---------------------------------------------------------
DIRECTION_LOW_DEV_THRESHOLD = -0.010


# ---------------------------------------------------------
# 静置窗口定义参数
# ---------------------------------------------------------
# REST_WINDOW_MINUTES：
#   一个静置窗口的总长度（分钟）
#
# REST_STEP_MINUTES：
#   静置窗口滑动步长（分钟）
#
# 现在：
#   120 分钟窗口，每 60 分钟滑动一次
#   也就是相邻窗口有一定重叠，但不至于重得过头。
# ---------------------------------------------------------
REST_WINDOW_MINUTES = 120
REST_STEP_MINUTES = 60

# REST_FEATURE_TAIL_MINUTES：
#   静置窗口做特征提取时，只取最后多少分钟
#
# 原因：
#   静置刚开始可能存在极化恢复，不够干净；
#   后段更适合提微小差异特征。
# ---------------------------------------------------------
REST_FEATURE_TAIL_MINUTES = 10


# ---------------------------------------------------------
# 近似 CC 充电窗口定义参数
# ---------------------------------------------------------
# CC_WINDOW_MINUTES：
#   一个近似恒流充电窗口长度
#
# CC_STEP_MINUTES：
#   滑动步长
#
# 现在：
#   15 分钟窗口，每 15 分钟滑一次
#   基本相当于不重叠
# ---------------------------------------------------------
CC_WINDOW_MINUTES = 15
CC_STEP_MINUTES = 15


# ---------------------------------------------------------
# 电流方向定义
# ---------------------------------------------------------
# 当前约定：
#   正电流 = 充电
#   负电流 = 放电
#
# 如果以后发现某组数据相反，才改成 -1
# ---------------------------------------------------------
CHARGE_SIGN = 1


# ---------------------------------------------------------
# dV/dt 平滑参数
# ---------------------------------------------------------
# 单体电压在求导前先做 rolling mean 平滑，降低噪声，用窗口内所有电压平均值替代窗口中心时刻电压，完成平滑降噪
# ---------------------------------------------------------
DV_SMOOTH_WINDOW = 3


# ---------------------------------------------------------
# DQ（数据质量）筛查参数
# ---------------------------------------------------------
# DQ_DVDT_ABS_MAX：
#   单体电压变化率绝对值上限
#
# DQ_DV_STEP_MAX：
#   相邻采样点之间，单体电压跳变量上限
#
# 如果超出这些值，更像坏数据/毛刺/采样异常，而不是轻微微短路
# ---------------------------------------------------------
DQ_DVDT_ABS_MAX = 0.1
DQ_DV_STEP_MAX = 0.05


# ---------------------------------------------------------
# 均衡电流过滤阈值
# ---------------------------------------------------------
# 若窗口中某个均衡电流 I_CNV_Cell_* 的绝对值超过此值，
# 则认为均衡在开启。
#
# 均衡会主动扰动单体电压，因此不适合拿来做 IF 微短路筛查。
# ---------------------------------------------------------
BALANCE_CURRENT_ABS_MAX = 0.01


# ---------------------------------------------------------
# 工况纯度约束：REST
# ---------------------------------------------------------
# 仅仅电流接近 0 还不够，还要额外要求：
#   - 包压变化小
#   - SOC 变化小
#   - 单体中位数变化小
#
# 用于过滤：
#   - 过渡段
#   - CV 尾段
#   - 看起来像静置但其实不纯的窗口
# ---------------------------------------------------------
REST_PACK_DV_ABS_MAX = 0.03
REST_SOC_DELTA_ABS_MAX = 0.05
REST_MEDIAN_CELL_DV_ABS_MAX = 0.006


# ---------------------------------------------------------
# 工况纯度约束：CC
# ---------------------------------------------------------
# 近似恒流充电窗口除了电流符合条件，
# 还要求包压 / SOC / 单体中位数趋势不要明显违背“充电”方向。
# ---------------------------------------------------------
CC_PACK_DV_MIN = 0.02
CC_SOC_DELTA_MIN = -0.02
CC_MEDIAN_CELL_DV_MIN = -0.001


# ---------------------------------------------------------
# 通道质量硬过滤
# ---------------------------------------------------------
# 若某单体相对同一时刻所有单体中位数的偏离过大，
# 更像：
#   - 通道异常
#   - 传感器偏置
#   - 采样问题
# 而不太像轻微微短路
# ---------------------------------------------------------
CHANNEL_DEV_HARD_MAX = 0.12


# ---------------------------------------------------------
# 方向标签自适应阈值参数
# ---------------------------------------------------------
# 后面不是只用固定阈值，而是会结合当前数据中
# end_dev_signed_min 的分布，自动估一个更合适的方向阈值。
# ---------------------------------------------------------
DIRECTION_BASELINE_QUANTILE = 0.01
DIRECTION_MAD_K = 4.0


# ---------------------------------------------------------
# IF 健康训练池参数
# ---------------------------------------------------------
# 虽然 Isolation Forest 是无监督模型，
# 但这里会优先挑“更像健康”的窗口去训练 IF，
# 减少被异常工况污染导致的过敏。
# ---------------------------------------------------------
HEALTHY_TRAIN_QUANTILE = 0.90
MIN_HEALTHY_TRAIN_WINDOWS = 100


# ---------------------------------------------------------
# 分块 carry 参数
# ---------------------------------------------------------
# 分块处理时，上一块尾部会保留一小段数据，
# 用来和下一块拼接，防止窗口被 chunk 边界切断。
# ---------------------------------------------------------
CHUNK_CARRY_MARGIN_SECONDS = 120


# ---------------------------------------------------------
# 边缘侧是否缓存所有重采样片段
# ---------------------------------------------------------
# True：
#   可以回查报警窗口对应的原始/重采样曲线
#   但会更占内存
#
# False：
#   更省内存
# ---------------------------------------------------------
ENABLE_RAW_SEGMENT_CACHE = False

# EVENT_MERGE_GAP_MINUTES：
#   两个报警窗口之间如果时间间隔不超过这个值，
#   则认为可以合并成同一事件
# ---------------------------------------------------------
EVENT_MERGE_GAP_MINUTES = 20


# ---------------------------------------------------------
# Isolation Forest 参数
# ---------------------------------------------------------
N_ESTIMATORS = 200
RANDOM_STATE = 42


# ---------------------------------------------------------
# EWMA 动态阈值参数
# ---------------------------------------------------------
# INIT_HEALTHY_WINDOWS：
#   初始化 EWMA 基线时，先拿多少个高置信健康窗口做起点
#
# EWMA_ALPHA：
#   EWMA 基线更新速度
#   越小越稳
#
# THRESHOLD_K：
#   threshold = mu - k * sigma
#   k 越大，阈值越低，越保守
#
# CONSECUTIVE_TRIGGER_N：
#   连续多少个窗口跌破阈值才正式 trigger_alarm = 1
# ---------------------------------------------------------
INIT_HEALTHY_WINDOWS = 50
EWMA_ALPHA = 0.01
THRESHOLD_K = 4.0
CONSECUTIVE_TRIGGER_N = 5


# ---------------------------------------------------------
# EWMA 稳定性保护参数
# ---------------------------------------------------------
# EWMA_SIGMA_FLOOR：
#   sigma 下限，防止阈值过度贴近均值
#
# EWMA_UPDATE_Z_MAX：
#   若当前点离基线太远，即使没跌破阈值，也不让它更新基线
#
# EWMA_MU_MAX_STEP_UP：
#   限制 mu 单步上升幅度，防止阈值突然被整体抬高
# ---------------------------------------------------------
EWMA_SIGMA_FLOOR = 0.03
EWMA_UPDATE_Z_MAX = 2.5
EWMA_MU_MAX_STEP_UP = 0.002


# ---------------------------------------------------------
# 系统模式参数模板
# ---------------------------------------------------------
# PARAM_TEMPLATE_SYS1：
#   更适合普通型 / 相对稳定型系统
#   特点：更敏感
#
# PARAM_TEMPLATE_SYS6：
#   更适合高噪声 / 混采样 / 更复杂系统
#   特点：更保守
# ---------------------------------------------------------
# ---------------------------------------------------------
# 手工强制系统模式
# ---------------------------------------------------------
# 可选：
#   None   -> 使用自动判别
#   "sys1" -> 强制使用 sys1 模板
#   "sys6" -> 强制使用 sys6 模板
FORCE_SYSTEM_MODE = None

PARAM_TEMPLATE_SYS1 = {
    "THRESHOLD_K": 2.0,
    "CONSECUTIVE_TRIGGER_N": 2,
    "EWMA_SIGMA_FLOOR": 0.02,
    "DIRECTION_LOW_DEV_THRESHOLD": -0.003
}

PARAM_TEMPLATE_SYS6 = {
    "THRESHOLD_K": 4.0,
    "CONSECUTIVE_TRIGGER_N": 5,
    "EWMA_SIGMA_FLOOR": 0.03,
    "DIRECTION_LOW_DEV_THRESHOLD": -0.010
}

# ---------------------------------------------------------
# 输出文件路径
# ---------------------------------------------------------
# dq_filtered_windows.csv
#   - 被数据质量检查（DQ）筛掉的窗口记录表
#   - 里面会写明 dq_reason
#   - 后续可以用它分析：哪些窗口被均衡、通道异常、坏数据等原因拦掉了
#
# combined_rest_cc_results.csv
#   - REST 与 CC 两个分支最终结果的合并表
#   - 是最核心的综合结果之一
#   - 里面通常会包含：
#       time_start / time_end
#       if_score
#       dynamic_threshold
#       trigger_alarm
#       direction_flag
#       model_branch 等
# ---------------------------------------------------------
# 输出文件
# ---------------------------------------------------------
# rest_window_features.csv
#   - REST 窗口特征表
#   - 这是 REST 分支进入 IF 之前的输入特征表
#
# cc_window_features.csv
#   - CC 窗口特征表
#
# rest_iforest_scores.csv
#   - REST 分支经过 IF + EWMA 后的最终结果表
#
# cc_iforest_scores.csv
#   - CC 分支经过 IF + EWMA 后的最终结果表
# ---------------------------------------------------------

# ---------------------------------------------------------
# 事件级输出参数
# ---------------------------------------------------------
# OUTPUT_EVENT_PATH：
#   事件级报警输出文件
#   它不是单个窗口，而是把重叠/接近的报警窗口合并成“事件”
#
# ---------------------------------------------------------
# 输出文件路径
# ---------------------------------------------------------
# dq_filtered_windows.csv
#   - 被数据质量检查（DQ）筛掉的窗口记录表
#   - 里面会写明 dq_reason
#   - 后续可以用它分析：哪些窗口被均衡、通道异常、坏数据等原因拦掉了
#
# combined_rest_cc_results.csv
#   - REST 与 CC 两个分支最终结果的合并表
#   - 是最核心的综合结果之一
#   - 里面通常会包含：
#       time_start / time_end
#       if_score
#       dynamic_threshold
#       trigger_alarm
#       direction_flag
#       model_branch 等

# ---------------------------------------------------------
# 当前系统的输出目录
# ---------------------------------------------------------
# 建议每个 sys 单独一个文件夹，避免不同系统结果互相覆盖
FILE_PATH = r"F:\BMS\field_data\field_data\data_sys_28.csv"
OUTPUT_DIR = r"edge_outputs/sys_28"

# DQ 过滤窗口输出
OUTPUT_DQ_PATH = f"{OUTPUT_DIR}/dq_filtered_windows.csv"

# REST / CC 合并总表输出
OUTPUT_COMBINED_RESULT_PATH = f"{OUTPUT_DIR}/combined_rest_cc_results.csv"

# 事件级输出
OUTPUT_EVENT_PATH = f"{OUTPUT_DIR}/micro_short_candidate_events.csv"

# REST / CC 特征表输出
OUTPUT_REST_FEATURE_PATH = f"{OUTPUT_DIR}/rest_window_features.csv"
OUTPUT_CC_FEATURE_PATH = f"{OUTPUT_DIR}/cc_window_features.csv"

# REST / CC IF 结果输出
OUTPUT_REST_SCORE_PATH = f"{OUTPUT_DIR}/rest_iforest_scores.csv"
OUTPUT_CC_SCORE_PATH = f"{OUTPUT_DIR}/cc_iforest_scores.csv"

# =========================================================
# 1. 读取数据（小样本/调试用，与原版兼容）
# =========================================================
def load_data(file_path, nrows=None):
    """
    从 CSV 中读取原始数据，并打印基本信息。

    作用
    ---------------------------------------------------------
    这个函数主要用于：
        1. 小样本快速读取
        2. 看列名
        3. 调试
        4. 读取前部样本做字段识别 / 阈值拟合

    注意
    ---------------------------------------------------------
    超大文件的全量处理不要靠它一次性读完；
    真正的大文件全量处理走 iterate_csv_chunks()。
    """
    print(f"读取文件: {file_path}")
    df = pd.read_csv(file_path, nrows=nrows)

    print("原始数据形状:", df.shape)
    print("列名:")
    print(df.columns.tolist())

    return df


# =========================================================
# 1.1 分块读取 CSV（核心省内存机制）
# =========================================================
def iterate_csv_chunks(file_path, chunksize=300000):
    """
    分块读取 CSV，只读取真正需要的列，并尽量降低内存占用。

    设计目的
    ---------------------------------------------------------
    原始文件可能有几百万行。
    如果一次性读入：
        - 内存容易爆
        - 速度慢
        - 程序不稳定

    所以这里做了三层优化：
        1. 先只读表头，识别列名
        2. 用 usecols 只读必要列
        3. 数值列统一转 float32，减小内存占用

    返回
    ---------------------------------------------------------
    逐块 yield：
        (chunk_idx, chunk_df)
    """
    # 1. 先读表头，获得全部列名
    header_df = pd.read_csv(file_path, nrows=0)
    cols = header_df.columns.tolist()

    # 2. 只保留我们真正需要的列
    usecols = []
    for c in cols:
        if (
            c == "Timestamp"
            or c == "U_Battery"
            or c == "I_Battery"
            or "SOC" in c
            or "Temperat" in c
            or c.startswith("U_Cell_")
            or c.startswith("I_CNV_Cell_")
        ):
            usecols.append(c)

    # 3. 数值列转 float32，时间列保持字符串/原始格式后续再解析
    dtype_map = {}
    for c in usecols:
        if c != "Timestamp":
            dtype_map[c] = "float32"

    # 4. 创建分块读取器
    reader = pd.read_csv(
        file_path,
        usecols=usecols,
        dtype=dtype_map,
        chunksize=chunksize
    )

    # 5. 逐块返回
    for chunk_idx, chunk in enumerate(reader, start=1):
        mem_mb = chunk.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"\n[CHUNK] 读取第 {chunk_idx} 个chunk, 形状={chunk.shape}, 内存约={mem_mb:.2f} MB")
        yield chunk_idx, chunk


# =========================================================
# 2. 自动识别字段
# =========================================================
def identify_columns(df):
    """
    自动识别关键列：
        - 时间列
        - 总压
        - 总流
        - SOC
        - 温度列
        - 单体电压列
        - 均衡电流列

    返回
    ---------------------------------------------------------
    info : dict
        包含后续所有流程会反复用到的字段映射
    """
    cols = df.columns.tolist()

    time_col = next((c for c in cols if "Timestamp" in c or "time" in c.lower()), None)
    u_battery_col = next((c for c in cols if c == "U_Battery" or "U_Battery" in c), None)
    i_battery_col = next((c for c in cols if c == "I_Battery" or "I_Battery" in c), None)
    soc_col = next((c for c in cols if "SOC" in c), None)

    temp_cols = [c for c in cols if "Temperat" in c]
    cell_u_cols = [c for c in cols if c.startswith("U_Cell_")]
    cnv_i_cols = [c for c in cols if c.startswith("I_CNV_Cell_")]

    # 排序，保证 U_Cell_1, U_Cell_2 ... U_Cell_8 顺序正确
    cell_u_cols = sorted(cell_u_cols, key=lambda x: int(x.split("_")[-1]))
    cnv_i_cols = sorted(cnv_i_cols, key=lambda x: int(x.split("_")[-1]))

    info = {
        "time_col": time_col,
        "u_battery_col": u_battery_col,
        "i_battery_col": i_battery_col,
        "soc_col": soc_col,
        "temp_cols": temp_cols,
        "cell_u_cols": cell_u_cols,
        "cnv_i_cols": cnv_i_cols
    }

    print("\n识别字段结果:")
    for k, v in info.items():
        print(f"{k}: {v}")

    return info


def validate_required_columns(info):
    """
    提前检查“微短路筛查”必须字段是否存在。

    必须字段
    ---------------------------------------------------------
    1. 时间列
    2. 总电流列
    3. 至少一个单体电压列

    为什么提前检查？
    ---------------------------------------------------------
    如果这些缺了，不如尽早报错；
    否则后面在 DQ / 特征提取深处才崩，更难排查。
    """
    missing = []

    if info["time_col"] is None:
        missing.append("时间列 Timestamp/time")
    if info["i_battery_col"] is None:
        missing.append("总电流列 I_Battery")
    if len(info["cell_u_cols"]) == 0:
        missing.append("单体电压列 U_Cell_*")

    if missing:
        raise ValueError("缺少微短路辨识必要字段: " + ", ".join(missing))

    if info["u_battery_col"] is None:
        print("[WARN] 未识别到 U_Battery，rest/CC 工况纯度会少一个包压趋势约束。")
    if info["soc_col"] is None:
        print("[WARN] 未识别到 SOC，rest/CC 工况纯度会少一个 SOC 趋势约束。")
    if len(info["cnv_i_cols"]) == 0:
        print("[WARN] 未识别到 I_CNV_Cell_*，无法剔除均衡开启窗口。")


# =========================================================
# 3. 基础预处理（原版，保留兼容，仅小样本用）
# =========================================================
def basic_preprocess(df, info):
    """
    原版基础预处理。
    主要给小样本兼容使用。

    处理内容
    ---------------------------------------------------------
    1. 只保留需要的列
    2. 时间列转 datetime
    3. 数值列转 numeric
    4. 按时间排序
    5. 前向/后向填充
    6. 丢掉仍有缺失的行
    """
    selected_cols = []

    if info["time_col"] is not None:
        selected_cols.append(info["time_col"])
    if info["u_battery_col"] is not None:
        selected_cols.append(info["u_battery_col"])
    if info["i_battery_col"] is not None:
        selected_cols.append(info["i_battery_col"])
    if info["soc_col"] is not None:
        selected_cols.append(info["soc_col"])

    selected_cols += info["temp_cols"]
    selected_cols += info["cell_u_cols"]
    selected_cols += info["cnv_i_cols"]

    df = df[selected_cols].copy()

    if info["time_col"] is None:
        raise ValueError("未识别到时间列，无法继续。")

    df[info["time_col"]] = pd.to_datetime(df[info["time_col"]], errors="coerce")
    df = df.dropna(subset=[info["time_col"]]).copy()

    for c in df.columns:
        if c != info["time_col"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(info["time_col"]).reset_index(drop=True)
    df = df.ffill().bfill().dropna().reset_index(drop=True)

    print("\n基础预处理后数据形状:", df.shape)
    return df


# =========================================================
# 3.1 chunk 版基础预处理（轻量、省内存）
# =========================================================
def basic_preprocess_chunk(df, info):
    """
    针对单个 chunk 的轻量预处理版本。

    相比 basic_preprocess() 的特点
    ---------------------------------------------------------
    1. 尽量少复制数据
    2. 数值列统一压成 float32 节省内存
    3. 只有在时间乱序时才排序，避免额外开销
    """
    selected_cols = []

    if info["time_col"] is not None:
        selected_cols.append(info["time_col"])
    if info["u_battery_col"] is not None:
        selected_cols.append(info["u_battery_col"])
    if info["i_battery_col"] is not None:
        selected_cols.append(info["i_battery_col"])
    if info["soc_col"] is not None:
        selected_cols.append(info["soc_col"])

    selected_cols += info["temp_cols"]
    selected_cols += info["cell_u_cols"]
    selected_cols += info["cnv_i_cols"]

    df = df.loc[:, selected_cols]

    time_col = info["time_col"]
    if time_col is None:
        raise ValueError("未识别到时间列，无法继续。")

    # 时间列解析，无法解析的时间直接删掉
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # 数值列转 float32，省内存
    for c in df.columns:
        if c != time_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            df[c] = df[c].astype("float32")
    # 只有在时间不递增时才排序
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(time_col, kind="mergesort")

    # 前后填充，再把仍然缺失的行丢掉
    df = df.ffill().bfill().dropna()
    df = df.reset_index(drop=True)

    return df


# =========================================================
# 4. 按时间断点切连续片段
# =========================================================
def split_continuous_segments(df, info, gap_threshold_seconds=30):
    """
    根据相邻时间差，把数据切成多个连续片段。

    设计目的
    ---------------------------------------------------------
    如果时间差太大，说明中间这段时间没数据了；
    这时前后数据不应该当作同一个连续序列去做重采样和窗口滑动。

    参数
    ---------------------------------------------------------
    gap_threshold_seconds :
        若相邻时间差 > 这个阈值，就认为断开
    """
    time_col = info["time_col"]

    df = df.copy()
    df["dt_seconds"] = df[time_col].diff().dt.total_seconds()
    df["is_break"] = (df["dt_seconds"].isna()) | (df["dt_seconds"] > gap_threshold_seconds)
    df["segment_id"] = df["is_break"].cumsum()

    segments = []
    for seg_id, seg_df in df.groupby("segment_id"):
        seg_df = seg_df.drop(columns=["dt_seconds", "is_break", "segment_id"]).reset_index(drop=True)
        if len(seg_df) >= 2:
            segments.append(seg_df)

    print(f"[SPLIT] 连续片段数量: {len(segments)}")
    if len(segments) > 0:
        print("[SPLIT] 前几个片段长度:", [len(s) for s in segments[:5]])

    return segments


# =========================================================
# 5. 单连续片段重采样
# =========================================================
def resample_segment(seg_df, info, rule="10s"):
    """
    对单个连续片段做重采样。

    步骤
    ---------------------------------------------------------
    1. 以时间列做索引
    2. 只保留数值列
    3. 按 rule 重采样取均值
    4. 用时间插值补齐
    """
    time_col = info["time_col"]
    seg = seg_df.copy().set_index(time_col).sort_index()
    numeric_cols = seg.select_dtypes(include=[np.number]).columns.tolist()
    seg = seg[numeric_cols]
    resampled = seg.resample(rule).mean()
    resampled = resampled.interpolate(method="time").ffill().bfill()
    resampled = resampled.reset_index()
    return resampled


def resample_all_segments(segments, info, rule="10s"):
    """
    对所有连续片段逐个重采样。

    若某个片段重采样失败，则跳过，不影响整体流程。
    """
    resampled_segments = []
    for i, seg in enumerate(segments, start=1):
        try:
            rs = resample_segment(seg, info, rule=rule)
            if len(rs) >= 2:
                resampled_segments.append(rs)
        except Exception as e:
            print(f"片段 {i} 重采样失败，跳过。原因: {e}")

    print(f"[RESAMPLE] 重采样后可用片段数量: {len(resampled_segments)}")
    if len(resampled_segments) > 0:
        print("[RESAMPLE] 前几个重采样片段长度:", [len(s) for s in resampled_segments[:5]])

    return resampled_segments


# =========================================================
# 7. 单体电压平滑
# =========================================================
def smooth_cell_voltage(cell_data, smooth_window=3):
    """
    对单体电压做简单 rolling mean 平滑，降低差分求 dV/dt 时的噪声。
    """
    smoothed = cell_data.rolling(
        window=smooth_window,
        center=True,
        min_periods=1
    ).mean()
    return smoothed


# =========================================================
# 8. DQ 数据质量检查
# =========================================================
def dq_check_window(win, info):
    """
    对窗口做基础数据质量检查。

    检查内容
    ---------------------------------------------------------
    1. 单体电压单步跳变是否过大（dv_step）
    2. 单体电压变化率是否过大（dvdt）

    如果过大，更像：
        - 采样毛刺
        - 数据异常
        - 通道问题
    而不适合直接进入正常 IF 流程
    """
    cell_cols = info["cell_u_cols"]
    time_col = info["time_col"]

    cell_data = win[cell_cols].copy()

    dv = cell_data.diff().abs()
    dt = win[time_col].diff().dt.total_seconds().replace(0, np.nan)
    dvdt = cell_data.diff().div(dt, axis=0).abs()

    max_dv = np.nanmax(dv.values)
    max_dv_col = dv.stack().idxmax()[1] if not dv.dropna(how="all").empty else None

    max_dvdt = np.nanmax(dvdt.values)
    max_dvdt_col = dvdt.stack().idxmax()[1] if not dvdt.dropna(how="all").empty else None

    if max_dv > DQ_DV_STEP_MAX:
        return 1, f"dv_step_too_large: {max_dv_col}, max_dv={max_dv:.6f}V"

    if max_dvdt > DQ_DVDT_ABS_MAX:
        return 1, f"dvdt_too_large: {max_dvdt_col}, max_dvdt={max_dvdt:.6f}V/s"

    return 0, "pass"


# =========================================================
# 9. 静置窗口特征提取
# =========================================================
def extract_rest_features(win, info):
    """
    提取静置窗口（rest）的特征。

    这些特征主要围绕：
    ---------------------------------------------------------
    1. 静置压降一致性
    2. 温度一致性
    3. dV/dt 一致性
    4. 方向信息（低压方向还是高压方向）

    返回
    ---------------------------------------------------------
    feat : dict
        一行窗口特征
    """
    feat = {}

    # 只取静置窗口最后一段做特征提取，避开静置初期极化恢复
    tail_points = int((REST_FEATURE_TAIL_MINUTES * 60) / get_resample_seconds(RESAMPLE_RULE))
    if len(win) > tail_points:
        win = win.iloc[-tail_points:].copy()

    cell_cols = info["cell_u_cols"]
    temp_cols = info["temp_cols"]
    time_col = info["time_col"]

    cell_data = win[cell_cols].copy()

    # 1. 静置压降相对中位数残差
    # -----------------------------------------------------
    # 先看每个单体从窗口开始到窗口结束掉了多少电压
    drop = cell_data.iloc[0] - cell_data.iloc[-1]

    # 取所有单体掉压的中位数作为参考
    drop_median = np.median(drop.values)

    # 计算每个单体掉压相对中位数的偏差绝对值
    drop_res = (drop - drop_median).abs()

    feat["rest_drop_res_mean"] = drop_res.mean()
    feat["rest_drop_res_max"] = drop_res.max()

    # 保留带方向的差值，方便后面解释
    drop_diff = drop - drop_median
    feat["rest_drop_diff_min"] = drop_diff.min()
    feat["rest_drop_diff_max"] = drop_diff.max()

    # 哪个单体掉压最大，记为一个候选单体
    max_drop_cell = drop.idxmax()
    feat["candidate_cell_by_drop"] = max_drop_cell

    # 2. 温度偏差，是静置窗口才进入该分支
    # -----------------------------------------------------
    # 温度列存在时，计算每个时刻各温度相对均值的偏差
    if len(temp_cols) > 0:
        temp_data = win[temp_cols].copy()
        temp_mean_row = temp_data.mean(axis=1)
        temp_dev = temp_data.sub(temp_mean_row, axis=0).abs()

        feat["temp_dev_mean"] = temp_dev.mean().mean()
        feat["temp_dev_max"] = temp_dev.max().max()
    else:
        feat["temp_dev_mean"] = 0.0
        feat["temp_dev_max"] = 0.0

    # 3. dV/dt 一致性残差
    # -----------------------------------------------------
    # 对单体电压先平滑，再求 dV/dt
    cell_smooth = smooth_cell_voltage(cell_data, smooth_window=DV_SMOOTH_WINDOW)

    dt = win[time_col].diff().dt.total_seconds()
    dt = dt.replace(0, np.nan)

    dv = cell_smooth.diff()
    dvdt = dv.div(dt, axis=0)

    # 每个时刻取所有单体 dV/dt 的中位数
    dvdt_median = dvdt.median(axis=1)

    # 看每个单体相对中位数的偏差
    dvdt_res = dvdt.sub(dvdt_median, axis=0).abs()

    feat["dvdt_res_mean"] = np.nanmean(dvdt_res.values)
    feat["dvdt_res_max"] = np.nanmax(dvdt_res.values)

    # 4. 微短路方向解释信息
    # -----------------------------------------------------
    # 看窗口末端各单体电压相对中位数的偏差
    end_v = cell_data.iloc[-1]
    end_v_median = np.median(end_v.values)
    end_dev_signed = end_v - end_v_median

    feat["end_dev_signed_min"] = end_dev_signed.min()
    feat["end_dev_signed_max"] = end_dev_signed.max()

    # 末端最低电压单体
    lowest_cell = end_v.idxmin()
    feat["candidate_cell_by_voltage"] = lowest_cell

    # 用当前方向阈值先打一版方向标签，偏小得越多越接近于微短路
    if end_dev_signed.min() <= DIRECTION_LOW_DEV_THRESHOLD:
        feat["direction_flag"] = "low_voltage_like_micro_short"
    else:
        feat["direction_flag"] = "high_voltage_or_non_micro_short"

    return feat


# =========================================================
# 10. CC窗口特征提取
# =========================================================
def extract_cc_features(win, info):
    """
    提取近似 CC 充电窗口的特征。

    与 rest 的区别
    ---------------------------------------------------------
    CC 更关注：
        - 末端单体偏差
        - 温度偏差
        - dV/dt 一致性偏差
    """
    feat = {}

    cell_cols = info["cell_u_cols"]
    temp_cols = info["temp_cols"]
    time_col = info["time_col"]

    cell_data = win[cell_cols].copy()

    # 1. 末端电压偏差
    end_v = cell_data.iloc[-1]
    end_v_median = np.median(end_v.values)

    end_dev_abs = (end_v - end_v_median).abs()
    feat["cc_end_dev_mean"] = end_dev_abs.mean()
    feat["cc_end_dev_max"] = end_dev_abs.max()

    # 2. 带方向的末端偏差
    end_dev_signed = end_v - end_v_median
    feat["end_dev_signed_min"] = end_dev_signed.min()
    feat["end_dev_signed_max"] = end_dev_signed.max()

    lowest_cell = end_v.idxmin()
    feat["candidate_cell_by_voltage"] = lowest_cell

    if end_dev_signed.min() <= DIRECTION_LOW_DEV_THRESHOLD:
        feat["direction_flag"] = "low_voltage_like_micro_short"
    else:
        feat["direction_flag"] = "high_voltage_or_non_micro_short"

    # 3. 温度偏差
    if len(temp_cols) > 0:
        temp_data = win[temp_cols].copy()
        temp_mean_row = temp_data.mean(axis=1)
        temp_dev = temp_data.sub(temp_mean_row, axis=0).abs()
        feat["temp_dev_mean"] = temp_dev.mean().mean()
        feat["temp_dev_max"] = temp_dev.max().max()
    else:
        feat["temp_dev_mean"] = 0.0
        feat["temp_dev_max"] = 0.0

    # 4. dV/dt 一致性残差
    cell_smooth = smooth_cell_voltage(cell_data, smooth_window=DV_SMOOTH_WINDOW)
    dt = win[time_col].diff().dt.total_seconds().replace(0, np.nan)
    dv = cell_smooth.diff()
    dvdt = dv.div(dt, axis=0)
    dvdt_median = dvdt.median(axis=1)
    dvdt_res = dvdt.sub(dvdt_median, axis=0).abs()

    feat["dvdt_res_mean"] = np.nanmean(dvdt_res.values)
    feat["dvdt_res_max"] = np.nanmax(dvdt_res.values)

    return feat


def get_resample_seconds(rule="10s"):
    """
    把 "30s" 这种字符串转换成秒数整数。
    """
    td = pd.to_timedelta(rule)
    return int(td.total_seconds())


def get_max_window_seconds():
    """
    返回当前 rest / cc 两种窗口中更长的那个窗口长度（秒）。
    """
    return int(max(REST_WINDOW_MINUTES, CC_WINDOW_MINUTES) * 60)


def get_chunk_carry_seconds():
    """
    返回分块拼接时，上一块尾部至少要保留多少秒。

    组成：
    ---------------------------------------------------------
    1. 最长窗口长度
    2. 时间断点保护值
    3. 平滑所需时间
    4. 额外 margin
    """
    smooth_seconds = DV_SMOOTH_WINDOW * get_resample_seconds(RESAMPLE_RULE)
    return get_max_window_seconds() + TIME_GAP_THRESHOLD_SECONDS + smooth_seconds + CHUNK_CARRY_MARGIN_SECONDS


def series_delta(series):
    """
    取一个序列“最后一个值 - 第一个值”。

    用于计算：
        - 包压变化
        - SOC 变化
        - 单体中位数变化
    """
    if series is None or len(series) < 2:
        return np.nan
    return float(series.iloc[-1] - series.iloc[0])


def has_balancing_current(win, info):
    """
    判断当前窗口里是否存在均衡开启。

    方法：
    ---------------------------------------------------------
    看所有 I_CNV_Cell_* 的绝对值，
    只要有一个超过 BALANCE_CURRENT_ABS_MAX，就认为均衡在工作。
    """
    cnv_cols = info["cnv_i_cols"]
    if len(cnv_cols) == 0:
        return False, "pass"

    cnv_data = win[cnv_cols].abs()
    if cnv_data.empty:
        return False, "pass"

    max_abs = float(np.nanmax(cnv_data.values))
    if not np.isfinite(max_abs) or max_abs <= BALANCE_CURRENT_ABS_MAX:
        return False, "pass"

    max_col = cnv_data.stack().idxmax()[1]
    return True, f"balancing_active: {max_col}, max_abs={max_abs:.6f}A"


def get_pack_voltage_delta(win, info):
    """
    计算窗口内包压变化量（末值 - 首值）。
    """
    col = info["u_battery_col"]
    if col is None or col not in win.columns:
        return np.nan
    return series_delta(win[col])


def get_soc_delta(win, info):
    """
    计算窗口内 SOC 变化量（末值 - 首值）。
    """
    col = info["soc_col"]
    if col is None or col not in win.columns:
        return np.nan
    return series_delta(win[col])


def get_median_cell_delta(win, info):
    """
    计算窗口内“单体中位数电压”的变化量（末值 - 首值）。
    """
    cell_cols = info["cell_u_cols"]
    if len(cell_cols) == 0:
        return np.nan
    median_v = win[cell_cols].median(axis=1)
    return series_delta(median_v)


def passes_rest_trend_constraints(win, info):
    """
    判断窗口是否满足 REST 工况纯度约束。

    逻辑：
    ---------------------------------------------------------
    静置不仅要电流小，还应该：
        - 包压变化不大
        - SOC 变化不大
        - 单体中位数变化不大
    """
    pack_dv = get_pack_voltage_delta(win, info)
    soc_delta = get_soc_delta(win, info)
    cell_median_dv = get_median_cell_delta(win, info)

    if np.isfinite(pack_dv) and abs(pack_dv) > REST_PACK_DV_ABS_MAX:
        return False
    if np.isfinite(soc_delta) and abs(soc_delta) > REST_SOC_DELTA_ABS_MAX:
        return False
    if np.isfinite(cell_median_dv) and abs(cell_median_dv) > REST_MEDIAN_CELL_DV_ABS_MAX:
        return False

    return True


def passes_cc_trend_constraints(win, info):
    """
    判断窗口是否满足 CC 工况纯度约束。

    逻辑：
    ---------------------------------------------------------
    近似恒流充电除了电流满足条件，
    还应满足：
        - 包压至少不要明显不升
        - SOC 至少不要明显下降
        - 单体中位数电压至少不要明显下降
    """
    pack_dv = get_pack_voltage_delta(win, info)
    soc_delta = get_soc_delta(win, info)
    cell_median_dv = get_median_cell_delta(win, info)

    if np.isfinite(pack_dv) and pack_dv < CC_PACK_DV_MIN:
        return False
    if np.isfinite(soc_delta) and soc_delta < CC_SOC_DELTA_MIN:
        return False
    if np.isfinite(cell_median_dv) and cell_median_dv < CC_MEDIAN_CELL_DV_MIN:
        return False

    return True


def dq_check_channel_deviation(win, info):
    """
    检查是否存在“某个单体长期偏离同一时刻中位数过大”的情况。

    作用：
    ---------------------------------------------------------
    这种情况更像通道/采样异常，而不太像微短路。
    """
    cell_cols = info["cell_u_cols"]
    if len(cell_cols) == 0:
        return 1, "no_cell_voltage_columns"

    cell_data = win[cell_cols]
    row_median = cell_data.median(axis=1)
    dev = cell_data.sub(row_median, axis=0).abs()
    max_dev = float(np.nanmax(dev.values))

    if np.isfinite(max_dev) and max_dev > CHANNEL_DEV_HARD_MAX:
        max_col = dev.stack().idxmax()[1]
        return 1, f"channel_dev_too_large: {max_col}, max_dev={max_dev:.6f}V"

    return 0, "pass"


def run_full_dq_check(win, info):
    """
    对窗口执行完整 DQ 检查。

    顺序：
    ---------------------------------------------------------
    1. 极值电压检查
    2. 均衡电流检查
    3. 通道偏差检查
    4. 基础 dv / dvdt 检查

    一旦某一步不通过，就直接返回。
    """
    extreme_flag, extreme_reason = has_extreme_cell_voltage(win, info)
    if extreme_flag:
        return 1, extreme_reason

    balance_flag, balance_reason = has_balancing_current(win, info)
    if balance_flag:
        return 1, balance_reason

    channel_flag, channel_reason = dq_check_channel_deviation(win, info)
    if channel_flag == 1:
        return 1, channel_reason

    return dq_check_window(win, info)


def add_common_window_meta(win, info, window_type, segment_id, start, end, dq_flag, dq_reason):
    """
    给一个窗口生成通用元信息。

    这些信息后面会和特征一起保存，方便：
        - 排查
        - 画图
        - 事件合并
        - 回查时间段
    """
    return {
        "window_type": window_type,
        "segment_id": segment_id,
        "window_start_in_segment": start,
        "window_end_in_segment": end - 1,
        "time_start": win[info["time_col"]].iloc[0],
        "time_end": win[info["time_col"]].iloc[-1],
        "dq_flag": dq_flag,
        "dq_reason": dq_reason
    }


def should_skip_emitted_window(win, info, min_time_end_exclusive):
    """
    用于分块处理时避免重复输出已经处理过的窗口。

    如果当前窗口的结束时间 <= 上一轮已输出的最大结束时间，
    就跳过。
    """
    if min_time_end_exclusive is None:
        return False
    return win[info["time_col"]].iloc[-1] <= min_time_end_exclusive


def compute_direction_threshold(feature_df):
    """
    根据当前特征表，自适应估计“低压方向阈值”。

    设计思想：
    ---------------------------------------------------------
    固定 -3mV / -10mV 并不总适合所有组。
    所以这里会结合：
        - 分位数
        - MAD（稳健标准差）
    自动估一个更合理的阈值。
    """
    dev = pd.to_numeric(feature_df["end_dev_signed_min"], errors="coerce").dropna()
    if len(dev) < 10:
        return DIRECTION_LOW_DEV_THRESHOLD

    median_dev = float(dev.median())
    mad = float(np.median(np.abs(dev - median_dev)))
    robust_sigma = 1.4826 * mad
    quantile_threshold = float(dev.quantile(DIRECTION_BASELINE_QUANTILE))

    if robust_sigma > 0:
        robust_threshold = median_dev - DIRECTION_MAD_K * robust_sigma
        return float(min(DIRECTION_LOW_DEV_THRESHOLD, quantile_threshold, robust_threshold))

    return float(min(DIRECTION_LOW_DEV_THRESHOLD, quantile_threshold))


def apply_adaptive_direction_labels(feature_df, model_type):
    """
    对特征表重新打“方向标签”。

    为什么要重打？
    ---------------------------------------------------------
    在 extract_rest_features / extract_cc_features 里，
    方向标签先用的是固定阈值。

    这里会再根据整张表的分布，自适应估一个方向阈值，
    然后统一重打一遍方向标签，更稳一些。
    """
    if feature_df is None or len(feature_df) == 0:
        return feature_df

    df = feature_df.copy()
    threshold = compute_direction_threshold(df)
    df["direction_threshold"] = threshold

    low_dev = pd.to_numeric(df["end_dev_signed_min"], errors="coerce") <= threshold

    # 对 rest 来说，再额外要求：
    # 末端最低压单体 和 静置压降最大单体 要一致，才更像真的低压方向异常
    if model_type == "rest" and "candidate_cell_by_drop" in df.columns:
        consistency = df["candidate_cell_by_voltage"] == df["candidate_cell_by_drop"]
        df["direction_consistency_flag"] = consistency.astype(int)
        low_like = low_dev & consistency
    else:
        df["direction_consistency_flag"] = 1
        low_like = low_dev

    df["direction_flag"] = np.where(
        low_like,
        "low_voltage_like_micro_short",
        "high_voltage_or_non_micro_short"
    )

    return df


def get_model_columns(model_type):
    """
    返回不同模型分支（rest / cc）对应的特征列。

    rest 和 cc 用的特征不同，所以这里统一管理。
    """
    if model_type == "rest":
        return [
            "rest_drop_res_mean", "rest_drop_res_max",
            "temp_dev_mean", "temp_dev_max",
            "dvdt_res_mean", "dvdt_res_max"
        ]
    if model_type == "cc":
        return [
            "cc_end_dev_mean", "cc_end_dev_max",
            "temp_dev_mean", "temp_dev_max",
            "dvdt_res_mean", "dvdt_res_max"
        ]
    raise ValueError("model_type 只能是 'rest' 或 'cc'。")


def select_healthy_training_mask(feature_df, model_cols):
    """
    选择“更像健康”的窗口作为 IF 训练池。

    设计思想：
    ---------------------------------------------------------
    虽然 IF 是无监督模型，但如果直接拿全部窗口训练，
    可能会被明显异常工况污染。

    所以这里优先选：
        - 特征值较低残差窗口
        - 非低压方向窗口
        - 非 DQ 窗口
    作为更健康的训练池
    """
    X = feature_df[model_cols].copy().fillna(0.0)
    quantile_limit = X.quantile(HEALTHY_TRAIN_QUANTILE)
    mask = (X <= quantile_limit).all(axis=1)

    if "direction_flag" in feature_df.columns:
        mask &= feature_df["direction_flag"] != "low_voltage_like_micro_short"
    if "dq_flag" in feature_df.columns:
        mask &= feature_df["dq_flag"].fillna(0).astype(int) == 0

    # 如果筛得太少，就放宽一点
    if int(mask.sum()) < MIN_HEALTHY_TRAIN_WINDOWS:
        relaxed_limit = X.quantile(0.95)
        mask = (X <= relaxed_limit).all(axis=1)
        if "dq_flag" in feature_df.columns:
            mask &= feature_df["dq_flag"].fillna(0).astype(int) == 0

    # 还不够就全放开，避免完全没有训练池
    if int(mask.sum()) < MIN_HEALTHY_TRAIN_WINDOWS:
        mask = pd.Series(True, index=feature_df.index)

    return mask


def merge_triggered_windows_to_events(combined_df):
    """
    把连续/接近的触发窗口合并成“事件级报警”。

    为什么要做这个？
    ---------------------------------------------------------
    一个实际异常段在滑动窗口里可能会产生很多相邻报警窗口；
    如果不合并，看起来会像“几十个报警”，其实可能只是同一件事。

    返回
    ---------------------------------------------------------
    event_df :
        事件级结果表
    """
    if combined_df is None or len(combined_df) == 0 or "trigger_alarm" not in combined_df.columns:
        return pd.DataFrame()

    triggered = combined_df[combined_df["trigger_alarm"] == 1].copy()
    if len(triggered) == 0:
        return pd.DataFrame()

    triggered["candidate_cell"] = triggered.get("candidate_cell_by_voltage", "unknown")
    triggered = triggered.sort_values(["candidate_cell", "time_start", "time_end"]).reset_index(drop=True)

    merge_gap = pd.Timedelta(minutes=EVENT_MERGE_GAP_MINUTES)
    events = []
    current = None

    for _, row in triggered.iterrows():
        row_start = pd.to_datetime(row["time_start"])
        row_end = pd.to_datetime(row["time_end"])
        candidate_cell = row["candidate_cell"]

        if (
            current is None
            or candidate_cell != current["candidate_cell"]
            or row_start > current["time_end"] + merge_gap
        ):
            if current is not None:
                events.append(current)
            current = {
                "candidate_cell": candidate_cell,
                "time_start": row_start,
                "time_end": row_end,
                "window_count": 1,
                "min_if_score": float(row["if_score"]),
                "model_branches": {row.get("model_branch", row.get("window_type", "unknown"))},
                "direction_flags": {row.get("direction_flag", "unknown")}
            }
        else:
            current["time_end"] = max(current["time_end"], row_end)
            current["window_count"] += 1
            current["min_if_score"] = min(current["min_if_score"], float(row["if_score"]))
            current["model_branches"].add(row.get("model_branch", row.get("window_type", "unknown")))
            current["direction_flags"].add(row.get("direction_flag", "unknown"))

    if current is not None:
        events.append(current)

    event_df = pd.DataFrame(events)
    if len(event_df) == 0:
        return event_df

    event_df["duration_minutes"] = (
        event_df["time_end"] - event_df["time_start"]
    ).dt.total_seconds() / 60.0
    event_df["model_branches"] = event_df["model_branches"].apply(lambda x: ",".join(sorted(x)))
    event_df["direction_flags"] = event_df["direction_flags"].apply(lambda x: ",".join(sorted(x)))
    event_df = event_df.sort_values(["min_if_score", "window_count"], ascending=[True, False]).reset_index(drop=True)
    event_df.insert(0, "event_id", np.arange(1, len(event_df) + 1))
    return event_df


# =========================================================
# 11. 切 rest / cc 窗口、做DQ、提特征
# =========================================================
def build_rest_and_cc_feature_tables(resampled_segments, info, adaptive_thresholds,
                                     segment_id_offset=0,
                                     min_time_end_exclusive=None):
    """
    在重采样后的连续片段上：
        1. 找静置窗口(rest)
        2. 找近似 CC 充电窗口(cc)
        3. 做 DQ 检查
        4. DQ 通过后提特征

    参数
    ---------------------------------------------------------
    resampled_segments :
        当前 chunk 切出来并重采样后的连续片段列表

    segment_id_offset :
        让 segment_id 在所有 chunk 间保持全局唯一

    min_time_end_exclusive :
        上一轮已输出窗口的最大结束时间。
        用来避免跨 chunk 重复输出同一个窗口。
    """
    rest_features = []
    cc_features = []
    dq_records = []

    sample_seconds = get_resample_seconds(RESAMPLE_RULE)

    rest_window_points = int((REST_WINDOW_MINUTES * 60) / sample_seconds)
    rest_step_points = int((REST_STEP_MINUTES * 60) / sample_seconds)

    cc_window_points = int((CC_WINDOW_MINUTES * 60) / sample_seconds)
    cc_step_points = int((CC_STEP_MINUTES * 60) / sample_seconds)

    for seg_local_idx, seg in enumerate(resampled_segments, start=1):
        # segment_id 做全局唯一化
        seg_global_id = segment_id_offset + seg_local_idx

        seg = seg.copy().reset_index(drop=True)

        # ---------- 静置窗口 ----------
        if len(seg) >= rest_window_points:
            for start in range(0, len(seg) - rest_window_points + 1, rest_step_points):
                end = start + rest_window_points
                win = seg.iloc[start:end].copy()

                # 分块拼接时，跳过已经输出过的旧窗口
                if should_skip_emitted_window(win, info, min_time_end_exclusive):
                    continue

                # 先做工况判别：是不是静置窗口
                if is_rest_window_adaptive(win, info, adaptive_thresholds):
                    dq_flag, dq_reason = run_full_dq_check(win, info)
                    common_meta = add_common_window_meta(
                        win, info, "rest", seg_global_id, start, end, dq_flag, dq_reason
                    )

                    # DQ 不通过，记录到 dq_records
                    if dq_flag == 1:
                        dq_records.append(common_meta)
                        continue

                    # 再做工况纯度约束
                    if not passes_rest_trend_constraints(win, info):
                        continue

                    # 最后提特征
                    feat = extract_rest_features(win, info)
                    feat.update(common_meta)
                    rest_features.append(feat)

        # ---------- CC窗口 ----------
        if len(seg) >= cc_window_points:
            for start in range(0, len(seg) - cc_window_points + 1, cc_step_points):
                end = start + cc_window_points
                win = seg.iloc[start:end].copy()

                if should_skip_emitted_window(win, info, min_time_end_exclusive):
                    continue

                # 先做工况判别：是不是近似 CC 窗口
                if is_cc_window_adaptive(win, info, adaptive_thresholds,
                                         charge_sign=CHARGE_SIGN):
                    dq_flag, dq_reason = run_full_dq_check(win, info)
                    common_meta = add_common_window_meta(
                        win, info, "cc", seg_global_id, start, end, dq_flag, dq_reason
                    )

                    if dq_flag == 1:
                        dq_records.append(common_meta)
                        continue

                    if not passes_cc_trend_constraints(win, info):
                        continue

                    feat = extract_cc_features(win, info)
                    feat.update(common_meta)
                    cc_features.append(feat)

    rest_feature_df = pd.DataFrame(rest_features)
    cc_feature_df = pd.DataFrame(cc_features)
    dq_df = pd.DataFrame(dq_records)

    return rest_feature_df, cc_feature_df, dq_df


def debug_time_gaps(df, info, chunk_idx=None):
    """
    调试函数：查看当前 chunk 内时间差分布。

    什么时候有用？
    ---------------------------------------------------------
    当你怀疑：
        - 采样周期在变
        - 分段不合理
        - 连续片段被切碎
    时，这个函数非常有用。
    """
    time_col = info["time_col"]
    dt = df[time_col].diff().dt.total_seconds()

    print("\n================ 时间差调试信息 ================")
    if chunk_idx is not None:
        print(f"chunk_idx: {chunk_idx}")

    print("时间列前5个值:")
    print(df[time_col].head())

    print("\n时间差描述统计（秒）:")
    print(dt.describe())

    print("\n时间差 value_counts 前10项:")
    print(dt.value_counts(dropna=False).head(10))

    print("\n大于30秒的时间差数量:", (dt > TIME_GAP_THRESHOLD_SECONDS).sum())
    print("大于60秒的时间差数量:", (dt > 60).sum())
    print("大于300秒的时间差数量:", (dt > 300).sum())


def estimate_adaptive_gap_threshold(df, info):
    """
    根据当前数据块（chunk）的时间差分布，自适应估计“连续片段切分阈值”。

    设计目的
    ---------------------------------------------------------
    同一组数据中，可能前半段是 5 秒采样，后半段变成 61 秒采样。
    如果还死用 30 秒做断点阈值，
    那 61 秒采样会被误切成大量短片段。

    估计逻辑
    ---------------------------------------------------------
    1. 计算相邻时间差 dt
    2. 去掉无效值（NaN、<=0）
    3. 取出现次数最多的时间差 dominant_dt
    4. 实际断点阈值 =
       max(TIME_GAP_THRESHOLD_SECONDS, TIME_GAP_MULTIPLIER * dominant_dt)
    """
    time_col = info["time_col"]

    dt = df[time_col].diff().dt.total_seconds()
    dt = dt.dropna()
    dt = dt[dt > 0]

    if len(dt) == 0:
        return float(TIME_GAP_THRESHOLD_SECONDS)

    dominant_dt = float(dt.value_counts().idxmax())

    gap_threshold_seconds = max(
        float(TIME_GAP_THRESHOLD_SECONDS),
        float(TIME_GAP_MULTIPLIER) * dominant_dt
    )

    return gap_threshold_seconds


def analyze_system_pattern(df, info):
    """
    基于样本数据判断当前系统更像普通型(sys1)还是高噪声混采样型(sys6)。

    这个函数的作用
    ---------------------------------------------------------
    在真正开始全量分块处理之前，先用一小部分样本数据粗略判断：
        当前这组系统，更适合用“较敏感模板(sys1)”，
        还是更适合用“较保守模板(sys6)”。

    为什么需要这个判断？
    ---------------------------------------------------------
    因为不同系统差异很大：
        - 有的系统比较平稳，适合更敏感的阈值
        - 有的系统混采样明显、噪声更大，适合更保守的阈值
    如果全都硬套同一套模板，容易出现：
        - 有的组满屏红点
        - 有的组完全没信号
        - 有的组阈值严重失真

    当前这一版判别逻辑（回滚版）
    ---------------------------------------------------------
    主要看三件事：

    1. current_std
       - 电流标准差是否很大
       - 如果很大，通常意味着系统工况更激烈、更嘈杂

    2. has_mixed_sampling
       - 是否同时存在明显的短采样（例如 5s）和长采样（例如 60s）
       - 如果是，说明系统存在混采样特征

    3. ratio_dt_gt_30s
       - 大于 30 秒的时间差占比是否很高
       - 如果很高，也说明采样节奏偏复杂，不太适合敏感模板

    为什么这次要回滚？
    ---------------------------------------------------------
    之前我加过一个 near_zero_ratio（近零电流占比）优先判 sys1 的逻辑，
    本意是想把“低电流静置主导组”判回更温和的 sys1。

    但实际跑下来发现：
        sys25 虽然 near_zero_ratio 极高，
        但切到 sys1 后反而出现大量 REST 报警（747 个），
        比原来更差。

    这说明：
        “电流接近静置” ≠ “一定适合 sys1”
    在你当前这套 IF + EWMA 体系里，
    sys25 这种组虽然很静，但仍然更适合保守模板 sys6。

    所以这里回滚到更简单、更稳的判别逻辑：
        只要满足：
            - 电流波动大
            或
            - 存在明显混采样
            或
            - 大时间差占比较高
        就判成 sys6

    返回
    ---------------------------------------------------------
    pattern_info : dict
        包含：
            current_std
            ratio_dt_le_10s
            ratio_dt_ge_50s
            ratio_dt_gt_30s
            has_mixed_sampling
            recommended_mode
    """
    time_col = info["time_col"]
    i_col = info["i_battery_col"]

    pattern_info = {
        "current_std": np.nan,
        "ratio_dt_le_10s": 0.0,
        "ratio_dt_ge_50s": 0.0,
        "ratio_dt_gt_30s": 0.0,
        "has_mixed_sampling": False,
        "recommended_mode": "sys1"
    }

    # -----------------------------------------------------
    # 1. 电流波动：看当前系统电流标准差是否较大
    # -----------------------------------------------------
    if i_col is not None and i_col in df.columns:
        current_std = float(pd.to_numeric(df[i_col], errors="coerce").std())
        pattern_info["current_std"] = current_std
    else:
        current_std = 0.0

    # -----------------------------------------------------
    # 2. 时间差分布：看当前系统是否存在明显混采样
    # -----------------------------------------------------
    dt = df[time_col].diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]

    if len(dt) > 0:
        ratio_dt_le_10s = float((dt <= 10).mean())
        ratio_dt_ge_50s = float((dt >= 50).mean())
        ratio_dt_gt_30s = float((dt > 30).mean())

        pattern_info["ratio_dt_le_10s"] = ratio_dt_le_10s
        pattern_info["ratio_dt_ge_50s"] = ratio_dt_ge_50s
        pattern_info["ratio_dt_gt_30s"] = ratio_dt_gt_30s

        # 典型混采样：短采样和长采样都占比较明显
        has_mixed_sampling = (ratio_dt_le_10s > 0.15) and (ratio_dt_ge_50s > 0.15)
        pattern_info["has_mixed_sampling"] = bool(has_mixed_sampling)
    else:
        has_mixed_sampling = False
        ratio_dt_gt_30s = 0.0

    # -----------------------------------------------------
    # 3. 判别逻辑
    # -----------------------------------------------------
    # 只要满足下面任意一条，就按更保守的 sys6 处理：
    #   - 电流波动很大
    #   - 存在明显混采样
    #   - 大时间差占比高
    #
    # 否则才走更敏感的 sys1。
    if (
        current_std >= 10.0
        or has_mixed_sampling
        or ratio_dt_gt_30s >= 0.20
    ):
        pattern_info["recommended_mode"] = "sys6"
    else:
        pattern_info["recommended_mode"] = "sys1"

    return pattern_info


def apply_parameter_template(mode_name):
    """
    根据模式名，把对应参数模板写回全局变量。

    会被改写的全局参数
    ---------------------------------------------------------
    THRESHOLD_K
    CONSECUTIVE_TRIGGER_N
    EWMA_SIGMA_FLOOR
    DIRECTION_LOW_DEV_THRESHOLD
    """
    global THRESHOLD_K
    global CONSECUTIVE_TRIGGER_N
    global EWMA_SIGMA_FLOOR
    global DIRECTION_LOW_DEV_THRESHOLD

    if mode_name == "sys6":
        template = PARAM_TEMPLATE_SYS6
    else:
        template = PARAM_TEMPLATE_SYS1

    THRESHOLD_K = template["THRESHOLD_K"]
    CONSECUTIVE_TRIGGER_N = template["CONSECUTIVE_TRIGGER_N"]
    EWMA_SIGMA_FLOOR = template["EWMA_SIGMA_FLOOR"]
    DIRECTION_LOW_DEV_THRESHOLD = template["DIRECTION_LOW_DEV_THRESHOLD"]

    print("\n================ 已应用参数模板 ================")
    print(f"mode_name: {mode_name}")
    print(f"THRESHOLD_K = {THRESHOLD_K}")
    print(f"CONSECUTIVE_TRIGGER_N = {CONSECUTIVE_TRIGGER_N}")
    print(f"EWMA_SIGMA_FLOOR = {EWMA_SIGMA_FLOOR}")
    print(f"DIRECTION_LOW_DEV_THRESHOLD = {DIRECTION_LOW_DEV_THRESHOLD}")


def auto_select_and_apply_system_mode(df, info):
    """
    自动判别系统模式，并应用对应参数模板。
    如果 FORCE_SYSTEM_MODE 不为 None，则优先使用手工指定模式。
    """
    pattern_info = analyze_system_pattern(df, info)

    print("\n================ 系统模式判别信息 ================")
    print(f"current_std       : {pattern_info['current_std']:.6f}")
    print(f"ratio_dt_le_10s   : {pattern_info['ratio_dt_le_10s']:.4f}")
    print(f"ratio_dt_ge_50s   : {pattern_info['ratio_dt_ge_50s']:.4f}")
    print(f"ratio_dt_gt_30s   : {pattern_info['ratio_dt_gt_30s']:.4f}")
    print(f"has_mixed_sampling: {pattern_info['has_mixed_sampling']}")
    print(f"auto_mode         : {pattern_info['recommended_mode']}")

    if FORCE_SYSTEM_MODE is not None:
        pattern_info["recommended_mode"] = FORCE_SYSTEM_MODE
        print(f"force_mode        : {FORCE_SYSTEM_MODE}")

    apply_parameter_template(pattern_info["recommended_mode"])

    return pattern_info


# =========================================================
# 11.1 分块版主流程：边读 CSV 边提取特征
# =========================================================
def build_feature_tables_from_large_csv(file_path, info, adaptive_thresholds,
                                        chunksize=300000):
    """
    对超大 CSV 做分块处理，边读边提特征，避免一次性读入内存。

    核心机制
    ---------------------------------------------------------
    1. carry_df：
       保存上一块尾部，用来和下一块拼接，避免 chunk 边界把连续序列切断

    2. segment_id_offset：
       让每个 chunk 里的连续片段编号在全局范围内唯一

    3. last_emitted_time_end：
       避免跨 chunk 重复输出窗口

    返回
    ---------------------------------------------------------
    rest_feature_df :
        全部 REST 窗口特征表

    cc_feature_df :
        全部 CC 窗口特征表

    dq_df :
        被 DQ 筛掉的窗口记录表

    all_resampled_segments :
        若 ENABLE_RAW_SEGMENT_CACHE=True，则缓存所有重采样片段，
        方便后面画原始曲线回查
    """
    all_rest = []
    all_cc = []
    all_dq = []

    # 用 dict 存 segment_id -> DataFrame，方便后面按 segment_id 回查
    all_resampled_segments = {}

    carry_df = None
    time_col = info["time_col"]
    carry_seconds = get_chunk_carry_seconds()
    last_emitted_time_end = None

    global_seg_offset = 0

    for chunk_idx, raw_chunk in iterate_csv_chunks(file_path, chunksize=chunksize):
        # 1. 单 chunk 轻量预处理
        chunk = basic_preprocess_chunk(raw_chunk, info)
        if chunk_idx <= 3:
            debug_time_gaps(chunk, info, chunk_idx=chunk_idx)
        if len(chunk) == 0:
            continue

        current_chunk_max_time = chunk[time_col].max()

        # 2. 把上一 chunk 的尾巴拼进来，保证跨块连续性
        if carry_df is not None and len(carry_df) > 0:
            chunk = pd.concat([carry_df, chunk], axis=0, ignore_index=True)
            if not chunk[time_col].is_monotonic_increasing:
                chunk = chunk.sort_values(time_col, kind="mergesort").reset_index(drop=True)

        # 3. 根据当前 chunk 的主采样间隔，自适应估计时间断点阈值
        adaptive_gap_threshold = estimate_adaptive_gap_threshold(chunk, info)
        print(f"[CHUNK] 自适应时间断点阈值: {adaptive_gap_threshold:.1f} 秒")

        # 4. 切连续片段
        segments = split_continuous_segments(
            chunk,
            info,
            gap_threshold_seconds=adaptive_gap_threshold
        )

        if len(segments) == 0:
            carry_start_time = current_chunk_max_time - pd.Timedelta(seconds=carry_seconds)
            carry_df = chunk[chunk[time_col] >= carry_start_time].copy()
            continue

        # 5. 重采样
        resampled_segments = resample_all_segments(
            segments,
            info,
            rule=RESAMPLE_RULE
        )

        # 6. 更新下一 chunk 的 carry
        carry_start_time = current_chunk_max_time - pd.Timedelta(seconds=carry_seconds)
        carry_df = chunk[chunk[time_col] >= carry_start_time].copy()

        if len(resampled_segments) == 0:
            last_emitted_time_end = current_chunk_max_time
            continue

        # 7. 在这些片段上切窗口、做 DQ、提特征
        rest_df, cc_df, dq_df = build_rest_and_cc_feature_tables(
            resampled_segments,
            info,
            adaptive_thresholds,
            segment_id_offset=global_seg_offset,
            min_time_end_exclusive=last_emitted_time_end
        )

        # 8. 累积结果
        if len(rest_df) > 0:
            all_rest.append(rest_df)
        if len(cc_df) > 0:
            all_cc.append(cc_df)
        if len(dq_df) > 0:
            all_dq.append(dq_df)

        # 9. 如果允许缓存，就把重采样片段也存起来，后面可画图回查
        if ENABLE_RAW_SEGMENT_CACHE:
            for local_idx, seg in enumerate(resampled_segments, start=1):
                global_seg_id = global_seg_offset + local_idx
                all_resampled_segments[global_seg_id] = seg

        # 10. 更新 segment 偏移
        global_seg_offset += len(resampled_segments)
        last_emitted_time_end = current_chunk_max_time

        print(f"[CHUNK] 第 {chunk_idx} 个chunk处理完成。"
              f"  rest新增={len(rest_df)}, cc新增={len(cc_df)}, dq新增={len(dq_df)}")

    # ---- 汇总 ----
    rest_feature_df = pd.concat(all_rest, axis=0, ignore_index=True) if len(all_rest) > 0 else pd.DataFrame()
    cc_feature_df = pd.concat(all_cc, axis=0, ignore_index=True) if len(all_cc) > 0 else pd.DataFrame()
    dq_df = pd.concat(all_dq, axis=0, ignore_index=True) if len(all_dq) > 0 else pd.DataFrame()

    print("\n================ 分块处理汇总 ================")
    print(f"rest_feature_df: {rest_feature_df.shape}")
    print(f"cc_feature_df  : {cc_feature_df.shape}")
    print(f"dq_df          : {dq_df.shape}")
    print(f"累计重采样片段数 : {len(all_resampled_segments)}")

    return rest_feature_df, cc_feature_df, dq_df, all_resampled_segments


# =========================================================
# 12. 给特征表训练 IF
# =========================================================
def fit_iforest_and_score(feature_df, model_type):
    """
    对特征表训练 Isolation Forest，并输出每个窗口的异常分数。

    步骤
    ---------------------------------------------------------
    1. 取当前模型分支对应的特征列
    2. 构建“更像健康”的训练池
    3. 标准化
    4. 训练 Isolation Forest
    5. 对所有窗口打分
    """
    model_cols = get_model_columns(model_type)

    X = feature_df[model_cols].copy().fillna(0.0)
    healthy_mask = select_healthy_training_mask(feature_df, model_cols)
    X_train = X.loc[healthy_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)

    clf = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=0.01,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    clf.fit(X_train_scaled)
    raw_score = clf.score_samples(X_scaled)

    result_df = feature_df.copy()
    result_df["if_score"] = raw_score
    result_df["used_for_if_train"] = healthy_mask.astype(int).values
    return result_df, clf, scaler


# =========================================================
# 13. EWMA 动态阈值
# =========================================================
def apply_ewma_dynamic_threshold(result_df):
    """
    对 IF 连续分数应用 EWMA 动态阈值。

    核心思想
    ---------------------------------------------------------
    Isolation Forest 会给每个窗口一个 if_score，
    但不同时间段、不同工况下，分数基线可能会慢慢变化。

    所以这里不用固定阈值，而是：
        - 动态维护一个健康基线均值 mu
        - 动态维护一个健康波动 sigma
        - 阈值 = mu - THRESHOLD_K * sigma

    同时做了 4 个保护：
    ---------------------------------------------------------
    1. 优先用高置信健康窗口初始化基线
    2. sigma 设下限，防止阈值过度贴近均值
    3. 只有“没跌破阈值且距离基线不太远”的点才允许更新
    4. 限制 mu 单步上升速度，防止阈值突然抬高

    返回
    ---------------------------------------------------------
    df :
        增加了以下列：
            ewma_mu
            ewma_sigma
            dynamic_threshold
            below_threshold
            trigger_alarm
    """
    df = result_df.copy().reset_index(drop=True)

    if len(df) < INIT_HEALTHY_WINDOWS + 1:
        raise ValueError("窗口数量太少，不足以初始化健康基线。")

    # -----------------------------------------------------
    # 1. 初始化健康基线
    # -----------------------------------------------------
    # 优先使用被选入 IF 训练池的健康窗口做初始基线
    if "used_for_if_train" in df.columns and int(df["used_for_if_train"].sum()) >= INIT_HEALTHY_WINDOWS:
        init_scores = df.loc[df["used_for_if_train"] == 1, "if_score"].head(INIT_HEALTHY_WINDOWS)
    else:
        init_scores = df.loc[:INIT_HEALTHY_WINDOWS - 1, "if_score"]

    mu = float(init_scores.mean())
    sigma = float(init_scores.std(ddof=0))

    # sigma 下限保护
    sigma = max(sigma, EWMA_SIGMA_FLOOR)

    threshold_list = []
    mu_list = []
    sigma_list = []
    below_threshold_list = []
    trigger_list = []

    consecutive_count = 0

    # -----------------------------------------------------
    # 2. 逐窗口应用动态阈值
    # -----------------------------------------------------
    for idx in range(len(df)):
        score = float(df.loc[idx, "if_score"])

        # 当前动态阈值
        threshold = mu - THRESHOLD_K * sigma

        # 当前点是否低于阈值
        below = score < threshold

        if below:
            consecutive_count += 1
        else:
            consecutive_count = 0

        # 当前点离基线的“z分数距离”
        z = abs(score - mu) / max(sigma, 1e-8)

        # 只有“没跌破阈值 + 离基线不远”的点才允许更新 EWMA
        if (not below) and (z <= EWMA_UPDATE_Z_MAX):
            old_mu = mu

            new_mu = EWMA_ALPHA * score + (1 - EWMA_ALPHA) * mu
            new_sigma = EWMA_ALPHA * abs(score - mu) + (1 - EWMA_ALPHA) * sigma

            # 限制 mu 单步上升幅度
            mu = min(float(new_mu), float(old_mu) + EWMA_MU_MAX_STEP_UP)

            # sigma 保持下限
            sigma = max(float(new_sigma), EWMA_SIGMA_FLOOR)

        # 连续 N 个窗口跌破阈值才触发正式报警
        trigger = consecutive_count >= CONSECUTIVE_TRIGGER_N

        threshold_list.append(threshold)
        mu_list.append(mu)
        sigma_list.append(sigma)
        below_threshold_list.append(int(below))
        trigger_list.append(int(trigger))

    df["ewma_mu"] = mu_list
    df["ewma_sigma"] = sigma_list
    df["dynamic_threshold"] = threshold_list
    df["below_threshold"] = below_threshold_list
    df["trigger_alarm"] = trigger_list

    return df


# =========================================================
# 14. 绘制分数图
# =========================================================
def plot_scores_and_threshold(result_df, model_name="rest"):
    """
    绘制：
        - IF 分数曲线
        - EWMA 动态阈值曲线
        - 触发报警点

    图上解释
    ---------------------------------------------------------
    蓝线：
        IF Score，越低越异常

    橙线：
        动态阈值

    红点：
        触发报警的窗口
    """
    plt.figure(figsize=(15, 6))
    plt.plot(result_df["if_score"].values,
             label="IF Score (lower = more abnormal)", linewidth=1.2)
    plt.plot(result_df["dynamic_threshold"].values,
             label="Dynamic Threshold", linestyle="--")
    alarm_idx = result_df.index[result_df["trigger_alarm"] == 1]
    plt.scatter(alarm_idx, result_df.loc[alarm_idx, "if_score"],
                color="red", s=20, label="Triggered Alarm")
    plt.title(f"{model_name.upper()} Isolation Forest Score with EWMA Threshold")
    plt.xlabel("Window Index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# 15. 单模型完整流程
# =========================================================
def run_single_model_pipeline(feature_df, model_type, output_feature_path, output_score_path):
    """
    对一个分支（rest 或 cc）执行完整流程：

    流程
    ---------------------------------------------------------
    1. 保存窗口特征表
    2. 训练 IF 并打分
    3. 若窗口数足够，则继续做 EWMA 动态阈值
    4. 保存最终 score 表
    5. 打印最异常窗口
    6. 画图
    """
    if len(feature_df) == 0:
        print(f"\n{model_type} 模型没有可用窗口，跳过。")
        return None, None, None

    feature_df.to_csv(output_feature_path, index=False, encoding="utf-8-sig")
    print(f"\n{model_type} 窗口特征已保存到: {output_feature_path}")

    scored_df, clf, scaler = fit_iforest_and_score(feature_df, model_type=model_type)

    if len(scored_df) < INIT_HEALTHY_WINDOWS + 1:
        # 若窗口数不足，不做 EWMA，只保存 IF 分数
        print(f"{model_type} 窗口数量不足，无法做EWMA动态阈值，先只保存IF分数。")
        final_df = scored_df.copy()
        final_df["ewma_mu"] = np.nan
        final_df["ewma_sigma"] = np.nan
        final_df["dynamic_threshold"] = np.nan
        final_df["below_threshold"] = np.nan
        final_df["trigger_alarm"] = 0
    else:
        final_df = apply_ewma_dynamic_threshold(scored_df)

    final_df.to_csv(output_score_path, index=False, encoding="utf-8-sig")
    print(f"{model_type} 连续打分结果已保存到: {output_score_path}")

    print(f"\n{model_type} 最异常的前10个窗口（if_score 越低越异常）：")
    show_cols = ["window_type", "time_start", "time_end",
                 "if_score", "dynamic_threshold", "trigger_alarm"]
    print(final_df.sort_values("if_score", ascending=True).head(10)[show_cols])
    print(f"\n{model_type} 触发报警窗口数:", final_df["trigger_alarm"].sum())

    plot_scores_and_threshold(final_df, model_name=model_type)
    return final_df, clf, scaler


# =========================================================
# 16. 合并 rest / cc 结果
# =========================================================
def combine_rest_cc_results(rest_result_df, cc_result_df):
    """
    把 REST 和 CC 两个分支的结果合并成一张总表。

    额外增加：
        model_branch = "rest" 或 "cc"
    """
    parts = []
    if rest_result_df is not None and len(rest_result_df) > 0:
        temp = rest_result_df.copy()
        temp["model_branch"] = "rest"
        parts.append(temp)
    if cc_result_df is not None and len(cc_result_df) > 0:
        temp = cc_result_df.copy()
        temp["model_branch"] = "cc"
        parts.append(temp)
    if len(parts) == 0:
        return pd.DataFrame()
    combined_df = pd.concat(parts, axis=0, ignore_index=True)
    combined_df = combined_df.sort_values(["time_start", "time_end"]).reset_index(drop=True)
    return combined_df


# =========================================================
# 17. 触发报警窗口概览
# =========================================================
def get_triggered_windows(result_df, model_type="rest", top_k=20):
    """
    提取 trigger_alarm == 1 的窗口，并打印前 top_k 个最异常报警窗口。
    """
    if result_df is None or len(result_df) == 0:
        print(f"\n{model_type} 模型没有结果，无法提取报警窗口。")
        return pd.DataFrame()

    triggered_df = result_df[result_df["trigger_alarm"] == 1].copy()
    triggered_df = triggered_df.sort_values("if_score", ascending=True).reset_index(drop=True)

    print(f"\n================ {model_type.upper()} 模型触发报警窗口概览 ================")
    print(f"触发报警窗口总数: {len(triggered_df)}")
    if len(triggered_df) == 0:
        return triggered_df

    print(f"\n最异常的前 {min(top_k, len(triggered_df))} 个触发报警窗口：")
    print(triggered_df.head(top_k)[
        ["window_type", "segment_id", "time_start", "time_end",
         "if_score", "dynamic_threshold", "trigger_alarm"]
    ])
    return triggered_df


# =========================================================
# 18. 回查触发报警窗口的原始曲线
# =========================================================
def plot_triggered_windows_raw(resampled_segments, result_df, info,
                               model_type="rest", top_k=5):
    """
    把触发报警的窗口对应的原始/重采样曲线画出来。

    作用：
        用于人工复核
        看看报警窗口对应的单体电压、温度、电流长什么样
    """
    if result_df is None or len(result_df) == 0:
        print(f"\n{model_type} 模型没有结果，无法回查报警窗口。")
        return

    triggered_df = result_df[result_df["trigger_alarm"] == 1].copy()
    if len(triggered_df) == 0:
        print(f"\n{model_type} 模型没有触发报警窗口。")
        return

    triggered_df = triggered_df.sort_values("if_score", ascending=True).head(top_k)
    print(f"\n开始绘制 {model_type.upper()} 模型前 {len(triggered_df)} 个触发报警窗口...")

    for rank, (_, row) in enumerate(triggered_df.iterrows(), start=1):
        segment_id = int(row["segment_id"])
        start = int(row["window_start_in_segment"])
        end = int(row["window_end_in_segment"])
        score = row["if_score"]
        threshold = row["dynamic_threshold"]
        window_type = row["window_type"]

        if segment_id not in resampled_segments:
            print(f"segment_id={segment_id} 不在 resampled_segments 中，跳过。")
            continue

        seg = resampled_segments[segment_id]
        win = seg.iloc[start:end + 1].copy()
        x = win[info["time_col"]]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

        for col in info["cell_u_cols"]:
            axes[0].plot(x, win[col], linewidth=1, label=col)
        axes[0].set_title(
            f"Triggered Alarm #{rank} | Type={window_type} | "
            f"Score={score:.4f} | Threshold={threshold:.4f} | "
            f"Segment={segment_id} | Rows={start}-{end}"
        )
        axes[0].set_ylabel("Cell Voltage (V)")
        axes[0].legend(loc="best", ncol=4, fontsize=8)
        axes[0].grid(True, alpha=0.3)

        if len(info["temp_cols"]) > 0:
            for col in info["temp_cols"]:
                axes[1].plot(x, win[col], linewidth=1, label=col)
            axes[1].set_ylabel("Temperature (°C)")
            axes[1].legend(loc="best", ncol=4, fontsize=8)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No temperature columns found",
                         transform=axes[1].transAxes, ha="center", va="center")

        if info["i_battery_col"] is not None:
            axes[2].plot(x, win[info["i_battery_col"]],
                         color="black", linewidth=1.2, label=info["i_battery_col"])
            axes[2].legend(loc="best")
            axes[2].grid(True, alpha=0.3)

        axes[2].set_ylabel("Battery Current (A)")
        axes[2].set_xlabel("Time")
        plt.tight_layout()
        plt.show()


# =========================================================
# 19. 回查最低分窗口的原始曲线
# =========================================================
def plot_lowest_score_windows_raw(resampled_segments, result_df, info,
                                  model_type="rest", top_k=5):
    """
    把 if_score 最低（最异常）的窗口对应曲线画出来。

    和 plot_triggered_windows_raw 的区别：
        这里不要求 trigger_alarm == 1
        只是看“分数最低”的那些窗口长什么样
    """
    if result_df is None or len(result_df) == 0:
        print(f"\n{model_type} 模型没有结果，无法回查最低分窗口。")
        return

    low_df = result_df.sort_values("if_score", ascending=True).head(top_k)
    print(f"\n开始绘制 {model_type.upper()} 模型前 {len(low_df)} 个最低分窗口...")

    for rank, (_, row) in enumerate(low_df.iterrows(), start=1):
        segment_id = int(row["segment_id"])
        start = int(row["window_start_in_segment"])
        end = int(row["window_end_in_segment"])
        score = row["if_score"]
        threshold = row["dynamic_threshold"]
        alarm_flag = int(row["trigger_alarm"])
        window_type = row["window_type"]

        if segment_id not in resampled_segments:
            print(f"segment_id={segment_id} 不在 resampled_segments 中，跳过。")
            continue

        seg = resampled_segments[segment_id]
        win = seg.iloc[start:end + 1].copy()
        x = win[info["time_col"]]

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        for col in info["cell_u_cols"]:
            axes[0].plot(x, win[col], linewidth=1, label=col)
        axes[0].set_title(
            f"Lowest Score #{rank} | Type={window_type} | "
            f"Score={score:.4f} | Threshold={threshold:.4f} | "
            f"Alarm={alarm_flag} | Segment={segment_id} | Rows={start}-{end}"
        )
        axes[0].set_ylabel("Cell Voltage (V)")
        axes[0].legend(loc="best", ncol=4, fontsize=8)
        axes[0].grid(True, alpha=0.3)

        if len(info["temp_cols"]) > 0:
            for col in info["temp_cols"]:
                axes[1].plot(x, win[col], linewidth=1, label=col)
            axes[1].set_ylabel("Temperature (°C)")
            axes[1].legend(loc="best", ncol=4, fontsize=8)
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "No temperature columns found",
                         transform=axes[1].transAxes, ha="center", va="center")

        if info["i_battery_col"] is not None:
            axes[2].plot(x, win[info["i_battery_col"]],
                         color="black", linewidth=1.2, label=info["i_battery_col"])
            axes[2].legend(loc="best")
            axes[2].grid(True, alpha=0.3)

        axes[2].set_ylabel("Battery Current (A)")
        axes[2].set_xlabel("Time")
        plt.tight_layout()
        plt.show()


# =========================================================
# 20. 主函数（分块版本）
# =========================================================
def main():
    """
    分块版主流程。

    整体流程总览
    ---------------------------------------------------------
    1. 小样本读取：
       识别字段

    2. 中等样本读取：
       做系统模式判别（sys1 / sys6）
       拟合自适应工况阈值

    3. 全量分块读取 CSV：
       切连续片段 -> 重采样 -> 切 rest/cc 窗口 -> DQ -> 提特征

    4. 分别跑 REST / CC 两个分支：
       IF -> EWMA -> 得到报警结果

    5. 输出与合并：
       - DQ 窗口表
       - rest/cc 特征表
       - rest/cc score 表
       - 合并总表
       - 事件级报警表
    """
    # 确保当前系统输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # --------------------------------------------------
    # 1. 先读少量行，用于识别列
    # --------------------------------------------------
    df_head = load_data(FILE_PATH, nrows=50000)
    info = identify_columns(df_head)
    validate_required_columns(info)
    del df_head

    # --------------------------------------------------
    # 2. 再读一个中等大小样本，用于：
    #    - 系统模式判别
    #    - 自适应工况阈值拟合
    # --------------------------------------------------
    df_sample = load_data(FILE_PATH, nrows=ADAPTIVE_FIT_NROWS)
    df_sample_clean = basic_preprocess_chunk(df_sample, info)
    del df_sample

    # --------------------------------------------------
    # 2.1 自动判别当前系统更像 sys1 还是 sys6，并应用模板参数
    # --------------------------------------------------
    pattern_info = auto_select_and_apply_system_mode(df_sample_clean, info)

    # 调试：电流分布
    i_col = info["i_battery_col"]
    if i_col is not None:
        print("\n================ 电流分布调试信息（基于阈值拟合样本） ================")
        print(df_sample_clean[i_col].describe())
        print("正电流样本数:", (df_sample_clean[i_col] > 0).sum())
        print("负电流样本数:", (df_sample_clean[i_col] < 0).sum())
        print("接近0电流样本数（|I|<=0.5）:", (df_sample_clean[i_col].abs() <= 0.5).sum())

    # 拟合当前系统的自适应工况阈值
    adaptive_thresholds = fit_adaptive_current_thresholds(
        df_sample_clean,
        info,
        charge_sign=CHARGE_SIGN
    )
    print_adaptive_thresholds(adaptive_thresholds)
    del df_sample_clean

    # --------------------------------------------------
    # 3. 全量分块提特征
    # --------------------------------------------------
    (rest_feature_df,
     cc_feature_df,
     dq_df,
     resampled_segments_dict) = build_feature_tables_from_large_csv(
        FILE_PATH,
        info,
        adaptive_thresholds,
        chunksize=CHUNKSIZE
    )

    # 对 REST / CC 特征表分别重新打方向标签（自适应阈值）
    rest_feature_df = apply_adaptive_direction_labels(rest_feature_df, model_type="rest")
    cc_feature_df = apply_adaptive_direction_labels(cc_feature_df, model_type="cc")

    # 保存 DQ 窗口表
    if len(dq_df) > 0:
        dq_df.to_csv(OUTPUT_DQ_PATH, index=False, encoding="utf-8-sig")
        print(f"\nDQ筛掉的窗口已保存到: {OUTPUT_DQ_PATH}")

    # --------------------------------------------------
    # 4. 分别跑 REST / CC 两个模型分支
    # --------------------------------------------------
    rest_result_df, _, _ = run_single_model_pipeline(
        rest_feature_df,
        model_type="rest",
        output_feature_path=OUTPUT_REST_FEATURE_PATH,
        output_score_path=OUTPUT_REST_SCORE_PATH
    )

    cc_result_df, _, _ = run_single_model_pipeline(
        cc_feature_df,
        model_type="cc",
        output_feature_path=OUTPUT_CC_FEATURE_PATH,
        output_score_path=OUTPUT_CC_SCORE_PATH
    )

    # --------------------------------------------------
    # 5. 打印触发报警窗口概览
    # --------------------------------------------------
    get_triggered_windows(rest_result_df, model_type="rest", top_k=20)
    get_triggered_windows(cc_result_df, model_type="cc", top_k=20)

    # --------------------------------------------------
    # 6. 回查报警窗口原始曲线（默认关闭，省内存）
    # --------------------------------------------------
    if ENABLE_RAW_SEGMENT_CACHE:
        plot_triggered_windows_raw(resampled_segments_dict, rest_result_df, info,
                                   model_type="rest", top_k=5)
        plot_triggered_windows_raw(resampled_segments_dict, cc_result_df, info,
                                   model_type="cc", top_k=5)
    else:
        print("\n[RAW] ENABLE_RAW_SEGMENT_CACHE=False，跳过报警窗口原始曲线回查以节省边缘侧内存。")

    # --------------------------------------------------
    # 7. 回查最低分窗口原始曲线（默认关闭，省内存）
    # --------------------------------------------------
    if ENABLE_RAW_SEGMENT_CACHE:
        plot_lowest_score_windows_raw(resampled_segments_dict, rest_result_df, info,
                                      model_type="rest", top_k=5)
        plot_lowest_score_windows_raw(resampled_segments_dict, cc_result_df, info,
                                      model_type="cc", top_k=5)
    else:
        print("[RAW] ENABLE_RAW_SEGMENT_CACHE=False，跳过最低分窗口原始曲线回查。")

    # --------------------------------------------------
    # 8. 合并 REST / CC 结果
    # --------------------------------------------------
    combined_result_df = combine_rest_cc_results(rest_result_df, cc_result_df)

    if len(combined_result_df) > 0:
        combined_result_df.to_csv(OUTPUT_COMBINED_RESULT_PATH, index=False, encoding="utf-8-sig")
        print(f"\n合并后的全局结果表已保存到: {OUTPUT_COMBINED_RESULT_PATH}")

        # 合并成事件级报警
        event_df = merge_triggered_windows_to_events(combined_result_df)
        if len(event_df) > 0:
            event_df.to_csv(OUTPUT_EVENT_PATH, index=False, encoding="utf-8-sig")
            print(f"事件级微短路候选结果已保存到: {OUTPUT_EVENT_PATH}")
            print("\n事件级候选前20条：")
            print(event_df.head(20))
        else:
            print("\n没有可合并的事件级报警。")

        print("\n合并后的前20个最异常窗口：")
        show_cols = ["model_branch", "window_type", "time_start", "time_end",
                     "if_score", "dynamic_threshold", "trigger_alarm", "direction_flag"]
        available_cols = [c for c in show_cols if c in combined_result_df.columns]
        print(
            combined_result_df.sort_values("if_score", ascending=True)
            .head(20)[available_cols]
        )

    print("\n全部流程执行完毕。")


# =========================================================
# 程序入口
# =========================================================
if __name__ == "__main__":
    main()