import numpy as np
import pandas as pd


# =========================================================
# 自适应工况判别模块（增强版）
# =========================================================
# 这个文件专门负责“工况判别”和“基础物理有效性检查”。
#
# 你可以把它理解成：
#   在真正做微短路异常检测（Isolation Forest）之前，
#   先用一些更基础、更物理、更朴素的规则，判断：
#
#   1. 这个窗口是不是“静置窗口（rest）”
#   2. 这个窗口是不是“近似恒流充电窗口（CC）”
#   3. 这个窗口里有没有明显不合理的数据（如极端电压）
#
# 为什么要单独拆成这个模块？
# ---------------------------------------------------------
# 因为不同 data_sys_x.csv 的数据分布差异很大：
#   - 有的组大多数时间接近静置
#   - 有的组充放电频繁
#   - 有的组 5s / 60s 混采样
#   - 有的组电流波动特别大
#
# 如果工况阈值全都写死，很容易出现：
#   - 某些组筛得太严，窗口几乎没有
#   - 某些组筛得太松，很多不纯窗口混进来
#
# 所以这里的思路是：
#   用“当前这组数据自己的电流分布”来反推适合它自己的阈值。
# =========================================================


# ---------------------------------------------------------
# 静置判据硬约束：宁缺毋滥
# ---------------------------------------------------------
# 这两个值是静置判定中的“硬约束”：
#
# REST_STD_I_HARD_MAX
#   - 静置窗口内，电流标准差允许的最大值
#   - 静置不只是“平均电流接近 0”，还应该“整体很稳定”
#
# REST_RANGE_I_HARD_MAX
#   - 静置窗口内，电流极差(max-min)允许的最大值
#   - 防止窗口中间混入明显的动作/脉冲/跳变
#
# 总体思想：
#   对静置窗口宁可保守一点，也不要把明显不稳的窗口误判成静置。
# ---------------------------------------------------------
REST_STD_I_HARD_MAX = 0.05
REST_RANGE_I_HARD_MAX = 0.30


# ---------------------------------------------------------
# 单体电压物理有效范围
# ---------------------------------------------------------
# 如果某个窗口里单体电压已经离谱到超出这个范围，
# 那通常就不是“微短路早期轻微异常”，而更可能是：
#   - 采样错误
#   - 传感器问题
#   - 数据坏点
#   - 硬件故障
#
# 这种窗口不应该继续进入正常 IF 微短路筛查。
# ---------------------------------------------------------
CELL_VOLTAGE_MIN_VALID = 2.5
CELL_VOLTAGE_MAX_VALID = 4.5


def fit_adaptive_current_thresholds(df, info,
                                    charge_sign=1,
                                    rest_abs_quantile=0.80,
                                    cc_pos_quantile=0.90,
                                    min_rest_abs_cap=0.1,
                                    min_cc_mean_floor=5.0):
    """
    基于当前系统的电流分布，自适应生成静置(rest)与 CC(近似恒流充电)判别阈值。

    参数说明
    ---------------------------------------------------------
    df :
        已做基础预处理的数据表。
        至少需要包含 I_Battery 列。

    info :
        identify_columns() 识别出来的字段信息字典。
        这里主要使用：
            info["i_battery_col"]

    charge_sign :
        充电方向定义。
        当前代码约定：
            正电流 = 充电
            负电流 = 放电
        所以一般设为 1。
        如果你的系统刚好相反，才改成 -1。

    rest_abs_quantile :
        估计静置平均绝对电流阈值时，取小电流池中的哪个分位数。
        值越大，静置阈值越宽松。

    cc_pos_quantile :
        估计 CC 平均充电电流下限时，取正向充电电流中的哪个分位数。
        值越大，CC 门槛越高，越保守。

    min_rest_abs_cap :
        静置平均绝对电流阈值的“下限保护值”。
        防止阈值小得离谱。

    min_cc_mean_floor :
        CC 平均电流下限的“下限保护值”。
        防止 CC 门槛低得离谱。

    返回值
    ---------------------------------------------------------
    thresholds : dict
        包含：
            rest_mean_abs_i_max
            rest_max_abs_i_max
            rest_std_i_max
            rest_range_i_max
            cc_mean_i_min
            cc_std_i_max
            cc_range_i_max

    设计思想
    ---------------------------------------------------------
    1. 静置(rest)：
       静置窗口应该满足：
         - 平均绝对电流小
         - 最大绝对电流小
         - 电流标准差小
         - 电流极差小

    2. 近似 CC 充电：
       CC 窗口应该满足：
         - 平均充电电流足够大
         - 电流波动不要太大
         - 电流范围不要太宽

    这里不写死阈值，而是根据当前系统的电流分布自动估。
    """
    i_col = info["i_battery_col"]
    if i_col is None:
        raise ValueError("未识别到 I_Battery 列，无法进行自适应工况判别。")

    i_data = pd.to_numeric(df[i_col], errors="coerce")
    i_data = i_data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(i_data) == 0:
        raise ValueError("I_Battery 全为空，无法进行自适应工况判别。")

    abs_i = i_data.abs()

    # ---------------------------
    # 静置阈值
    # ---------------------------
    # 先从所有电流里取“相对小电流”的那部分数据作为静置候选池，
    # 避免极端大电流直接把静置阈值拉高。
    small_current_pool = abs_i[abs_i <= abs_i.quantile(0.90)]

    if len(small_current_pool) == 0:
        # 如果连小电流池都没有，退回保护值
        rest_mean_abs_i_max = min_rest_abs_cap
    else:
        # 用小电流池里的某个分位数估计“静置平均电流上限”
        rest_mean_abs_i_max = max(
            small_current_pool.quantile(rest_abs_quantile),
            min_rest_abs_cap
        )

    # 静置窗口允许的最大绝对电流，再适当放宽一点
    rest_max_abs_i_max = max(rest_mean_abs_i_max * 2.0, min_rest_abs_cap * 2.0)

    # ---------------------------
    # CC阈值
    # ---------------------------
    # 把方向统一成“充电为正”
    signed_i = charge_sign * i_data

    # 只取正向充电电流
    pos_i = signed_i[signed_i > 0]

    if len(pos_i) == 0:
        # 如果完全没有正向充电样本，就认为当前数据不支持 CC
        cc_mean_i_min = np.inf
        cc_std_i_max = 0.0
        cc_range_i_max = 0.0
    else:
        # 先估一个“足够像 CC 的平均电流下限”
        cc_mean_i_min = max(pos_i.quantile(cc_pos_quantile), min_cc_mean_floor)

        # 再从中取高电流池，用它估波动/范围上限
        high_pos_pool = pos_i[pos_i >= cc_mean_i_min]

        if len(high_pos_pool) >= 5:
            cc_std_i_max = max(high_pos_pool.std(ddof=0) * 0.5, 1.0)
            cc_range_i_max = max((high_pos_pool.max() - high_pos_pool.min()) * 0.5, 3.0)
        else:
            # 样本太少时，退回默认保守值
            cc_std_i_max = 2.0
            cc_range_i_max = 5.0

    thresholds = {
        "rest_mean_abs_i_max": float(rest_mean_abs_i_max),
        "rest_max_abs_i_max": float(rest_max_abs_i_max),
        "rest_std_i_max": float(min(REST_STD_I_HARD_MAX, rest_mean_abs_i_max)),
        "rest_range_i_max": float(REST_RANGE_I_HARD_MAX),
        "cc_mean_i_min": float(cc_mean_i_min),
        "cc_std_i_max": float(cc_std_i_max),
        "cc_range_i_max": float(cc_range_i_max)
    }

    return thresholds


def print_adaptive_thresholds(thresholds):
    """
    打印自适应工况判别阈值，方便调试和记录。

    常关注的字段
    ---------------------------------------------------------
    rest_mean_abs_i_max :
        静置窗口平均绝对电流上限

    rest_max_abs_i_max :
        静置窗口最大绝对电流上限

    cc_mean_i_min :
        CC 窗口平均充电电流下限

    如果某组的 cc_mean_i_min 特别高，
    往往说明：
        - 该组充电样本很少
        - 或者该组只有很强的充电才会被识别为 CC
    """
    print("\n================ 自适应工况判别阈值 ================")
    for k, v in thresholds.items():
        print(f"{k}: {v:.6f}")


def has_extreme_cell_voltage(win, info):
    """
    检查窗口内是否存在超出物理合理范围的单体电压。

    为什么做这个检查？
    ---------------------------------------------------------
    微短路一般是“轻微偏差”，不会让单体电压瞬间离谱到完全不合理。
    如果这里已经超范围，更像：
        - 坏数据
        - 采样故障
        - 通道问题
        - 硬件极端故障

    返回
    ---------------------------------------------------------
    (flag, reason)
        flag=True  : 当前窗口存在极端电压，应该剔除
        flag=False : 通过
    """
    cell_cols = info["cell_u_cols"]
    cell_data = win[cell_cols]

    v_min = np.nanmin(cell_data.values)
    v_max = np.nanmax(cell_data.values)

    if v_min < CELL_VOLTAGE_MIN_VALID:
        return True, f"cell_voltage_too_low: min_v={v_min:.6f}V"

    if v_max > CELL_VOLTAGE_MAX_VALID:
        return True, f"cell_voltage_too_high: max_v={v_max:.6f}V"

    return False, "pass"


def is_rest_window_adaptive(win, info, thresholds):
    """
    使用“自适应阈值 + 稳定性硬约束”判断一个窗口是不是静置窗口。

    判据
    ---------------------------------------------------------
    cond1 :
        平均绝对电流要小

    cond2 :
        最大绝对电流要小

    cond3 :
        电流标准差要小（窗口内部要稳定）

    cond4 :
        电流极差要小（不能有明显跳变）

    只有四个条件都满足，才判为静置窗口。
    """
    i_col = info["i_battery_col"]
    if i_col is None:
        return False

    i_data = win[i_col]
    i_abs = i_data.abs()

    cond1 = i_abs.mean() <= thresholds["rest_mean_abs_i_max"]
    cond2 = i_abs.max() <= thresholds["rest_max_abs_i_max"]
    cond3 = i_data.std() <= thresholds["rest_std_i_max"]
    cond4 = (i_data.max() - i_data.min()) <= thresholds["rest_range_i_max"]

    return bool(cond1 and cond2 and cond3 and cond4)


def is_cc_window_adaptive(win, info, thresholds, charge_sign=1):
    """
    使用自适应阈值判断一个窗口是不是近似 CC 充电窗口。

    判据
    ---------------------------------------------------------
    cond1 :
        平均充电电流要足够大

    cond2 :
        电流标准差不能太大（不能抖得厉害）

    cond3 :
        电流范围不能太宽（不能明显偏离恒流）

    只有三个条件都满足，才判为近似 CC 充电窗口。
    """
    i_col = info["i_battery_col"]
    if i_col is None:
        return False

    i_data = win[i_col]
    signed_i = charge_sign * i_data

    cond1 = signed_i.mean() >= thresholds["cc_mean_i_min"]
    cond2 = i_data.std() <= thresholds["cc_std_i_max"]
    cond3 = (i_data.max() - i_data.min()) <= thresholds["cc_range_i_max"]

    return bool(cond1 and cond2 and cond3)