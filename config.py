"""
城市空中交通流量协同优化仿真 - 全局配置参数
UAM Strategic Conflict Management - Global Configuration
=========================================================
基于4D轨迹的战略冲突管理，简化版（去除任务偏好约束，同质化处理）
"""

# ======================== 空域参数 (Airspace) ========================
AIRSPACE_LENGTH = 10000        # X方向空域范围 (m)
AIRSPACE_WIDTH  = 10000        # Y方向空域范围 (m)
AIRSPACE_HEIGHT = 300          # Z方向空域范围 (m)

GRID_SIZE_XY = 100             # 水平网格分辨率 (m)
GRID_SIZE_Z  = 30              # 垂直网格分辨率 (m)

NX = AIRSPACE_LENGTH // GRID_SIZE_XY   # X方向网格数 = 100
NY = AIRSPACE_WIDTH  // GRID_SIZE_XY   # Y方向网格数 = 100
NZ = AIRSPACE_HEIGHT // GRID_SIZE_Z    # Z方向网格数 = 10

CELL_CAPACITY = 1              # 单网格-单时间窗最大容量

# ======================== 航班参数 (Flight) ========================
NUM_FLIGHTS   = 100            # 航班数量 (架次/小时)
CRUISE_SPEED  = 15.0           # 基础巡航速度 (m/s)
ETD_MIN       = 1              # 最早预计起飞时间 (s)
ETD_MAX       = 3600           # 最晚预计起飞时间 (s)

# ======================== 冲突检测参数 (Conflict Detection) ========================
T_SEP = 60                    # 确定性时间安全间隔阈值 (s)

# ======================== 优化目标参数 (Cost Function) ========================
ALPHA_1 = 0.5                 # 地面等待时间权重
ALPHA_2 = 0.5                 # 总飞行时间权重
W_SEP   = 1e6                 # 冲突惩罚权重（极大正数，确保优先消除冲突）

# ======================== 策略参数范围 (Strategy Bounds) ========================
MAX_GROUND_HOLD     = 1200    # 最大地面等待时间 (s)
SPEED_FACTOR_MIN    = 0.80    # 最小速度调整因子 (80%基础速度)
SPEED_FACTOR_MAX    = 1.20    # 最大速度调整因子 (120%基础速度)
REROUTE_PENALTY_BASE = 300.0  # 重路由基础惩罚代价

# ======================== 遗传算法参数 (GA Hyper-Parameters) ========================
GA_POP_SIZE       = 60        # 种群大小
GA_GENERATIONS    = 100       # 最大进化代数
GA_CROSSOVER_RATE = 0.85      # 交叉概率
GA_MUTATION_RATE  = 0.20      # 变异概率
GA_TOURNAMENT_K   = 3         # 锦标赛选择参赛个体数
GA_ELITE_RATIO    = 0.10      # 精英保留比例

# 连续编码 -> 离散策略 的映射阈值
#   [0.00, 0.40)  -> 策略0: 地面等待 (Ground Holding)
#   [0.40, 0.70)  -> 策略1: 速度调整 (Speed Adjustment)
#   [0.70, 1.00]  -> 策略2: 局部重路由 (Rerouting)
STRATEGY_THRESH_GH = 0.40
STRATEGY_THRESH_SA = 0.70

# ======================== 障碍物/地形参数 (Obstacles / Terrain) ========================
OBSTACLE_SEED       = 2024        # 障碍物生成种子（固定值 → 每次运行地形一致）
NUM_BUILDINGS       = 18          # 建筑物数量
BUILDING_XY_MIN     = 2           # 建筑物底面最小边长 (网格数)
BUILDING_XY_MAX     = 6           # 建筑物底面最大边长 (网格数)
BUILDING_HEIGHT_MIN = 2           # 建筑物最低高度 (z层数, 即 60m)
BUILDING_HEIGHT_MAX = 7           # 建筑物最高高度 (z层数, 即 210m)
BUILDING_MARGIN     = 5           # 建筑物距离空域边界的最小间距 (网格数)
