"""
模块一：城市空域离散化 (Airspace Discretization)
=================================================
使用三维网格矩阵 (AirMatrix) 对城市低空空域进行离散化建模。
空域范围: 10km × 10km × 300m
网格分辨率: 100m × 100m × 30m  →  网格维度 100 × 100 × 10
支持确定性障碍物（建筑物/地形）生成。
"""

import math
import numpy as np
import config


class AirMatrix:
    """
    三维网格空域模型 (3D Air-space Grid Matrix)

    属性:
        nx, ny, nz   : 三个方向的网格数量
        dx, dy, dz   : 各方向网格的实际尺寸 (m)
        directions    : 26连通邻域的方向向量列表
        move_costs    : 对应每个方向向量的实际移动距离 (m)
        obstacles     : set of (x, y, z)  被障碍物占据的网格集合
        buildings     : list of dict  每栋建筑物的描述信息
    """

    def __init__(self):
        # 网格维度
        self.nx = config.NX   # 100
        self.ny = config.NY   # 100
        self.nz = config.NZ   # 10

        # 单个网格的物理尺寸
        self.dx = config.GRID_SIZE_XY   # 100 m
        self.dy = config.GRID_SIZE_XY   # 100 m
        self.dz = config.GRID_SIZE_Z    # 30  m

        # 单网格容量约束
        self.cell_capacity = config.CELL_CAPACITY

        # ---------- 预计算 26 连通邻域的方向向量与对应移动代价 ----------
        self.directions = []    # list of (ddx, ddy, ddz)
        self.move_costs = []    # list of float，与 directions 一一对应

        for ddx in [-1, 0, 1]:
            for ddy in [-1, 0, 1]:
                for ddz in [-1, 0, 1]:
                    if ddx == 0 and ddy == 0 and ddz == 0:
                        continue
                    self.directions.append((ddx, ddy, ddz))
                    # 实际欧几里得距离 (考虑水平100m、垂直30m的非均匀网格)
                    cost = math.sqrt(
                        (ddx * self.dx) ** 2 +
                        (ddy * self.dy) ** 2 +
                        (ddz * self.dz) ** 2
                    )
                    self.move_costs.append(cost)

        # ---------- 障碍物（建筑物地形）----------
        self.obstacles = set()     # 被占据的网格坐标集合
        self.buildings = []        # 建筑物描述列表
        self._generate_buildings()

    # -----------------------------------------------------------------
    #  障碍物生成
    # -----------------------------------------------------------------

    def _generate_buildings(self):
        """
        使用固定随机种子确定性生成建筑物障碍。
        每栋建筑为一个从地面 (z=0) 向上延伸的长方体。
        结果在相同 OBSTACLE_SEED 下完全一致。
        """
        rng = np.random.RandomState(config.OBSTACLE_SEED)   # 固定种子

        margin = config.BUILDING_MARGIN
        placed = []  # 已放置建筑物的 (x0,y0,x1,y1) 列表，用于避免重叠

        attempts = 0
        while len(self.buildings) < config.NUM_BUILDINGS and attempts < 500:
            attempts += 1

            # 随机底面尺寸
            wx = rng.randint(config.BUILDING_XY_MIN, config.BUILDING_XY_MAX + 1)
            wy = rng.randint(config.BUILDING_XY_MIN, config.BUILDING_XY_MAX + 1)
            # 随机高度 (z层数)
            hz = rng.randint(config.BUILDING_HEIGHT_MIN,
                             config.BUILDING_HEIGHT_MAX + 1)
            hz = min(hz, self.nz)   # 不超过空域上限

            # 随机位置（远离边界 margin 格，保证航班在边界能起降）
            x0 = rng.randint(margin, self.nx - margin - wx)
            y0 = rng.randint(margin, self.ny - margin - wy)
            x1 = x0 + wx
            y1 = y0 + wy

            # 检查与已有建筑物是否重叠（留 1 格间隔）
            overlap = False
            for (px0, py0, px1, py1) in placed:
                if not (x1 + 1 <= px0 or x0 >= px1 + 1 or
                        y1 + 1 <= py0 or y0 >= py1 + 1):
                    overlap = True
                    break
            if overlap:
                continue

            # 记录建筑物
            building = {
                'x0': x0, 'y0': y0,
                'x1': x1, 'y1': y1,      # x1/y1 为开区间上界
                'height_z': hz,            # z层数 (从 z=0 到 z=hz-1)
                'width_x': wx, 'width_y': wy,
                'real_height_m': hz * self.dz,
            }
            self.buildings.append(building)
            placed.append((x0, y0, x1, y1))

            # 将建筑物占据的所有网格标记为障碍
            for bx in range(x0, x1):
                for by in range(y0, y1):
                    for bz in range(hz):
                        self.obstacles.add((bx, by, bz))

    # -----------------------------------------------------------------
    #  公共方法
    # -----------------------------------------------------------------

    def is_valid(self, x: int, y: int, z: int) -> bool:
        """判断网格坐标 (x, y, z) 是否在空域范围内 且 未被障碍物占据"""
        if not (0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz):
            return False
        return (x, y, z) not in self.obstacles

    def is_in_bounds(self, x: int, y: int, z: int) -> bool:
        """仅判断坐标是否在空域范围内（不考虑障碍物）"""
        return 0 <= x < self.nx and 0 <= y < self.ny and 0 <= z < self.nz

    def get_neighbors(self, x: int, y: int, z: int):
        """
        获取 (x,y,z) 的所有合法 26 连通邻居及对应移动代价
        自动排除障碍物网格。

        Returns:
            list of ((nx, ny, nz), cost)
        """
        neighbors = []
        for i, (ddx, ddy, ddz) in enumerate(self.directions):
            nx_ = x + ddx
            ny_ = y + ddy
            nz_ = z + ddz
            if self.is_valid(nx_, ny_, nz_):
                neighbors.append(((nx_, ny_, nz_), self.move_costs[i]))
        return neighbors

    def grid_to_real(self, x: int, y: int, z: int):
        """将网格坐标转换为真实世界坐标 (网格中心点, 单位: m)"""
        return (
            x * self.dx + self.dx / 2.0,
            y * self.dy + self.dy / 2.0,
            z * self.dz + self.dz / 2.0,
        )

    def real_distance(self, p1: tuple, p2: tuple) -> float:
        """计算两个网格坐标之间的真实欧几里得距离 (m)"""
        return math.sqrt(
            ((p1[0] - p2[0]) * self.dx) ** 2 +
            ((p1[1] - p2[1]) * self.dy) ** 2 +
            ((p1[2] - p2[2]) * self.dz) ** 2
        )

    def __repr__(self):
        return (
            f"AirMatrix(dims={self.nx}×{self.ny}×{self.nz}, "
            f"cell={self.dx}m×{self.dy}m×{self.dz}m, "
            f"total={self.nx * self.ny * self.nz} cells, "
            f"buildings={len(self.buildings)}, "
            f"obstacle_cells={len(self.obstacles)})"
        )
