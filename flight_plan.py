"""
模块二：初始飞行计划生成 (Initial Flight Plan Generation)
=========================================================
随机生成 N 个航班的初始飞行计划，包括:
  - 起降点 (O-D) 在空域对立边界随机生成
  - ETD 在 [1, 3600]s 均匀分布
  - 使用 A* 算法进行 3D 路径规划
  - 恒定巡航速度计算 4D 轨迹
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

import config
from pathfinding import astar_3d, compute_4d_trajectory


# ======================================================================
#  航班数据类
# ======================================================================

@dataclass
class Flight:
    """单架航班的完整信息"""
    id: int                                                   # 航班唯一标识
    origin: Tuple[int, int, int]                              # 起点网格坐标
    destination: Tuple[int, int, int]                         # 终点网格坐标
    etd: float                                                # 预计起飞时间 (s)
    speed: float = config.CRUISE_SPEED                        # 巡航速度 (m/s)
    path_3d: List[Tuple[int, int, int]] = field(default_factory=list)
    trajectory_4d: List[Tuple[int, int, int, float]] = field(default_factory=list)


# ======================================================================
#  O-D 生成
# ======================================================================

def _generate_od_pair(rng: np.random.Generator,
                      nx: int, ny: int, nz: int,
                      obstacles: set = None):
    """
    在空域对立边界上随机生成一对起降点 (Origin, Destination)
    自动避开障碍物网格。

    策略:
      - 随机选择对立轴 (X轴 或 Y轴)
      - 起点在一侧边界面上随机取坐标
      - 终点在对面边界面上随机取坐标
      - Z坐标在 [0, nz-1] 范围内随机选取
      - 若落在障碍物上则重新采样

    Parameters
    ----------
    rng : numpy Generator
    nx, ny, nz : 各方向网格数
    obstacles : set of (x,y,z), optional

    Returns
    -------
    origin, destination : tuple (x, y, z)
    """
    if obstacles is None:
        obstacles = set()

    max_retries = 200
    for _ in range(max_retries):
        axis = rng.integers(0, 2)  # 0 → X轴对立, 1 → Y轴对立

        if axis == 0:
            side = rng.integers(0, 2)
            ox = 0 if side == 0 else nx - 1
            dx = nx - 1 if side == 0 else 0
            oy = int(rng.integers(0, ny))
            oz = int(rng.integers(0, nz))
            dy = int(rng.integers(0, ny))
            dz = int(rng.integers(0, nz))
        else:
            side = rng.integers(0, 2)
            oy = 0 if side == 0 else ny - 1
            dy = ny - 1 if side == 0 else 0
            ox = int(rng.integers(0, nx))
            oz = int(rng.integers(0, nz))
            dx = int(rng.integers(0, nx))
            dz = int(rng.integers(0, nz))

        origin = (int(ox), int(oy), int(oz))
        dest   = (int(dx), int(dy), int(dz))

        # 确保起降点均不在障碍物内
        if origin not in obstacles and dest not in obstacles:
            return origin, dest

    # 回退：强制使用边界角点高空层（极低概率走到这里）
    return (0, 0, nz - 1), (nx - 1, ny - 1, nz - 1)


# ======================================================================
#  批量航班生成
# ======================================================================

def generate_flights(airmatrix, n: int = config.NUM_FLIGHTS,
                     seed: int = 42) -> List[Flight]:
    """
    批量生成航班的初始飞行计划

    Parameters
    ----------
    airmatrix : AirMatrix
        三维空域网格对象
    n : int
        航班数量
    seed : int
        随机种子，保证可复现

    Returns
    -------
    flights : list of Flight
    """
    rng = np.random.default_rng(seed)
    flights = []

    print(f"  正在为 {n} 架航班执行 A* 路径规划...")

    for i in range(n):
        # 1. 随机 O-D（避开障碍物）
        origin, destination = _generate_od_pair(
            rng, airmatrix.nx, airmatrix.ny, airmatrix.nz,
            obstacles=airmatrix.obstacles
        )

        # 2. 随机 ETD
        etd = float(rng.uniform(config.ETD_MIN, config.ETD_MAX))

        # 3. A* 三维路径搜索
        path = astar_3d(origin, destination, airmatrix)
        if path is None:
            # 极端情况的安全回退：直线连接起终点
            print(f"  ⚠ 航班 {i}: A* 未找到路径，使用直线回退")
            path = [origin, destination]

        # 4. 计算 4D 轨迹 (恒定速度)
        traj = compute_4d_trajectory(path, etd, config.CRUISE_SPEED)

        flight = Flight(
            id=i,
            origin=origin,
            destination=destination,
            etd=etd,
            speed=config.CRUISE_SPEED,
            path_3d=path,
            trajectory_4d=traj,
        )
        flights.append(flight)

        # 进度反馈
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"    ✔ 已完成 {i + 1:>3d}/{n} 架航班")

    # 简要统计
    lengths = [len(f.path_3d) for f in flights]
    print(f"  路径长度统计: 最短={min(lengths)}, 最长={max(lengths)}, "
          f"平均={np.mean(lengths):.1f} 节点")

    return flights
