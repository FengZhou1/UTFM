"""
A* 三维寻路算法 与 4D 轨迹时间戳计算
======================================
提供:
  1. astar_3d()           - 3D 网格 A* 最短路径搜索
  2. compute_4d_trajectory() - 由3D路径 + 起飞时间 + 速度 → 4D轨迹
"""

import heapq
import math
import config


# ======================================================================
#  A* 三维网格寻路
# ======================================================================

def _heuristic(node: tuple, goal: tuple,
               dx: float, dy: float, dz: float) -> float:
    """
    启发函数: 欧几里得距离（考虑非均匀网格尺寸）
    保证可采纳 (admissible) 且一致 (consistent)
    """
    return math.sqrt(
        ((node[0] - goal[0]) * dx) ** 2 +
        ((node[1] - goal[1]) * dy) ** 2 +
        ((node[2] - goal[2]) * dz) ** 2
    )


def astar_3d(start: tuple, goal: tuple, airmatrix,
             penalties: dict = None) -> list:
    """
    三维网格 A* 寻路算法

    Parameters
    ----------
    start : (x, y, z)
        起点网格坐标 (整数索引)
    goal : (x, y, z)
        终点网格坐标 (整数索引)
    airmatrix : AirMatrix
        空域网格对象，提供 is_valid / directions / move_costs
    penalties : dict, optional
        额外节点代价 {(x,y,z): float}，用于重路由时引导路径避开拥堵区域

    Returns
    -------
    path : list of (x, y, z) | None
        从 start 到 goal 的有序路径节点列表；无可行路径时返回 None
    """
    if penalties is None:
        penalties = {}

    if start == goal:
        return [start]

    dx = airmatrix.dx
    dy = airmatrix.dy
    dz = airmatrix.dz
    directions = airmatrix.directions
    move_costs = airmatrix.move_costs

    # f = g + h；使用 counter 作为平局打断器，避免元组比较异常
    counter = 0
    h_start = _heuristic(start, goal, dx, dy, dz)
    open_heap = [(h_start, counter, start)]

    came_from = {}            # 前驱节点映射
    g_score = {start: 0.0}   # 起点到各节点的最短已知距离
    closed = set()            # 已确认最短路径的节点集合

    max_iters = 600_000       # 安全上限，防止极端场景下的无限循环

    while open_heap and max_iters > 0:
        max_iters -= 1
        f_curr, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        # === 到达目标，回溯构建路径 ===
        if current == goal:
            path = []
            node = current
            while node is not None:
                path.append(node)
                node = came_from.get(node)
            path.reverse()
            return path

        cx, cy, cz = current
        g_cur = g_score[current]

        for i, (ddx, ddy, ddz) in enumerate(directions):
            nx_ = cx + ddx
            ny_ = cy + ddy
            nz_ = cz + ddz

            if not airmatrix.is_valid(nx_, ny_, nz_):
                continue

            neighbor = (nx_, ny_, nz_)
            if neighbor in closed:
                continue

            # 移动代价 = 实际欧几里得距离 + 额外惩罚代价
            tentative_g = g_cur + move_costs[i] + penalties.get(neighbor, 0.0)

            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                h = _heuristic(neighbor, goal, dx, dy, dz)
                counter += 1
                heapq.heappush(open_heap, (tentative_g + h, counter, neighbor))

    return None  # 未找到可行路径


# ======================================================================
#  4D 轨迹计算（3D路径 → 带时间戳的4D-Trajectory）
# ======================================================================

def compute_4d_trajectory(path_3d: list, etd: float, speed: float) -> list:
    """
    将三维空间路径转换为带时间戳的四维轨迹

    Parameters
    ----------
    path_3d : list of (x, y, z)
        A* 输出的三维路径节点序列
    etd : float
        预计起飞时间 (s)
    speed : float
        恒定巡航速度 (m/s)

    Returns
    -------
    trajectory_4d : list of (x, y, z, t)
        四维轨迹序列；t 为各网格节点的到达时间 (s)
    """
    if not path_3d:
        return []

    dx = config.GRID_SIZE_XY
    dy = config.GRID_SIZE_XY
    dz = config.GRID_SIZE_Z

    trajectory = [(path_3d[0][0], path_3d[0][1], path_3d[0][2], etd)]
    t = etd

    for i in range(1, len(path_3d)):
        # 相邻两节点间的实际欧几里得距离
        seg_dist = math.sqrt(
            ((path_3d[i][0] - path_3d[i - 1][0]) * dx) ** 2 +
            ((path_3d[i][1] - path_3d[i - 1][1]) * dy) ** 2 +
            ((path_3d[i][2] - path_3d[i - 1][2]) * dz) ** 2
        )
        t += seg_dist / speed
        trajectory.append((path_3d[i][0], path_3d[i][1], path_3d[i][2], t))

    return trajectory
