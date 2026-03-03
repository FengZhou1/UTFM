"""
模块四：战略冲突解决 - 遗传算法 (Strategic Conflict Resolution via GA)
=======================================================================
优化目标:
    Z = α₁·Σ(地面等待时间) + α₂·Σ(飞行时间) + W_sep·剩余冲突数

可用策略:
    0 - 地面等待 (Ground Holding) : 推迟起飞 Δt ∈ [0, 1200]s
    1 - 速度调整 (Speed Adjustment): 速度因子 ∈ [0.8, 1.2]
    2 - 局部重路由 (Rerouting)     : 施加惩罚后重新 A* 寻路

编码方式:
    仅将存在冲突的航班纳入优化。
    每个基因 = [strategy_raw, param_raw]，连续编码 ∈ [0,1]²
    - strategy_raw 映射为离散策略类型
    - param_raw    映射为对应策略的实际参数
"""

import time
import numpy as np

import config
from pathfinding import astar_3d, compute_4d_trajectory
from conflict_detection import detect_conflicts


class ConflictResolver:
    """基于遗传算法的战略冲突解决器"""

    def __init__(self, flights: list, airmatrix):
        """
        Parameters
        ----------
        flights : list of Flight
            所有航班对象
        airmatrix : AirMatrix
            空域网格
        """
        self.all_flights = flights
        self.airmatrix = airmatrix
        self.flight_dict = {f.id: f for f in flights}

        # 所有航班的原始 4D 轨迹（不可变参考基准）
        self.base_trajectories = {f.id: f.trajectory_4d for f in flights}

    # ==================================================================
    #  主入口
    # ==================================================================

    def resolve(self, conflict_flight_ids: set) -> tuple:
        """
        运行遗传算法，消除/减少航班冲突

        Parameters
        ----------
        conflict_flight_ids : set of int
            涉及冲突的航班 ID

        Returns
        -------
        best_trajectories : dict  {fid: [(x,y,z,t), ...]}
        stats : dict              优化统计数据
        """
        self.conflict_ids = sorted(conflict_flight_ids)
        self.n_genes = len(self.conflict_ids)

        if self.n_genes == 0:
            print("  无冲突航班，跳过优化。")
            return dict(self.base_trajectories), self._empty_stats()

        pop_size = config.GA_POP_SIZE
        n_elite = max(1, int(pop_size * config.GA_ELITE_RATIO))

        # -------- 初始化种群 --------
        population = self._init_population(pop_size)

        best_history = []
        noc_history = []
        best_cost_ever = float('inf')
        best_chrom_ever = None
        best_noc_ever = None

        t_start = time.time()

        for gen in range(config.GA_GENERATIONS):
            # ---- 评估 ----
            results = [self._evaluate(chrom) for chrom in population]
            costs = [r[0] for r in results]
            nocs = [r[1] for r in results]

            # ---- 记录本代最优 ----
            idx_best = int(np.argmin(costs))
            if costs[idx_best] < best_cost_ever:
                best_cost_ever = costs[idx_best]
                best_chrom_ever = population[idx_best].copy()
                best_noc_ever = nocs[idx_best]

            best_history.append(best_cost_ever)
            noc_history.append(best_noc_ever)

            # ---- 进度日志 ----
            if (gen + 1) % 10 == 0 or gen == 0:
                elapsed = time.time() - t_start
                print(
                    f"  [GA] 第 {gen + 1:>3d}/{config.GA_GENERATIONS} 代 | "
                    f"最优费用={best_cost_ever:>14.1f} | "
                    f"剩余冲突={best_noc_ever:>4d} | "
                    f"耗时={elapsed:>6.1f}s"
                )

            # ---- 早停 ----
            if best_noc_ever == 0:
                print(f"  [GA] ✔ 第 {gen + 1} 代已消除全部冲突，提前终止！")
                break

            # ---- 生成下一代种群 ----
            sorted_idx = np.argsort(costs)
            new_pop = [population[i].copy() for i in sorted_idx[:n_elite]]

            while len(new_pop) < pop_size:
                p1 = self._tournament_select(population, costs)
                p2 = self._tournament_select(population, costs)

                if np.random.random() < config.GA_CROSSOVER_RATE:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = self._mutate(c1)
                c2 = self._mutate(c2)

                new_pop.append(c1)
                if len(new_pop) < pop_size:
                    new_pop.append(c2)

            population = new_pop[:pop_size]

        # -------- 构建最终解 --------
        best_trajs, total_gh, strategy_stats = self._build_solution(best_chrom_ever)

        stats = {
            'best_cost': best_cost_ever,
            'best_noc': best_noc_ever,
            'total_ground_hold': total_gh,
            'best_history': best_history,
            'noc_history': noc_history,
            'strategy_stats': strategy_stats,
            'time_elapsed': time.time() - t_start,
            'generations_run': len(best_history),
        }
        return best_trajs, stats

    # ==================================================================
    #  种群初始化
    # ==================================================================

    def _init_population(self, pop_size: int) -> list:
        """随机初始化种群，每条染色体为 (n_genes, 2) 的连续编码数组"""
        return [np.random.random((self.n_genes, 2)) for _ in range(pop_size)]

    # ==================================================================
    #  基因解码
    # ==================================================================

    def _decode_gene(self, gene: np.ndarray) -> tuple:
        """
        将连续编码基因解码为 (策略类型, 实际参数)

        gene[0] ∈ [0,1] → 策略类型:
            [0, 0.40) → 0 (地面等待)
            [0.40, 0.70) → 1 (速度调整)
            [0.70, 1.0] → 2 (重路由)

        gene[1] ∈ [0,1] → 实际参数:
            策略0: delay = p * MAX_GROUND_HOLD
            策略1: factor = SPEED_MIN + p * (SPEED_MAX - SPEED_MIN)
            策略2: penalty_factor = 2 + p * 8
        """
        s_raw = gene[0]
        p_raw = float(np.clip(gene[1], 0.0, 1.0))

        if s_raw < config.STRATEGY_THRESH_GH:
            return 0, p_raw * config.MAX_GROUND_HOLD
        elif s_raw < config.STRATEGY_THRESH_SA:
            return 1, (config.SPEED_FACTOR_MIN +
                       p_raw * (config.SPEED_FACTOR_MAX - config.SPEED_FACTOR_MIN))
        else:
            return 2, 2.0 + p_raw * 8.0

    # ==================================================================
    #  适应度评估
    # ==================================================================

    def _evaluate(self, chromosome: np.ndarray) -> tuple:
        """
        评估一条染色体的代价与冲突数

        Returns (cost, noc)
        """
        # 拷贝基础轨迹（浅拷贝字典，列表值在修改时会被整体替换）
        modified_trajs = dict(self.base_trajectories)
        total_ground_hold = 0.0
        reroute_queue = []

        # ---------- Phase 1: 地面等待 & 速度调整（无需重新寻路）----------
        for i, fid in enumerate(self.conflict_ids):
            strategy, param = self._decode_gene(chromosome[i])
            flight = self.flight_dict[fid]

            if strategy == 0:   # 地面等待
                delay = param
                total_ground_hold += delay
                new_traj = compute_4d_trajectory(
                    flight.path_3d, flight.etd + delay, flight.speed)
                modified_trajs[fid] = new_traj

            elif strategy == 1:  # 速度调整
                new_speed = config.CRUISE_SPEED * param
                new_traj = compute_4d_trajectory(
                    flight.path_3d, flight.etd, new_speed)
                modified_trajs[fid] = new_traj

            else:  # strategy == 2, 重路由 → 延迟处理
                reroute_queue.append((i, fid, param))

        # ---------- Phase 2: 重路由（需知其它航班的当前轨迹）----------
        for (_, fid, penalty_factor) in reroute_queue:
            flight = self.flight_dict[fid]
            penalties = self._build_cell_penalties(modified_trajs, fid)
            scaled = {k: v * penalty_factor for k, v in penalties.items()}

            new_path = astar_3d(
                flight.origin, flight.destination,
                self.airmatrix, scaled)
            if new_path is not None:
                new_traj = compute_4d_trajectory(
                    new_path, flight.etd, flight.speed)
                modified_trajs[fid] = new_traj

        # ---------- 计算总飞行时间 ----------
        total_flight_time = 0.0
        for traj in modified_trajs.values():
            if len(traj) > 1:
                total_flight_time += traj[-1][3] - traj[0][3]

        # ---------- 检测冲突 ----------
        noc, _ = detect_conflicts(modified_trajs, config.T_SEP)

        # ---------- 代价函数 ----------
        cost = (config.ALPHA_1 * total_ground_hold +
                config.ALPHA_2 * total_flight_time +
                config.W_SEP * noc)

        return cost, noc

    # ==================================================================
    #  重路由惩罚地图构建
    # ==================================================================

    def _build_cell_penalties(self, trajectories: dict,
                              exclude_fid: int) -> dict:
        """
        构建网格惩罚代价表，用于引导 A* 避开拥堵网格

        思路: 统计除 exclude_fid 外，每个网格被多少架航班途经，
              赋予 count × REROUTE_PENALTY_BASE 的额外代价。
        """
        cell_count = {}
        for fid, traj in trajectories.items():
            if fid == exclude_fid:
                continue
            visited = set()
            for (x, y, z, _) in traj:
                cell = (x, y, z)
                if cell not in visited:
                    visited.add(cell)
                    cell_count[cell] = cell_count.get(cell, 0) + 1

        return {cell: cnt * config.REROUTE_PENALTY_BASE
                for cell, cnt in cell_count.items()}

    # ==================================================================
    #  遗传算子
    # ==================================================================

    def _tournament_select(self, population: list,
                           costs: list) -> np.ndarray:
        """锦标赛选择: 随机取 K 个个体，返回最优者的副本"""
        k = min(config.GA_TOURNAMENT_K, len(population))
        indices = np.random.choice(len(population), size=k, replace=False)
        winner = indices[np.argmin([costs[i] for i in indices])]
        return population[winner].copy()

    def _crossover(self, p1: np.ndarray,
                   p2: np.ndarray) -> tuple:
        """均匀交叉: 以 50% 概率逐基因交换"""
        c1, c2 = p1.copy(), p2.copy()
        mask = np.random.random(self.n_genes) < 0.5
        for i in range(self.n_genes):
            if mask[i]:
                c1[i], c2[i] = c2[i].copy(), c1[i].copy()
        return c1, c2

    def _mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        高斯变异:
          - 对策略维度施加 σ=0.15 的扰动
          - 对参数维度施加 σ=0.20 的扰动
        变异后裁剪至 [0, 1) / [0, 1]
        """
        chrom = chromosome.copy()
        for i in range(self.n_genes):
            if np.random.random() < config.GA_MUTATION_RATE:
                chrom[i, 0] += np.random.normal(0, 0.15)
                chrom[i, 0] = np.clip(chrom[i, 0], 0.0, 0.999)
            if np.random.random() < config.GA_MUTATION_RATE:
                chrom[i, 1] += np.random.normal(0, 0.20)
                chrom[i, 1] = np.clip(chrom[i, 1], 0.0, 1.0)
        return chrom

    # ==================================================================
    #  最终解构建
    # ==================================================================

    def _build_solution(self, chromosome: np.ndarray) -> tuple:
        """
        根据最优染色体重建完整的修正轨迹

        Returns
        -------
        modified_trajs : dict {fid: 4d_traj}
        total_ground_hold : float
        strategy_stats : dict  {ground_hold: n, speed_adjust: n, reroute: n}
        """
        modified_trajs = dict(self.base_trajectories)
        total_ground_hold = 0.0
        counts = {0: 0, 1: 0, 2: 0}
        reroute_queue = []

        for i, fid in enumerate(self.conflict_ids):
            strategy, param = self._decode_gene(chromosome[i])
            counts[strategy] += 1
            flight = self.flight_dict[fid]

            if strategy == 0:
                delay = param
                total_ground_hold += delay
                modified_trajs[fid] = compute_4d_trajectory(
                    flight.path_3d, flight.etd + delay, flight.speed)
            elif strategy == 1:
                new_speed = config.CRUISE_SPEED * param
                modified_trajs[fid] = compute_4d_trajectory(
                    flight.path_3d, flight.etd, new_speed)
            else:
                reroute_queue.append((fid, param))

        for (fid, penalty_factor) in reroute_queue:
            flight = self.flight_dict[fid]
            penalties = self._build_cell_penalties(modified_trajs, fid)
            scaled = {k: v * penalty_factor for k, v in penalties.items()}
            new_path = astar_3d(
                flight.origin, flight.destination,
                self.airmatrix, scaled)
            if new_path is not None:
                modified_trajs[fid] = compute_4d_trajectory(
                    new_path, flight.etd, flight.speed)

        strategy_stats = {
            'ground_hold': counts[0],
            'speed_adjust': counts[1],
            'reroute': counts[2],
        }
        return modified_trajs, total_ground_hold, strategy_stats

    # ------------------------------------------------------------------
    def _empty_stats(self) -> dict:
        return {
            'best_cost': 0, 'best_noc': 0, 'total_ground_hold': 0,
            'best_history': [], 'noc_history': [],
            'strategy_stats': {'ground_hold': 0, 'speed_adjust': 0, 'reroute': 0},
            'time_elapsed': 0, 'generations_run': 0,
        }
