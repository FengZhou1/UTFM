"""
城市空中交通 (UAM) 流量协同优化仿真 - 主运行脚本
===================================================
流程:
  1. 初始化空域网格 (AirMatrix)
  2. 生成 100 架次初始飞行计划 (A* + 4D-Trajectory)
  3. 检测初始四维冲突
  4. 运行遗传算法进行战略冲突解决
  5. 输出最终冲突数量、平均延迟等统计量
  6. 可视化结果
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，兼容无GUI环境
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D           # 3D 绘图
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # 建筑物方块
from matplotlib.patches import Patch

import config
from airspace import AirMatrix
from flight_plan import generate_flights
from conflict_detection import detect_conflicts, get_conflicting_flight_ids
from conflict_resolution import ConflictResolver


# ======================================================================
#  可视化
# ======================================================================

def _draw_buildings_2d(ax, airmatrix, color='gray', alpha=0.55):
    """在 2D 子图上绘制建筑物底面轮廓"""
    for b in airmatrix.buildings:
        rect = plt.Rectangle(
            (b['x0'], b['y0']),
            b['width_x'], b['width_y'],
            linewidth=0.5, edgecolor='dimgray',
            facecolor=color, alpha=alpha, zorder=2
        )
        ax.add_patch(rect)


def _draw_buildings_3d(ax, airmatrix, dx, dy, dz):
    """
    在 3D 子图上绘制建筑物为半透明方块。
    坐标使用实际物理坐标 (m)。
    """
    for b in airmatrix.buildings:
        x0_r = b['x0'] * dx
        y0_r = b['y0'] * dy
        x1_r = b['x1'] * dx
        y1_r = b['y1'] * dy
        z0_r = 0
        z1_r = b['height_z'] * dz

        # 6 个面的顶点
        verts = [
            # 底面
            [(x0_r, y0_r, z0_r), (x1_r, y0_r, z0_r),
             (x1_r, y1_r, z0_r), (x0_r, y1_r, z0_r)],
            # 顶面
            [(x0_r, y0_r, z1_r), (x1_r, y0_r, z1_r),
             (x1_r, y1_r, z1_r), (x0_r, y1_r, z1_r)],
            # 前
            [(x0_r, y0_r, z0_r), (x1_r, y0_r, z0_r),
             (x1_r, y0_r, z1_r), (x0_r, y0_r, z1_r)],
            # 后
            [(x0_r, y1_r, z0_r), (x1_r, y1_r, z0_r),
             (x1_r, y1_r, z1_r), (x0_r, y1_r, z1_r)],
            # 左
            [(x0_r, y0_r, z0_r), (x0_r, y1_r, z0_r),
             (x0_r, y1_r, z1_r), (x0_r, y0_r, z1_r)],
            # 右
            [(x1_r, y0_r, z0_r), (x1_r, y1_r, z0_r),
             (x1_r, y1_r, z1_r), (x1_r, y0_r, z1_r)],
        ]
        poly = Poly3DCollection(verts, alpha=0.35,
                                facecolor='silver',
                                edgecolor='dimgray', linewidth=0.3)
        ax.add_collection3d(poly)


def visualize_results(initial_trajs, resolved_trajs,
                      noc_initial, noc_final,
                      initial_conflicts, final_conflicts,
                      stats, delays, airmatrix,
                      save_path='utfm_results.png'):
    """
    生成综合可视化结果:
      (1) 初始 2D 轨迹 + 建筑物 + 冲突点
      (2) 优化后 3D 轨迹 + 建筑物  ★ 新增立体视图
      (3) GA 收敛曲线
      (4) 统计摘要
    """

    fig = plt.figure(figsize=(18, 16))

    # ---------- (1) 初始轨迹 2D + 建筑物 + 冲突点 ----------
    ax1 = fig.add_subplot(2, 2, 1)
    _draw_buildings_2d(ax1, airmatrix)

    for traj in initial_trajs.values():
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax1.plot(xs, ys, alpha=0.22, linewidth=0.5, color='steelblue')

    conflict_xy = set()
    for rec in initial_conflicts:
        conflict_xy.add((rec['cell'][0], rec['cell'][1]))
    if conflict_xy:
        cx, cy = zip(*conflict_xy)
        ax1.scatter(cx, cy, c='red', s=8, alpha=0.5,
                    zorder=5, label=f'Conflict cells ({len(conflict_xy)})')
    ax1.set_title(f'Initial Trajectories  (NOC = {noc_initial})',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Grid Index')
    ax1.set_ylabel('Y Grid Index')
    ax1.legend(fontsize=9, handles=[
        Patch(facecolor='gray', edgecolor='dimgray', alpha=0.55,
              label=f'Buildings ({len(airmatrix.buildings)})'),
        plt.Line2D([], [], color='steelblue', alpha=0.5, label='Trajectories'),
        plt.Line2D([], [], marker='o', color='red', linestyle='',
                   markersize=4, alpha=0.6, label=f'Conflicts ({len(conflict_xy)})')
    ])
    ax1.set_xlim(-2, config.NX + 2)
    ax1.set_ylim(-2, config.NY + 2)
    ax1.set_aspect('equal')

    # ---------- (2) 优化后轨迹 3D + 建筑物 ★ ----------
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    dx, dy, dz = airmatrix.dx, airmatrix.dy, airmatrix.dz

    # 绘制建筑物方块
    _draw_buildings_3d(ax2, airmatrix, dx, dy, dz)

    # 绘制 3D 轨迹（使用实际坐标 m）
    for traj in resolved_trajs.values():
        xs = [p[0] * dx for p in traj]
        ys = [p[1] * dy for p in traj]
        zs = [p[2] * dz for p in traj]
        ax2.plot(xs, ys, zs, alpha=0.3, linewidth=0.6, color='forestgreen')

    # 标注剩余冲突点
    if final_conflicts:
        final_cells = set()
        for rec in final_conflicts:
            final_cells.add(rec['cell'])
        if final_cells:
            fcx = [c[0] * dx for c in final_cells]
            fcy = [c[1] * dy for c in final_cells]
            fcz = [c[2] * dz for c in final_cells]
            ax2.scatter(fcx, fcy, fcz, c='red', s=12, alpha=0.7,
                        zorder=10, label=f'Remaining conflicts')

    ax2.set_title(f'Resolved 3D Trajectories  (NOC = {noc_final})',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)', fontsize=9)
    ax2.set_ylabel('Y (m)', fontsize=9)
    ax2.set_zlabel('Z (m)', fontsize=9)
    ax2.set_xlim(0, config.NX * dx)
    ax2.set_ylim(0, config.NY * dy)
    ax2.set_zlim(0, config.NZ * dz)
    ax2.view_init(elev=28, azim=-55)

    # 手动创建图例
    legend_elements = [
        Patch(facecolor='silver', edgecolor='dimgray', alpha=0.4,
              label='Buildings'),
        plt.Line2D([], [], color='forestgreen', alpha=0.5,
                   label='Resolved trajectories'),
    ]
    if final_conflicts:
        legend_elements.append(
            plt.Line2D([], [], marker='o', color='red', linestyle='',
                       markersize=4, alpha=0.7, label='Remaining conflicts')
        )
    ax2.legend(handles=legend_elements, fontsize=8, loc='upper left')

    # ---------- (3) GA 收敛曲线 ----------
    ax3 = fig.add_subplot(2, 2, 3)
    if stats['best_history']:
        gens = range(1, len(stats['best_history']) + 1)

        color_cost = 'tab:blue'
        ax3.set_xlabel('Generation', fontsize=11)
        ax3.set_ylabel('Best Cost', color=color_cost, fontsize=11)
        ax3.plot(gens, stats['best_history'], color=color_cost,
                 linewidth=1.5, label='Best Cost')
        ax3.tick_params(axis='y', labelcolor=color_cost)

        ax3_twin = ax3.twinx()
        color_noc = 'tab:red'
        ax3_twin.set_ylabel('NOC (Conflicts)', color=color_noc, fontsize=11)
        ax3_twin.plot(gens, stats['noc_history'], color=color_noc,
                      linewidth=1.5, linestyle='--', label='NOC')
        ax3_twin.tick_params(axis='y', labelcolor=color_noc)

        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2,
                   loc='center right', fontsize=9)

    ax3.set_title('GA Convergence', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ---------- (4) 统计摘要 ----------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    ss = stats['strategy_stats']
    reduction = (1 - noc_final / max(noc_initial, 1)) * 100

    info = (
        "━━━━━  Optimization Summary  ━━━━━\n\n"
        f"  Initial Conflicts (NOC):   {noc_initial}\n"
        f"  Final Conflicts   (NOC):   {noc_final}\n"
        f"  Conflict Reduction:        {reduction:.1f}%\n\n"
        f"  Avg Flight Delay:          {np.mean(delays):.1f} s\n"
        f"  Max Flight Delay:          {np.max(delays):.1f} s\n"
        f"  Total Ground Hold:         {stats['total_ground_hold']:.1f} s\n\n"
        "━━━━  Strategy Distribution  ━━━━\n\n"
        f"  Ground Holding:   {ss['ground_hold']:>3d} flights\n"
        f"  Speed Adjustment: {ss['speed_adjust']:>3d} flights\n"
        f"  Rerouting:        {ss['reroute']:>3d} flights\n\n"
        "━━━━━  Terrain Info  ━━━━━\n\n"
        f"  Buildings:        {len(airmatrix.buildings)}\n"
        f"  Obstacle Cells:   {len(airmatrix.obstacles)}\n\n"
        f"  GA Runtime:  {stats['time_elapsed']:.1f}s  "
        f"({stats['generations_run']} gens)"
    )
    ax4.text(0.05, 0.95, info, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6',
                       facecolor='lightyellow', alpha=0.85))

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ======================================================================
#  主流程
# ======================================================================

def main():
    np.random.seed(42)

    print("=" * 70)
    print("   城市空中交通 (UAM) 流量协同优化仿真系统")
    print("   Strategic Conflict Management based on 4D Trajectories")
    print("=" * 70)

    # ===================== Step 1: 初始化空域 =====================
    print("\n[Step 1] 初始化空域网格...")
    t0 = time.time()
    airmatrix = AirMatrix()
    print(f"  {airmatrix}")
    print(f"  空域范围: {config.AIRSPACE_LENGTH/1000:.0f}km × "
          f"{config.AIRSPACE_WIDTH/1000:.0f}km × "
          f"{config.AIRSPACE_HEIGHT}m")
    print(f"  建筑物: {len(airmatrix.buildings)} 栋, "
          f"障碍网格: {len(airmatrix.obstacles)} 个")

    # ===================== Step 2: 生成飞行计划 =====================
    print(f"\n[Step 2] 生成 {config.NUM_FLIGHTS} 架航班的初始飞行计划 "
          f"(A* 3D寻路, 巡航速度 {config.CRUISE_SPEED} m/s)...")
    t1 = time.time()
    flights = generate_flights(airmatrix, n=config.NUM_FLIGHTS, seed=42)
    print(f"  飞行计划生成完成，耗时 {time.time() - t1:.2f}s")

    # 基本统计
    flight_times = []
    for f in flights:
        if len(f.trajectory_4d) > 1:
            ft = f.trajectory_4d[-1][3] - f.trajectory_4d[0][3]
            flight_times.append(ft)
    print(f"  航班飞行时长: 最短={min(flight_times):.0f}s, "
          f"最长={max(flight_times):.0f}s, "
          f"平均={np.mean(flight_times):.0f}s")

    # ===================== Step 3: 初始冲突检测 =====================
    print(f"\n[Step 3] 检测初始 4D 飞行冲突 (t_sep = {config.T_SEP}s)...")
    traj_dict_initial = {f.id: f.trajectory_4d for f in flights}
    noc_initial, conflicts_initial = detect_conflicts(traj_dict_initial)
    conflict_fids = get_conflicting_flight_ids(conflicts_initial)

    print(f"  初始冲突对数 (NOC):    {noc_initial}")
    print(f"  冲突事件总数:          {len(conflicts_initial)}")
    print(f"  涉及冲突航班数:        {len(conflict_fids)} / {config.NUM_FLIGHTS}")

    if conflicts_initial:
        print(f"\n  部分冲突记录 (前 5 条):")
        for rec in conflicts_initial[:5]:
            print(f"    航班 {rec['flight_a']:>3d} ↔ "
                  f"航班 {rec['flight_b']:>3d}  |  "
                  f"网格 {rec['cell']}  |  "
                  f"Δt = {rec['time_diff']:.1f}s")

    if noc_initial == 0:
        print("\n  ✔ 无冲突，无需优化！")
        print("=" * 70)
        return

    # ===================== Step 4: 遗传算法冲突解决 =====================
    print(f"\n[Step 4] 运行遗传算法进行战略冲突解决...")
    print(f"  优化航班数: {len(conflict_fids)}  |  "
          f"种群: {config.GA_POP_SIZE}  |  "
          f"代数: {config.GA_GENERATIONS}  |  "
          f"交叉率: {config.GA_CROSSOVER_RATE}  |  "
          f"变异率: {config.GA_MUTATION_RATE}")
    print(f"  代价函数: Z = {config.ALPHA_1}·ΣGH + "
          f"{config.ALPHA_2}·ΣFT + {config.W_SEP:.0e}·NOC")
    print()

    resolver = ConflictResolver(flights, airmatrix)
    resolved_trajs, stats = resolver.resolve(conflict_fids)

    # ===================== Step 5: 最终冲突检测 =====================
    print(f"\n[Step 5] 检测优化后冲突...")
    noc_final, conflicts_final = detect_conflicts(resolved_trajs)
    print(f"  最终冲突对数 (NOC): {noc_final}")

    # ===================== Step 6: 结果汇总 =====================
    # 计算延迟: 起飞时间差（仅地面等待策略产生正延迟）
    delays = []
    for f in flights:
        orig_start_t = f.trajectory_4d[0][3] if f.trajectory_4d else 0
        resolved_traj = resolved_trajs.get(f.id, f.trajectory_4d)
        resolved_start_t = resolved_traj[0][3] if resolved_traj else 0
        delays.append(max(0.0, resolved_start_t - orig_start_t))
    delays = np.array(delays)

    print("\n" + "=" * 70)
    print("   ✈  优化结果摘要")
    print("=" * 70)
    reduction = (1 - noc_final / max(noc_initial, 1)) * 100
    print(f"  初始冲突对数 (NOC):     {noc_initial}")
    print(f"  最终冲突对数 (NOC):     {noc_final}")
    print(f"  冲突消除率:             {reduction:.1f}%")
    print(f"  平均航班延迟:           {np.mean(delays):.1f} 秒")
    print(f"  最大航班延迟:           {np.max(delays):.1f} 秒")
    print(f"  总地面等待时间:         {stats['total_ground_hold']:.1f} 秒")
    ss = stats['strategy_stats']
    print(f"  策略分布:")
    print(f"    地面等待 (GH):        {ss['ground_hold']} 架次")
    print(f"    速度调整 (SA):        {ss['speed_adjust']} 架次")
    print(f"    局部重路由 (RR):      {ss['reroute']} 架次")
    print(f"  GA 运行时间:            {stats['time_elapsed']:.1f} 秒")
    print(f"  GA 迭代代数:            {stats['generations_run']} 代")
    print("=" * 70)

    # ===================== Step 7: 可视化 =====================
    print(f"\n[Step 7] 生成可视化图表...")
    visualize_results(
        initial_trajs=traj_dict_initial,
        resolved_trajs=resolved_trajs,
        noc_initial=noc_initial,
        noc_final=noc_final,
        initial_conflicts=conflicts_initial,
        final_conflicts=conflicts_final,
        stats=stats,
        delays=delays,
        airmatrix=airmatrix,
        save_path='utfm_results.png',
    )
    print("  ✔ 结果图表已保存至  utfm_results.png")
    print(f"\n总运行时间: {time.time() - t0:.1f} 秒")
    print("仿真完成！\n")


# ======================================================================
if __name__ == '__main__':
    main()
