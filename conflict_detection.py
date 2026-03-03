"""
模块三：四维飞行冲突检测 (4D Conflict Detection)
=================================================
冲突定义:
  当两架航班的 3D 轨迹交汇于同一网格 w，且到达时间差满足
      |t_a^w - t_b^w| < t_sep
  时，判定为一次 4D 冲突。

核心参数:
  t_sep = 60 秒（确定性安全时间间隔）
"""

import config


def detect_conflicts(trajectories_dict: dict,
                     t_sep: float = config.T_SEP) -> tuple:
    """
    检测所有航班间的四维飞行冲突

    Parameters
    ----------
    trajectories_dict : dict
        {flight_id: [(x, y, z, t), ...]}  每架航班的 4D 轨迹
    t_sep : float
        安全时间间隔阈值 (秒)

    Returns
    -------
    noc : int
        冲突对数 (Number of Conflicts)，每对航班只计一次
    conflict_records : list of dict
        所有冲突事件的详细记录，每条记录包含:
        - flight_a, flight_b : 冲突航班 ID
        - cell              : 发生冲突的网格坐标 (x, y, z)
        - time_a, time_b    : 各自到达该网格的时间 (s)
        - time_diff         : 时间差绝对值 (s)
    """
    # --------- 步骤 1: 构建「网格 → 占用航班列表」索引 ---------
    cell_occupancy = {}   # key: (x,y,z), value: [(flight_id, arrival_time)]

    for fid, traj in trajectories_dict.items():
        for (x, y, z, t) in traj:
            cell = (x, y, z)
            if cell not in cell_occupancy:
                cell_occupancy[cell] = []
            cell_occupancy[cell].append((fid, t))

    # --------- 步骤 2: 逐网格检测冲突 ---------
    conflict_pairs = set()     # 唯一冲突对 (min_id, max_id)
    conflict_records = []      # 详细冲突事件列表

    for cell, occupants in cell_occupancy.items():
        if len(occupants) < 2:
            continue

        # 按到达时间排序，便于利用排序剪枝
        occupants.sort(key=lambda item: item[1])

        n = len(occupants)
        for i in range(n):
            fid_a, t_a = occupants[i]
            for j in range(i + 1, n):
                fid_b, t_b = occupants[j]

                # 已排序: t_b >= t_a，一旦差值超阈值则后续更不可能冲突
                if t_b - t_a >= t_sep:
                    break

                # 排除同一架航班自身的"冲突"
                if fid_a == fid_b:
                    continue

                # 记录冲突
                pair = (min(fid_a, fid_b), max(fid_a, fid_b))
                conflict_pairs.add(pair)
                conflict_records.append({
                    'flight_a': fid_a,
                    'flight_b': fid_b,
                    'cell': cell,
                    'time_a': round(t_a, 2),
                    'time_b': round(t_b, 2),
                    'time_diff': round(abs(t_a - t_b), 2),
                })

    return len(conflict_pairs), conflict_records


def get_conflicting_flight_ids(conflict_records: list) -> set:
    """
    从冲突记录中提取所有涉及冲突的航班 ID 集合

    Parameters
    ----------
    conflict_records : list of dict
        detect_conflicts 的输出

    Returns
    -------
    ids : set of int
    """
    ids = set()
    for rec in conflict_records:
        ids.add(rec['flight_a'])
        ids.add(rec['flight_b'])
    return ids
