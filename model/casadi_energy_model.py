def optimize_energy_system_casadi(
    load, wind, pv,
    wind_p_nom=50000,
    pv_p_nom=30000,
    battery_p_nom=50000,
    strompreis=180,
    wind_cap_cost=1200000,
    pv_cap_cost=500000,
    battery_cap_cost=180000,
    battery_nonlin_exp=0.85,
    battery_max_hours=8,
    battery_efficiency=0.9
):
    """
    非线性敏感性分析用CasADi调度优化器。
    参数:
      - load, wind, pv: numpy数组或列表, 长度N
      - *_p_nom: 装机容量/MW
      - *_cap_cost: 单位投资
      - strompreis: 备份边际成本
      - battery_nonlin_exp: 储能投资的非线性指数（默认0.85）
      - battery_max_hours: 储能时长
    返回:
      - total_cost
      - 各时刻[风、光、备份、储能充/放/SOC]的解
    """
    import casadi as ca
    N = len(load)
    dt = 0.25  # 15分钟 -> 小时

    opti = ca.Opti()
    # 决策变量
    wind_gen = opti.variable(N)
    pv_gen = opti.variable(N)
    backup_gen = opti.variable(N)
    charge = opti.variable(N)
    discharge = opti.variable(N)
    soc = opti.variable(N+1)

    # 装机容量约束
    opti.subject_to(wind_gen >= 0)
    opti.subject_to(wind_gen <= wind_p_nom * wind)
    opti.subject_to(pv_gen >= 0)
    opti.subject_to(pv_gen <= pv_p_nom * pv)
    opti.subject_to(backup_gen >= 0)
    opti.subject_to(backup_gen <= 120000)  # 如有最大备份功率
    opti.subject_to(charge >= 0)
    opti.subject_to(discharge >= 0)
    opti.subject_to(charge <= battery_p_nom)
    opti.subject_to(discharge <= battery_p_nom)

    # SOC 约束
    opti.subject_to(soc[0] == 0.5 * battery_p_nom * battery_max_hours)
    opti.subject_to(soc[-1] == 0.5 * battery_p_nom * battery_max_hours)
    opti.subject_to(soc >= 0)
    opti.subject_to(soc <= battery_p_nom * battery_max_hours)

    # 功率平衡+SOC动态
    for t in range(N):
        opti.subject_to(
            wind_gen[t] + pv_gen[t] + backup_gen[t] + discharge[t] - charge[t] == load[t]
        )
        opti.subject_to(
            soc[t+1] == soc[t] + (battery_efficiency * charge[t] - discharge[t]/battery_efficiency) * dt
        )

    # 非线性电池投资（比如按容量^0.85递减）
    total_battery_cost = battery_cap_cost * (battery_p_nom * battery_max_hours / (50000*8))**battery_nonlin_exp
    total_wind_cost = wind_cap_cost * wind_p_nom / 50000
    total_pv_cost = pv_cap_cost * pv_p_nom / 30000
    total_backup_cost = strompreis * ca.sumsqr(backup_gen) * dt  # 线性时可用ca.sum1(backup_gen)
    # 你可以自由组合其它非线性/分段结构

    # 总目标
    total_cost = total_battery_cost + total_wind_cost + total_pv_cost + total_backup_cost
    opti.minimize(total_cost)

    # 求解
    opti.solver('ipopt')
    sol = opti.solve()
    return float(sol.value(total_cost)), {
        "wind": sol.value(wind_gen),
        "pv": sol.value(pv_gen),
        "backup": sol.value(backup_gen),
        "charge": sol.value(charge),
        "discharge": sol.value(discharge),
        "soc": sol.value(soc)
    }