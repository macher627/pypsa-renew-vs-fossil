import casadi as ca
import numpy as np
def optimize_energy_system_casadi(
    load, wind, pv,
    wind_p_nom=60000,
    pv_p_nom=20000,
    battery_p_nom=30000,
    strompreis=100,
    wind_cap_cost=1200000,
    pv_cap_cost=500000,
    battery_cap_cost=180000,
    battery_nonlin_exp=0.85,
    battery_max_hours=16,
    battery_efficiency=0.9,
    co2_emission_factor=0.5,
    CO2_LIMIT=1e6
):
    N = len(load)
    dt = 0.25  # 15min

    opti = ca.Opti()
    wind_gen = opti.variable(N)
    pv_gen = opti.variable(N)
    backup_gen = opti.variable(N)
    charge = opti.variable(N)
    discharge = opti.variable(N)
    soc = opti.variable(N+1)

    # Calculate backup_max dynamically based on load
    backup_max = max(float(np.max(load)), 120000)
    opti.subject_to(wind_gen >= 0)
    opti.subject_to(wind_gen <= wind_p_nom * wind)
    opti.subject_to(pv_gen >= 0)
    opti.subject_to(pv_gen <= pv_p_nom * pv)
    opti.subject_to(backup_gen >= 0)
    opti.subject_to(backup_gen <= backup_max)
    opti.subject_to(charge >= 0)
    opti.subject_to(discharge >= 0)
    opti.subject_to(charge <= battery_p_nom)
    opti.subject_to(discharge <= battery_p_nom)

    # SOC constraints
    opti.subject_to(soc[0] == 0.5 * battery_p_nom * battery_max_hours)
    opti.subject_to(soc[-1] <= 0.5* battery_p_nom * battery_max_hours)
    opti.subject_to(soc[-1] >= 0)
    opti.subject_to(soc >= 0)
    opti.subject_to(soc <= battery_p_nom * battery_max_hours)

    # Energy balance and SOC update
    for t in range(N):
        opti.subject_to(
            wind_gen[t] + pv_gen[t] + backup_gen[t] + discharge[t] - charge[t] == load[t]
        )
        opti.subject_to(
            soc[t+1] == soc[t] + (battery_efficiency * charge[t] - discharge[t]/battery_efficiency) * dt
        )

    # CO2 constraint
    total_co2 = ca.sum1(backup_gen) * dt * co2_emission_factor
    opti.subject_to(total_co2 <= CO2_LIMIT)

    # Costs
    total_battery_cost = battery_cap_cost * (battery_p_nom * battery_max_hours / (50000*8))**battery_nonlin_exp
    total_wind_cost = wind_cap_cost * wind_p_nom / 50000
    total_pv_cost = pv_cap_cost * pv_p_nom / 30000
    # Recommend using linear backup cost for first runs
    total_backup_cost = strompreis * ca.sum1(backup_gen) * dt

    total_cost = total_battery_cost + total_wind_cost + total_pv_cost + total_backup_cost
    opti.minimize(total_cost)

    # Solve
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

