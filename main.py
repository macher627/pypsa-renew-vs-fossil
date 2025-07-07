from model.build_network import build_pypsa_network
from pypsa.descriptors import get_switchable_as_dense as sw
from model.build_network import build_system_with_renewables
from model.build_network import build_system_fossil_only
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

net_renew = build_system_with_renewables()
net_fossil = build_system_fossil_only()

net_renew.optimize()
net_fossil.optimize()

net_renew.export_to_netcdf("system_renewables.nc")
net_fossil.export_to_netcdf("system_fossil.nc")

# two system
net_renew = pypsa.Network("system_renewables.nc")
net_fossil = pypsa.Network("system_fossil.nc")

def extract_dispatch_df(network, days=7):
    n = 96 * days  #
    df = pd.DataFrame()
    df["Load"] = network.loads_t.p["load"]
    if "pv" in network.generators.index:
        df["PV"] = network.generators_t.p.get("pv", 0)
    if "wind" in network.generators.index:
        df["Wind"] = network.generators_t.p.get("wind", 0)
    if "backup" in network.generators.index:
        df["Backup"] = network.generators_t.p.get("backup", 0)
    if "battery" in network.storage_units.index:
        df["Battery"] = network.storage_units_t.p_dispatch.get("battery", 0)
    return df.iloc[:n]

# take dispatch Data
df_renew = extract_dispatch_df(net_renew)
df_fossil = extract_dispatch_df(net_fossil)

# plot
def plot_dispatch(df, title, filename):
    #only produce
    gen_cols = [col for col in df.columns if col != "Load"]

    ax = df[gen_cols].fillna(0).plot(
        figsize=(15, 6), stacked=True, alpha=0.7, cmap="tab10"
    )

    df["Load"].plot(ax=ax, color="black", linestyle="--", linewidth=2, label="Load", zorder=5)
    plt.title(title)
    plt.ylabel("Power [MW]")
    plt.xlabel("Time")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# output plot
plot_dispatch(df_renew, "System A: Renewables + Backup", "dispatch_renewables.png")
plot_dispatch(df_fossil, "System B: Fossil Only", "dispatch_fossil.png")

# sensitive analysis strompreise
def run_sensitivity(param_name, param_values, **kwargs):
    results = []
    for value in param_values:
        kw = kwargs.copy()
        kw[param_name] = value
        net = build_system_with_renewables(**kw)
        net.optimize()
        total_cost = net.objective
        results.append({param_name: value, "total_cost": total_cost})
    df = pd.DataFrame(results)
    return df

# --- 1. Strompreis  ---
strompreise = [60, 100, 140, 180, 220]  # €/MWh
df_strom = run_sensitivity("backup_marginal_cost", strompreise)
plt.figure(figsize=(8, 5))
plt.plot(df_strom["backup_marginal_cost"], df_strom["total_cost"], marker="o")
plt.xlabel("Backup Marginal Cost (Strompreis) [€/MWh]")
plt.ylabel("Total System Cost [€]")
plt.title("Sensitivity: Strompreis (Backup Marginal Cost)")
plt.grid(True)
plt.tight_layout()
plt.savefig("sens_strompreis.png")
plt.show()

# --- 2. Wind Investitionskosten  ---
wind_costs = [800000, 1200000, 1600000, 2000000]  # €/MW
df_wind = run_sensitivity("wind_capital_cost", wind_costs)
plt.figure(figsize=(8, 5))
plt.plot(df_wind["wind_capital_cost"], df_wind["total_cost"], marker="o", color="orange")
plt.xlabel("Wind Capital Cost [€/MW]")
plt.ylabel("Total System Cost [€]")
plt.title("Sensitivity: Wind Capital Cost")
plt.grid(True)
plt.tight_layout()
plt.savefig("sens_wind_cost.png")
plt.show()

# --- 3. Storage Capacity---
battery_sizes = [10000, 30000, 50000, 80000]  # MW
df_battery = run_sensitivity("battery_p_nom", battery_sizes)
plt.figure(figsize=(8, 5))
plt.plot(df_battery["battery_p_nom"], df_battery["total_cost"], marker="o", color="green")
plt.xlabel("Battery Capacity [MW]")
plt.ylabel("Total System Cost [€]")
plt.title("Sensitivity: Storage Capacity")
plt.grid(True)
plt.tight_layout()
plt.savefig("sens_battery_size.png")
plt.show()

# casadi analysis
import numpy as np
import matplotlib.pyplot as plt
from model.casadi_energy_model import optimize_energy_system_casadi

# Read time series data and prepare input arrays
import pandas as pd
df = pd.read_csv("data/time_series_15min_singleindex_filtered.csv", parse_dates=["utc_timestamp"])
df = df[["DE_load_actual_entsoe_transparency", "DE_solar_generation_actual", "DE_wind_generation_actual"]].dropna()
df["pv"] = df["DE_solar_generation_actual"] / df["DE_solar_generation_actual"].max()
df["wind"] = df["DE_wind_generation_actual"] / df["DE_wind_generation_actual"].max()
df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}, inplace=True)
N = 96
load = df["load"].values[:N]
wind = df["wind"].values[:N]
pv = df["pv"].values[:N]

# Define the parameter ranges for sensitivity analysis
strompreis_list = [80, 120, 160, 200]
battery_p_nom_list = [5000, 20000, 50000, 100000]

# Initialize result matrix
Z = np.zeros((len(strompreis_list), len(battery_p_nom_list)))

# Sensitivity analysis loop
for i, strompreis in enumerate(strompreis_list):
    for j, battery_p_nom in enumerate(battery_p_nom_list):
        total_cost, sol = optimize_energy_system_casadi(
            load, wind, pv,
            battery_p_nom=battery_p_nom,
            strompreis=strompreis
        )
        Z[i, j] = total_cost

# Plot the sensitivity analysis heatmap
plt.figure(figsize=(8, 6))
plt.title("Gesamtkosten Sensitivitätsanalyse\n(Battery vs. Strompreis, CasADi)")
plt.imshow(Z, aspect='auto', origin='lower', cmap='YlGnBu',
           extent=[min(battery_p_nom_list), max(battery_p_nom_list), min(strompreis_list), max(strompreis_list)])
plt.colorbar(label="Total Cost [€]")
plt.xlabel("Battery Capacity [MW]")
plt.ylabel("Backup Strompreis [€/MWh]")
plt.xticks(battery_p_nom_list)
plt.yticks(strompreis_list)
plt.tight_layout()
plt.savefig("casadi_sensitivity_heatmap.png", dpi=300)
plt.show()

total_costs = []
for strompreis in strompreis_list:
    total_cost, sol = optimize_energy_system_casadi(
        load, wind, pv,
        strompreis=strompreis
    )
    total_costs.append(total_cost)

plt.figure(figsize=(6, 4))
plt.plot(strompreis_list, total_costs, marker='o')
plt.xlabel("Backup Strompreis [€/MWh]")
plt.ylabel("Gesamtkosten [€]")
plt.title("Gesamtkosten vs. Strompreis (CasADi)")
plt.grid(True)
plt.tight_layout()
plt.savefig("casadi_sensitivity_strompreis.png", dpi=300)
plt.show()