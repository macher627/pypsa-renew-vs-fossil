from model.build_network import build_pypsa_network
from pypsa.descriptors import get_switchable_as_dense as sw
from model.build_network import build_system_with_renewables
from model.build_network import build_system_fossil_only
import pandas as pd
import matplotlib.pyplot as plt
import pypsa


# Aufbau und Optimierung zweier Systeme:
# - net_renew: mit Erneuerbaren + Speicher + Backup
# - net_fossil: nur fossiles Backup

net_renew = build_system_with_renewables()
net_fossil = build_system_fossil_only()

# Durchführung der Optimierung für beide Systeme
net_renew.optimize()
net_fossil.optimize()

# Export der optimierten Netzwerke als NetCDF-Dateien
net_renew.export_to_netcdf("system_renewables.nc")
net_fossil.export_to_netcdf("system_fossil.nc")

# Wiederladen der optimierten Netzwerke für weitere Analyse
net_renew = pypsa.Network("system_renewables.nc")
net_fossil = pypsa.Network("system_fossil.nc")

# Funktion zum Extrahieren der Dispatch-Daten eines Netzwerks (für eine Woche = 7 Tage = 96*7 Zeitpunkte)
def extract_dispatch_df(network, days=7):
    n = 96 * days  # 96 Zeitschritte pro Tag (15-Minuten-Auflösung)
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
    return df.iloc[:n]  # Rückgabe der ersten n Zeitpunkte

# Anwendung der Funktion auf beide Systeme
df_renew = extract_dispatch_df(net_renew)
df_fossil = extract_dispatch_df(net_fossil)

# Funktion zur Visualisierung der Erzeugung vs. Last für ein gegebenes System
def plot_dispatch(df, title, filename):
    # Auswahl der Spalten außer "Load"
    gen_cols = [col for col in df.columns if col != "Load"]

    # Stapeldiagramm der Erzeugerleistung
    ax = df[gen_cols].fillna(0).plot(
        figsize=(15, 6), stacked=True, alpha=0.7, cmap="tab10"
    )

    # Darstellung der Lastkurve als gestrichelte Linie
    df["Load"].plot(ax=ax, color="black", linestyle="--", linewidth=2, label="Load", zorder=5)
    plt.title(title)
    plt.ylabel("Leistung [MW]")
    plt.xlabel("Zeit")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Erstellung und Speicherung der Dispatch-Diagramme
plot_dispatch(df_renew, "System A: Renewables + Backup", "dispatch_renewables.png")
plot_dispatch(df_fossil, "System B: Fossil Only", "dispatch_fossil.png")

# Funktion zur Durchführung einer einfachen Sensitivitätsanalyse:
# - variiert einen Parameter
# - führt jeweils Optimierung durch
# - gibt Gesamtkosten als Ergebnis zurück
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

# --- Sensitivitätsanalyse 1: Strompreis (Backup-Marginalkosten) ---
strompreise = [60, 100, 140, 180, 220]  # €/MWh
df_strom = run_sensitivity("backup_marginal_cost", strompreise)

plt.figure(figsize=(8, 5))
plt.plot(df_strom["backup_marginal_cost"], df_strom["total_cost"], marker="o")
plt.xlabel("Backup Grenzkosten (Strompreis) [€/MWh]")
plt.ylabel("Gesamtsystemkosten [€]")
plt.title("Sensitivität: Strompreis (Backup-Marginalkosten)")
plt.grid(True)
plt.tight_layout()
plt.savefig("sens_strompreis.png")
plt.show()

# --- Sensitivitätsanalyse 2: Wind-Investitionskosten ---
wind_costs = [800000, 1200000, 1600000, 2000000]  # €/MW
df_wind = run_sensitivity("wind_capital_cost", wind_costs)

plt.figure(figsize=(8, 5))
plt.plot(df_wind["wind_capital_cost"], df_wind["total_cost"], marker="o", color="orange")
plt.xlabel("Wind-Investitionskosten [€/MW]")
plt.ylabel("Gesamtsystemkosten [€]")
plt.title("Sensitivität: Wind-Investitionskosten")
plt.grid(True)
plt.tight_layout()
plt.savefig("sens_wind_cost.png")
plt.show()

# --- Sensitivitätsanalyse 3: Batteriespeicherkapazität ---
battery_sizes = [10000, 30000, 50000, 80000]  # MW
df_battery = run_sensitivity("battery_p_nom", battery_sizes)

plt.figure(figsize=(8, 5))
plt.plot(df_battery["battery_p_nom"], df_battery["total_cost"], marker="o", color="green")
plt.xlabel("Batterieleistung [MW]")
plt.ylabel("Gesamtsystemkosten [€]")
plt.title("Sensitivität: Speicherkapazität")
plt.grid(True)
plt.tight_layout()
plt.savefig("sens_battery_size.png")
plt.show()


# casadi analysis
import numpy as np
import matplotlib.pyplot as plt
from model.casadi_energy_model import optimize_energy_system_casadi

import pandas as pd
df = pd.read_csv("data/time_series_15min_singleindex_filtered.csv", parse_dates=["utc_timestamp"])
df = df[["DE_load_actual_entsoe_transparency", "DE_solar_generation_actual", "DE_wind_generation_actual"]].dropna()
df["pv"] = df["DE_solar_generation_actual"] / df["DE_solar_generation_actual"].max()
df["wind"] = df["DE_wind_generation_actual"] / df["DE_wind_generation_actual"].max()
df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}, inplace=True)
N = 96  # Number of time steps (e.g., 1 day)
load = df["load"].values[:N]
wind = df["wind"].values[:N]
pv = df["pv"].values[:N]

# Define sensitivity parameters
strompreis_list = [100, 120, 160, 200]
battery_p_nom_list = [1000, 20000, 50000, 70000]
CO2_LIMIT = 1e6   # For strong CO₂ constraint, set a low value (e.g., 1000 tons). Use 1e10 for "no limit".

# 2D Sensitivity analysis (Battery Capacity vs. Strompreis)
Z = np.zeros((len(strompreis_list), len(battery_p_nom_list)))
backup_util = np.zeros_like(Z)
battery_util = np.zeros_like(Z)

for i, strompreis in enumerate(strompreis_list):
    for j, battery_p_nom in enumerate(battery_p_nom_list):
        total_cost, sol = optimize_energy_system_casadi(
            load, wind, pv,
            battery_p_nom=battery_p_nom,
            strompreis=strompreis,
            CO2_LIMIT=CO2_LIMIT    # Add CO₂ constraint
        )
        Z[i, j] = total_cost
        # Optionally, collect backup and battery utilization for further plots
        backup_util[i, j] = np.sum(sol["backup"])
        battery_util[i, j] = np.sum(sol["charge"])

# Plot total cost heatmap
plt.figure(figsize=(8, 6))
plt.title(f"Total System Cost Sensitivity\n(Battery vs. Strompreis, CO₂ limit: {CO2_LIMIT})")
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

# Plot backup utilization heatmap (optional)
plt.figure(figsize=(8, 6))
plt.title("Total Backup Energy Dispatched [MWh]")
plt.imshow(backup_util, aspect='auto', origin='lower', cmap='YlOrRd',
           extent=[min(battery_p_nom_list), max(battery_p_nom_list), min(strompreis_list), max(strompreis_list)])
plt.colorbar(label="Backup Generation [MWh]")
plt.xlabel("Battery Capacity [MW]")
plt.ylabel("Backup Strompreis [€/MWh]")
plt.xticks(battery_p_nom_list)
plt.yticks(strompreis_list)
plt.tight_layout()
plt.savefig("casadi_sensitivity_backup_util.png", dpi=300)
plt.show()

# 1D Sensitivity analysis: Total cost vs Strompreis (at fixed battery)
fixed_battery = 30000
total_costs = []
backup_totals = []
for strompreis in strompreis_list:
    total_cost, sol = optimize_energy_system_casadi(
        load, wind, pv,
        strompreis=strompreis,
        battery_p_nom=fixed_battery,
        CO2_LIMIT=CO2_LIMIT
    )
    total_costs.append(total_cost)
    backup_totals.append(np.sum(sol["backup"]))

plt.figure(figsize=(6, 4))
plt.plot(strompreis_list, total_costs, marker='o', label="Total Cost")
plt.xlabel("Backup Strompreis [€/MWh]")
plt.ylabel("Total Cost [€]")
plt.title(f"Total Cost vs Strompreis (Battery: {fixed_battery} MW, CO₂ limit: {CO2_LIMIT})")
plt.grid(True)
plt.tight_layout()
plt.savefig("casadi_sensitivity_strompreis.png", dpi=300)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(strompreis_list, backup_totals, marker='s', color='orange', label="Backup Used")
plt.xlabel("Backup Strompreis [€/MWh]")
plt.ylabel("Total Backup Dispatched [MWh]")
plt.title(f"Backup Dispatch vs Strompreis (Battery: {fixed_battery} MW, CO₂ limit: {CO2_LIMIT})")
plt.grid(True)
plt.tight_layout()
plt.savefig("casadi_sensitivity_backup_vs_strompreis.png", dpi=300)
plt.show()

print("All sensitivity analysis and plots completed.")
