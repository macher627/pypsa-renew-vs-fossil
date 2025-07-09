import pandas as pd
import pypsa

def build_pypsa_network():
    # Laden der CSV-Datei mit Zeitreihendaten und Setzen des Zeitstempels als Index (UTC)
    df = pd.read_csv(
        "data/time_series_15min_singleindex_filtered.csv",
        parse_dates=["utc_timestamp"]
    )
    df.set_index("utc_timestamp", inplace=True)
    df.index = df.index.tz_convert(None)  # Entfernen der Zeitzoneninformation

    # Auswahl der benötigten Spalten (Last, Solar- und Windleistung) und Entfernen von NaNs
    df = df[[
        "DE_load_actual_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_wind_generation_actual"
    ]].dropna()

    # Erzeugung normierter Spalten für PV und Wind (zwischen 0 und 1)
    df["pv"] = df["DE_solar_generation_actual"] / df["DE_solar_generation_actual"].max()
    df["wind"] = df["DE_wind_generation_actual"] / df["DE_wind_generation_actual"].max()
    df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}, inplace=True)

    # Initialisierung des Netzmodells und Setzen der Zeitschnitte (Snapshots)
    net = pypsa.Network()
    net.set_snapshots(df.index)

    # Hinzufügen der Netzkomponenten

    # Ein Bus ("main") als zentrales Verbindungselement
    net.add("Bus", "main")

    # Last mit festem Lastprofil (p_set)
    net.add("Load", "load", bus="main", p_set=df["load"])

    # PV-Erzeuger mit fester Nennleistung, normierter Verfügbarkeit, null variablen Kosten
    net.add("Generator", "pv",
            bus="main",
            p_nom=80000,
            p_max_pu=df["pv"],
            marginal_cost=0,
            capital_cost=500000)

    # Wind-Erzeuger mit höherer Kapazität und Investitionskosten
    net.add("Generator", "wind",
            bus="main",
            p_nom=100000,
            p_max_pu=df["wind"],
            marginal_cost=0,
            capital_cost=1200000)

    # Fossiler Backup-Generator mit hohen variablen Kosten (z. B. Erdgas)
    net.add("Generator", "backup",
            bus="main",
            p_nom=120000,
            marginal_cost=80,
            capital_cost=200000)

    # Batterie als Speicher mit 4 Stunden Speicherdauer und Effizienzparametern
    net.add("StorageUnit", "battery",
            bus="main",
            p_nom=30000,
            max_hours=4,
            efficiency_store=0.9,
            efficiency_dispatch=0.9,
            capital_cost=400000)

    # Rückgabe des aufgebauten Netzobjekts
    return net

def build_system_with_renewables(
    wind_factor=2.0,                 # Skalierungsfaktor für die Windleistung
    pv_factor=2.5,                   # Skalierungsfaktor für die PV-Leistung
    backup_marginal_cost=None,      # Optional: variable Kosten für Backup-Generator
    wind_capital_cost=None,         # Optional: Investitionskosten für Wind [€/MW]
    pv_capital_cost=None,           # Optional: Investitionskosten für PV [€/MW]
    battery_p_nom=None,             # Optional: Batterie-Nennleistung [MW]
    battery_capital_cost=None,      # Optional: Investitionskosten Batterie [€/MW]
    battery_max_hours=None,         # Optional: maximale Speicherstunden [h]
    export_price=None,              # Optional: strompreis verkaufen [€/MWh]
    co2_limit=1e9,                 # CO2 limit [t]
    co2_emission_factor=0.5         # CO2 limit factor [t/MWh]
):
    import pandas as pd
    import pypsa

    # Annahmen zur CO₂-Bepreisung für fossile Erzeugung
    co2_price = 200  # €/Tonne CO₂
    co2_emission_factor = 0.5  # Tonne CO₂ pro MWh (z.B. Erdgas)

    # Standardwerte für Kostenparameter
    default_backup_cost = 80 + co2_price * co2_emission_factor  # =180 €/MWh
    default_wind_cost = 120000    # €/MW Wind
    default_pv_cost = 50000        # €/MW PV
    default_battery_p_nom = 5000   # MW Batterieleistung
    default_battery_cost = 180000   # €/MW Batterie CAPEX
    default_battery_hours = 8        # Stunden Speicherdauer
    default_export_price = 10        # strompreis verkaufen [€/MWh]

    # Zuweisung von Parametern (übergeben oder Standard)
    backup_marginal_cost = backup_marginal_cost if backup_marginal_cost is not None else default_backup_cost
    wind_capital_cost = wind_capital_cost if wind_capital_cost is not None else default_wind_cost
    pv_capital_cost = pv_capital_cost if pv_capital_cost is not None else default_pv_cost
    battery_p_nom = battery_p_nom if battery_p_nom is not None else default_battery_p_nom
    battery_capital_cost = battery_capital_cost if battery_capital_cost is not None else default_battery_cost
    battery_max_hours = battery_max_hours if battery_max_hours is not None else default_battery_hours
    export_price = export_price if export_price is not None else default_export_price

    # Laden der Zeitreihendaten (15-Minuten-Auflösung)
    df = pd.read_csv("data/time_series_15min_singleindex_filtered.csv", parse_dates=["utc_timestamp"])
    df.set_index("utc_timestamp", inplace=True)
    df.index = df.index.tz_convert(None)

    # Auswahl relevanter Spalten und Entfernung von NaNs
    df = df[[
        "DE_load_actual_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_wind_generation_actual"
    ]].dropna()

    # Normierung der PV- und Wind-Erzeugung zur Abbildung als p_max_pu
    df["pv"] = df["DE_solar_generation_actual"] / df["DE_solar_generation_actual"].max()
    df["wind"] = df["DE_wind_generation_actual"] / df["DE_wind_generation_actual"].max()
    df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}, inplace=True)

    # Initialisierung des PyPSA-Netzes und Setzen der Zeitschnitte
    net = pypsa.Network()
    net.set_snapshots(df.index)

    # Hinzufügen eines Busses ("main"), an den alle Komponenten angeschlossen werden
    net.add("Bus", "main")

    # Einbindung der Last mit einem festen Zeitprofil
    net.add("Load", "load", bus="main", p_set=df["load"])

    # PV-Erzeuger mit Skalierung, normierter Verfügbarkeit und Kostenparametern
    net.add("Generator", "pv", bus="main",
            p_nom=60000 * pv_factor,
            p_max_pu=df["pv"],
            marginal_cost=15 - export_price,
            capital_cost=pv_capital_cost,
            )


    # Wind-Erzeuger mit ähnlicher Struktur
    net.add("Generator", "wind", bus="main",
            p_nom=50000 * wind_factor,
            p_max_pu=df["wind"],
            marginal_cost=20-export_price,
            capital_cost=wind_capital_cost)

    # Fossiler Backup-Generator mit hohem variablen Kosten (inkl. CO₂-Preis)
    net.add("Generator", "backup", bus="main",
            p_nom=120000,
            marginal_cost=backup_marginal_cost - export_price,
            capital_cost=200000,
            carrier="gas"  # 关键：标记燃料类型
            )

    # Batterie als Speicher mit Lade-/Entlade-Effizienz, max. Stunden und CAPEX
    net.add("StorageUnit", "battery", bus="main",
            p_nom=battery_p_nom,
            max_hours=battery_max_hours,
            efficiency_store=0.9,
            efficiency_dispatch=0.9,
            capital_cost=battery_capital_cost,
            marginal_cost=20-export_price)

    # 加入碳排系数 (单位t/MWh)
    net.carriers.loc["gas", "co2_emissions"] = co2_emission_factor

    # 增加全局CO₂约束
    net.add("GlobalConstraint",
            "CO2Limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit)

    return net  # Rückgabe des Netzobjekts zur Optimierung


def build_system_fossil_only(
    co2_limit=1e10,    #
    co2_emission_factor=0.5
):
    import pandas as pd
    import pypsa

    df = pd.read_csv("data/time_series_15min_singleindex_filtered.csv", parse_dates=["utc_timestamp"])
    df.set_index("utc_timestamp", inplace=True)
    df.index = df.index.tz_convert(None)
    df = df[["DE_load_actual_entsoe_transparency"]].dropna()
    df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}, inplace=True)

    net = pypsa.Network()
    net.set_snapshots(df.index)
    net.add("Bus", "main")
    net.add("Load", "load", bus="main", p_set=df["load"])
    net.add("Generator", "backup", bus="main",
            p_nom=100000,
            marginal_cost=140,
            capital_cost=200000,
            carrier="gas"
    )


    net.carriers.loc["gas", "co2_emissions"] = co2_emission_factor
    net.add("GlobalConstraint",
            "CO2Limit",
            type="primary_energy",
            carrier_attribute="co2_emissions",
            sense="<=",
            constant=co2_limit)
    return net


