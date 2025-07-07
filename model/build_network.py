import pandas as pd
import pypsa

def build_pypsa_network():
    # 加载并设定时间戳为索引（UTC 格式）
    df = pd.read_csv(
        "data/time_series_15min_singleindex_filtered.csv",
        parse_dates=["utc_timestamp"]
    )
    df.set_index("utc_timestamp", inplace=True)
    df.index = df.index.tz_convert(None)

    # 只选出这三列（确保无缺失值）
    df = df[[
        "DE_load_actual_entsoe_transparency",
        "DE_solar_generation_actual",
        "DE_wind_generation_actual"
    ]].dropna()

    # 创建归一化列 + 重命名
    df["pv"] = df["DE_solar_generation_actual"] / df["DE_solar_generation_actual"].max()
    df["wind"] = df["DE_wind_generation_actual"] / df["DE_wind_generation_actual"].max()
    df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}, inplace=True)


    # ✅ 建网 + 设置时间
    net = pypsa.Network()
    net.set_snapshots(df.index)

    # 加入组件
    net.add("Bus", "main")
    net.add("Load", "load", bus="main", p_set=df["load"])
    net.add("Generator", "pv", bus="main", p_nom=80000, p_max_pu=df["pv"], marginal_cost=0, capital_cost=500000)
    net.add("Generator", "wind", bus="main", p_nom=100000, p_max_pu=df["wind"], marginal_cost=0, capital_cost=1200000)
    net.add("Generator", "backup", bus="main", p_nom=120000, marginal_cost=80, capital_cost=200000)
    net.add("StorageUnit", "battery", bus="main", p_nom=30000, max_hours=4,
            efficiency_store=0.9, efficiency_dispatch=0.9, capital_cost=400000)

    return net

def build_system_with_renewables(
    wind_factor=1.0,
    pv_factor=1.0,
    backup_marginal_cost=None,
    wind_capital_cost=None,
    pv_capital_cost=None,
    battery_p_nom=None,
    battery_capital_cost=None,
    battery_max_hours=None
):
    import pandas as pd
    import pypsa

    co2_price = 200  # €/ton
    co2_emission_factor = 0.5  # ton/MWh，比如燃气

    # 默认成本参数
    default_backup_cost = 80 + co2_price * co2_emission_factor  # =180 €/MWh
    default_wind_cost = 1200000
    default_pv_cost = 500000
    default_battery_p_nom = 50000
    default_battery_cost = 180000
    default_battery_hours = 8

    backup_marginal_cost = backup_marginal_cost if backup_marginal_cost is not None else default_backup_cost
    wind_capital_cost = wind_capital_cost if wind_capital_cost is not None else default_wind_cost
    pv_capital_cost = pv_capital_cost if pv_capital_cost is not None else default_pv_cost
    battery_p_nom = battery_p_nom if battery_p_nom is not None else default_battery_p_nom
    battery_capital_cost = battery_capital_cost if battery_capital_cost is not None else default_battery_cost
    battery_max_hours = battery_max_hours if battery_max_hours is not None else default_battery_hours

    df = pd.read_csv("data/time_series_15min_singleindex_filtered.csv", parse_dates=["utc_timestamp"])
    df.set_index("utc_timestamp", inplace=True)
    df.index = df.index.tz_convert(None)
    df = df[["DE_load_actual_entsoe_transparency", "DE_solar_generation_actual", "DE_wind_generation_actual"]].dropna()
    df["pv"] = df["DE_solar_generation_actual"] / df["DE_solar_generation_actual"].max()
    df["wind"] = df["DE_wind_generation_actual"] / df["DE_wind_generation_actual"].max()
    df.rename(columns={"DE_load_actual_entsoe_transparency": "load"}, inplace=True)

    net = pypsa.Network()
    net.set_snapshots(df.index)
    net.add("Bus", "main")
    net.add("Load", "load", bus="main", p_set=df["load"])
    net.add("Generator", "pv", bus="main", p_nom=30000*pv_factor, p_max_pu=df["pv"], marginal_cost=10, capital_cost=pv_capital_cost)
    net.add("Generator", "wind", bus="main", p_nom=50000*wind_factor, p_max_pu=df["wind"], marginal_cost=10, capital_cost=wind_capital_cost)
    net.add("Generator", "backup", bus="main", p_nom=120000, marginal_cost=backup_marginal_cost, capital_cost=200000)
    net.add("StorageUnit", "battery", bus="main", p_nom=battery_p_nom, max_hours=battery_max_hours,
            efficiency_store=0.9, efficiency_dispatch=0.9, capital_cost=battery_capital_cost, marginal_cost=10)
    return net


def build_system_fossil_only():
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
    net.add("Generator", "backup", bus="main", p_nom=100000, marginal_cost=140, capital_cost=200000)

    return net


