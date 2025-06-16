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

def build_system_with_renewables():
    import pandas as pd
    import pypsa

    co2_price = 200  # €/ton
    co2_emission_factor = 0.5  # ton/MWh，比如燃气

    backup_marginal_cost = 80 + co2_price * co2_emission_factor  # = 180 €/MWh
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
    net.add("Generator", "pv", bus="main", p_nom=30000, p_max_pu=df["pv"], marginal_cost=10, capital_cost=500000)
    net.add("Generator", "wind", bus="main", p_nom=50000, p_max_pu=df["wind"], marginal_cost=10, capital_cost=1200000)
    net.add("Generator", "backup",
            bus="main",
            p_nom=120000,
            marginal_cost=backup_marginal_cost,
            capital_cost=200000)
    net.add("StorageUnit", "battery", bus="main", p_nom=50000, max_hours=8,
            efficiency_store=0.9, efficiency_dispatch=0.9, capital_cost=180000,marginal_cost=10)

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


