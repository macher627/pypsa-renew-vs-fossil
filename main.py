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

# 加载两个系统
net_renew = pypsa.Network("system_renewables.nc")
net_fossil = pypsa.Network("system_fossil.nc")

def extract_dispatch_df(network, days=7):
    n = 96 * days  # 每天96个15min
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

# 提取调度数据
df_renew = extract_dispatch_df(net_renew)
df_fossil = extract_dispatch_df(net_fossil)

# 绘图函数
def plot_dispatch(df, title, filename):
    # 只选出除了 Load 以外的所有列（发电来源）
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

# 输出图像
plot_dispatch(df_renew, "System A: Renewables + Backup", "dispatch_renewables.png")
plot_dispatch(df_fossil, "System B: Fossil Only", "dispatch_fossil.png")