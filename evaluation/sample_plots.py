import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.preprocessing import MinMaxScaler
import glob

sns.set()
sns.color_palette("colorblind", 8)
# sns.set_context("poster")
sns.set_style("whitegrid")
mpl.rcParams["legend.loc"] = "upper right"
FONT_SIZE = 40
params = {
    "axes.labelsize": FONT_SIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONT_SIZE,
    "font.size": FONT_SIZE,  # was 10
    "legend.fontsize": FONT_SIZE,  # was 10
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
}

mpl.rcParams.update(params)

file_wind = "./data/wind_cosmo/wind_04.csv"
file_pv = "./data/pv_cosmo/pv_01.csv"


def get_df(file):
    date = pd.to_datetime("2016-01-01", utc=True)
    df = pd.read_csv(file, sep=";")
    df.TimeUTC = pd.to_datetime(df.TimeUTC, infer_datetime_format=True, utc=True)
    df.set_index("TimeUTC", inplace=True)

    df = df[df.index < date]

    if "wind" in file:
        mask = df.WindSpeed73m < 100.0
        df = df[mask]

    df[df.columns] = MinMaxScaler().fit_transform(df[df.columns])

    return df


files_wind = glob.glob("./data/wind_cosmo/*.csv")
to_sn = lambda x: x.split("/")[-1].replace(".csv", "")
for file_wind in files_wind[0:1]:
    df_wind = get_df(file_wind)

    ax1 = df_wind[12000:16000].PowerGeneration.plot(
        figsize=(16, 9), style="-", linewidth=2
    )
    df_wind[12000:16000].WindSpeed122m.plot(figsize=(16, 9), style="--", linewidth=2)

    plt.ylabel("Normalized Power")
    plt.xlabel("Time")

    plt.setp(ax1.get_xticklabels()[0], visible=False)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./doc/images/ts_{to_sn(file_wind)}.png")
    plt.close()

    df_wind.plot(
        x="WindSpeed122m", y="PowerGeneration", kind="scatter", figsize=(16, 9)
    )
    plt.tight_layout()
    plt.ylabel("Normalized Power Generation")
    plt.xlabel("Normalized Wind Speed")
    plt.setp(ax1.get_xticklabels()[0], visible=False)

    plt.savefig(f"./doc/images/sc_{to_sn(file_wind)}.png")
    plt.close()

files_pv = glob.glob("./data/pv_cosmo/*.csv")
for file_pv in files_pv[0:1]:
    df_pv = get_df(file_pv)

    df_pv[12000:12400].PowerGeneration.plot(figsize=(16, 9), style="-", linewidth=2)
    df_pv[12000:12400].DirectRadiation.plot(figsize=(16, 9), style="--", linewidth=2)
    plt.ylabel("Power")
    plt.xlabel("Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./doc/images/ts_{to_sn(file_pv)}.png")
    plt.close()

    df_pv.plot(
        x="DirectRadiation", y="PowerGeneration", kind="scatter", figsize=(16, 9)
    )
    plt.tight_layout()
    plt.ylabel("Normalized Power Generation")
    plt.xlabel("Normalized Direct Radiation")
    plt.savefig(f"./doc/images/sc_{to_sn(file_pv)}.png")
    plt.close()

