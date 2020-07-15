import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
import numpy as np
from scipy.stats.stats import pearsonr
from ern.utils import to_short_name

sns.set()
sns.color_palette("colorblind", 8)
# sns.set_context("poster")
sns.set_style("whitegrid")
mpl.rcParams["legend.loc"] = "upper right"
FONT_SIZE = 30
params = {
    "axes.labelsize": FONT_SIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONT_SIZE,
    "font.size": FONT_SIZE,  # was 10
    "legend.fontsize": FONT_SIZE,  # was 10
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
}


def get_power_data(files):
    power_wind = []
    for f in files:
        df = pd.read_csv(f, sep=";")

        mask = df.TimeUTC.str.contains("2015")

        df = df[mask]

        power_wind.append(df.PowerGeneration.values)

    power_wind = np.array(power_wind)

    return power_wind


def calc_pearson_r(power):
    nt = power.shape[0]
    pcc = np.zeros((nt, nt)) * 0.0

    for i in range(nt):
        for j in range(nt):
            pearson, p_value = pearsonr(power[i, :], power[j, :])
            pcc[i, j] = pearson

    return pcc


data_folder = "./data/wind/"
files = sorted(glob.glob(f"{data_folder}/*.csv"))

power_wind = get_power_data(files)
pearson_correlation = calc_pearson_r(power_wind)
print(
    "Mean Pearson wind",
    np.min(pearson_correlation),
    np.max(pearson_correlation),
    np.mean(pearson_correlation),
    np.median(pearson_correlation),
)


labels = [to_short_name(f).replace("_", "").upper() for f in files]
fig = plt.figure(figsize=(10, 10))
sns.heatmap(pearson_correlation, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
plt.yticks(rotation=45)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig("./doc/images/wind_correlation.png")
plt.close()

data_folder = "./data/pv/"
files = sorted(glob.glob(f"{data_folder}/*.csv"))
power_pv = get_power_data(files)
pearson_correlation = calc_pearson_r(power_pv)
print(
    "Mean Pearson pv",
    np.min(pearson_correlation),
    np.max(pearson_correlation),
    np.mean(pearson_correlation),
    np.median(pearson_correlation),
)


labels = [to_short_name(f).replace("_", "").upper() for f in files]
fig = plt.figure(figsize=(10, 10))
sns.heatmap(pearson_correlation, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
fig.savefig("./doc/images/pv_correlation.png")
plt.close()

data_folder = "./data/pv_cosmo_old/"
files = sorted(glob.glob(f"{data_folder}/*.csv"))
power_pv = get_power_data(files)
pearson_correlation = calc_pearson_r(power_pv)
print(
    "Mean Pearson pv cosmo",
    np.min(pearson_correlation),
    np.max(pearson_correlation),
    np.mean(pearson_correlation),
    np.median(pearson_correlation),
)

labels = [to_short_name(f).replace("_", "").upper() for f in files]
fig = plt.figure(figsize=(10, 10))
sns.heatmap(pearson_correlation, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
fig.savefig("./doc/images/pv_correlation_cosmo.png")
plt.close()

data_folder = "./data/wind_cosmo_old/"
files = sorted(glob.glob(f"{data_folder}/*.csv"))
power_pv = get_power_data(files)
pearson_correlation = calc_pearson_r(power_pv)
print(
    "Mean Pearson wind cosmo",
    np.min(pearson_correlation),
    np.max(pearson_correlation),
    np.mean(pearson_correlation),
    np.median(pearson_correlation),
)
labels = [to_short_name(f).replace("_", "").upper() for f in files]
fig = plt.figure(figsize=(10, 10))
sns.heatmap(pearson_correlation, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
fig.savefig("./doc/images/wind_correlation_cosmo.png")
plt.close()

