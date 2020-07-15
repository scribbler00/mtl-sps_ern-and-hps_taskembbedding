import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from ern.utils import get_eval_results
import copy

sns.set()
sns.color_palette("colorblind", 8)
# sns.set_context("poster")
sns.set_style("whitegrid")
mpl.rcParams["legend.loc"] = "upper center"
FONT_SIZE = 40
params = {
    "axes.labelsize": FONT_SIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONT_SIZE,
    "font.size": FONT_SIZE,  # was 10
    "legend.fontsize": FONT_SIZE - 5,  # was 10
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
}
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
mpl.rcParams.update(params)

wind_folder = "./results/test_forecasts/wind/forecasts/"
pv_folder = "./results/test_forecasts/pv/forecasts/"


def read_df(base_folder, suffix):
    df = pd.read_csv(base_folder + suffix + ".csv", sep=";")
    df.drop("Unnamed: 0", axis=1, inplace=True)

    df.Time = pd.to_datetime(df.Time, infer_datetime_format=True, utc=True)

    return df


def plot_df(
    df, park_id, start, end, ax, label="", plot_target=False, ls=":", alpha=0.75
):

    if park_id is not None:
        df_t = copy.copy(df[df.ParkId == park_id])
    else:
        df_t = copy.copy(df)

    df_t = df_t[start:end]
    df_t.reset_index(drop=True, inplace=True)

    if plot_target:
        (l,) = ax.plot(df_t.Y)
        if label != "":
            l.set_label("Y")

    (l,) = ax.plot(df_t.Yhat, alpha=alpha, linestyle=ls)
    l.set_label(label)


# combined plot

## wind
park_name = "wf25"
plot_target = True
# plt.figure(figsize=(6, 8))
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(16, 9))
ax = axs[0]
# for m, ls in zip(["mlp", "ern", "sn", ], [":", "-.", "--"]):
for m, ls in zip(["mlp", "ern", "sn", "cs"], [":", "-.", "--", ":"]):
    df = read_df(wind_folder, m)
    plot_df(df, park_name, 1100, 1300, ax=ax, label="", plot_target=plot_target, ls=ls)
    plot_target = False

df = read_df(f"{wind_folder}mlp_{park_name}", "")
plot_df(df, None, 1100, 1300, ax=ax, label="", plot_target=plot_target, ls="-.")
df = read_df(f"{wind_folder}lstm_{park_name}", "")
plot_df(df, None, 1100, 1300, ax=ax, label="", plot_target=plot_target, ls="--")

ax.set_xlabel("Timestep")
ax.set_ylabel("Normalized Power")


park_name = "pv_15"
plot_target = True
ax = axs[1]
## pv
# for m, ls in zip(["mlp", "ern", "sn"], [":", "-.", "--"]):
for m, ls in zip(["mlp", "ern", "sn", "cs"], [":", "-.", "--", ":"]):
    df = read_df(pv_folder, m)
    plot_df(
        df,
        park_name,
        1200,
        1300,
        ax=ax,
        label=m.upper(),
        plot_target=plot_target,
        ls=ls,
    )
    plot_target = False

df = read_df(f"{pv_folder}mlp_{park_name}", "")
plot_df(df, None, 1200, 1300, ax=ax, label="STL", plot_target=plot_target, ls="-.")
df = read_df(f"{pv_folder}lstm_{park_name}", "")
plot_df(df, None, 1200, 1300, ax=ax, label="LSTM", plot_target=plot_target, ls="--")

ax.set_xlabel("Timestep")
fig.legend()
fig.tight_layout()
fig.savefig(f"./doc/images/sample_forecast.png")
plt.close()


# one plot per model wind
park_name = "wf25"
plot_target = True
for m, ls in zip(["mlp", "ern", "sn", "cs"], [":", "-.", "--", ":"]):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    df = read_df(wind_folder, m)
    plot_df(
        df,
        park_name,
        1000,
        1300,
        ax=ax,
        label=m.upper(),
        plot_target=plot_target,
        ls=ls,
        alpha=1.0,
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized Power")
    plt.title(m.upper())
    fig.tight_layout()
    fig.savefig(f"./doc/images/{m.lower()}_wind.png")
    plt.close()

fig, ax = plt.subplots(1, figsize=(16, 9))
df = read_df(f"{wind_folder}mlp_{park_name}", "")
plot_df(
    df,
    None,
    1000,
    1300,
    ax=ax,
    label="STL",
    plot_target=plot_target,
    ls="-.",
    alpha=1.0,
)
ax.set_xlabel("Timestep")
ax.set_ylabel("Normalized Power")
plt.title("STL")
fig.tight_layout()
fig.savefig(f"./doc/images/stl_wind.png")
plt.close()

fig, ax = plt.subplots(1, figsize=(16, 9))
df = read_df(f"{wind_folder}lstm_{park_name}", "")
plot_df(
    df,
    None,
    1000,
    1300,
    ax=ax,
    label="LSTM",
    plot_target=plot_target,
    ls="--",
    alpha=1.0,
)
ax.set_xlabel("Timestep")
ax.set_ylabel("Normalized Power")
plt.title("LSTM")
fig.tight_layout()
fig.savefig(f"./doc/images/lstm_wind.png")
plt.close()

# one plot per model pv

park_name = "pv_15"
plot_target = True
for m, ls in zip(["mlp", "ern", "sn", "cs"], [":", "-.", "--", ":"]):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    df = read_df(pv_folder, m)
    plot_df(
        df,
        park_name,
        1000,
        1300,
        ax=ax,
        label=m.upper(),
        plot_target=plot_target,
        ls=ls,
        alpha=1.0,
    )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Normalized Power")
    plt.title(m.upper())
    fig.tight_layout()
    fig.savefig(f"./doc/images/{m.lower()}_pv.png")
    plt.close()

fig, ax = plt.subplots(1, figsize=(16, 9))
df = read_df(f"{pv_folder}mlp_{park_name}", "")
plot_df(
    df,
    None,
    1000,
    1300,
    ax=ax,
    label="STL",
    plot_target=plot_target,
    ls="-.",
    alpha=1.0,
)
ax.set_xlabel("Timestep")
ax.set_ylabel("Normalized Power")
plt.title("STL")
fig.tight_layout()
fig.savefig(f"./doc/images/stl_pv.png")
plt.close()

fig, ax = plt.subplots(1, figsize=(16, 9))
df = read_df(f"{pv_folder}lstm_{park_name}", "")
plot_df(
    df,
    None,
    1000,
    1300,
    ax=ax,
    label="LSTM",
    plot_target=plot_target,
    ls="--",
    alpha=1.0,
)
ax.set_xlabel("Timestep")
ax.set_ylabel("Normalized Power")
plt.title("LSTM")
fig.tight_layout()
fig.savefig(f"./doc/images/lstm_pv.png")
plt.close()
