import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from ern.utils import get_eval_results


sns.set()
sns.color_palette("colorblind", 8)
# sns.set_context("poster")
sns.set_style("whitegrid")
mpl.rcParams["legend.loc"] = "upper right"
FONT_SIZE = 25
params = {
    "axes.labelsize": FONT_SIZE,  # fontsize for x and y labels (was 10)
    "axes.titlesize": FONT_SIZE,
    "font.size": FONT_SIZE,  # was 10
    "legend.fontsize": FONT_SIZE,  # was 10
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
}

mpl.rcParams.update(params)

data_set_type = "wind"
# data_set_type = "wind"
data_type = "wind"
base_folder = "/home/scribbler/tmp/ern/"

df = get_eval_results(f"{base_folder}{data_set_type}", data_type=data_type)


if len(data_set_type) == 0:
    data_set_type = "open"

sns.boxplot(x="ModelType", y="RMSE", data=df, showmeans=True, showfliers=True)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.savefig(f"./results/{data_set_type}_{data_type}.pdf")
plt.show()
# plt.close()


# sns.scatterplot(x="ParkId", y="RMSE", hue="ModelType", data=df)
# sns.lineplot(x="ParkId", y="RMSE", hue="ModelType", data=df)
# plt.xticks(rotation=45)
# plt.show()
plt.close()
