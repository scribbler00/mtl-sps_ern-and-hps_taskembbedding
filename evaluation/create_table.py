import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from ern.utils import get_eval_results, get_test_results
import numpy as np

base_folder = "/home/scribbler/tmp/ern/"
base_folder = "/home/scribbler/workspace/ern_and_te_for_mtl_regression_problems/results/test_forecasts/"
eval_type = "test"


def get_stats(df):
    errors = df.RMSE
    return np.percentile(errors, [10, 50, 90])


def get_skill_score(df_baseline, df_reference):
    parks = df_baseline.ParkId
    ss = []
    for park in parks:
        cur_df_reference = df_reference[df_reference.ParkId == park]
        cur_df_baseline = df_baseline[df_baseline.ParkId == park]
        if cur_df_reference.shape[0] > 0:
            ss.append(1 - cur_df_reference.RMSE.values / cur_df_baseline.RMSE.values)

    if len(ss) == 0:
        return np.nan
    else:
        return np.mean(ss)  # * 100


def eval_single_experiment(df_to_eval, a=1):
    q1s, q5s, q9s, sss, models = [], [], [], [], []
    for model, df in df_to_eval.groupby("ModelType"):
        q1, q5, q9 = get_stats(df)
        ss = get_skill_score(df_to_eval[df_to_eval.ModelType == "BASELINE"], df)
        q1s.append(q1)
        q5s.append(q5)
        q9s.append(q9)
        # print(ss)
        # if len(ss) == 0:
        #     ss = [-1]
        sss.append(ss)
        models.append(model)

    df_eval = pd.DataFrame(
        {
            "Quantile 0.1": q1s,
            "Quantile 0.5": q5s,
            "Quantile 0.9": q9s,
            "Mean SS": sss,
            "Model": models,
        }
    )
    return df_eval


if eval_type == "test":
    df_open_wind = get_test_results(f"{base_folder}/wind/")
    df_open_pv = get_test_results(f"{base_folder}/pv/")

    df_cosmo_wind = get_test_results(f"{base_folder}/wind_cosmo_old")
    df_cosmo_pv = get_test_results(f"{base_folder}/pv_cosmo_old")
else:
    df_cosmo_wind = get_eval_results(f"{base_folder}/wind_cosmo_old", data_type="wind")
    df_cosmo_pv = get_eval_results(f"{base_folder}/pv_cosmo_old", data_type="pv")

    df_open_wind = get_eval_results(f"{base_folder}/wind/", data_type="wind")
    df_open_pv = get_eval_results(f"{base_folder}/pv/", data_type="pv")


eval_dfs = []
df_eval = eval_single_experiment(df_cosmo_wind)
df_eval["Dataset"] = "Wind2015"
eval_dfs.append(df_eval)

df_eval = eval_single_experiment(df_open_wind)
df_eval["Dataset"] = "EuropeWindFarm"
eval_dfs.append(df_eval)

df_eval = pd.concat(eval_dfs, axis=0)

df_eval.set_index(["Dataset", "Model"], inplace=True)
print(df_eval.round(3).T.to_latex())

eval_dfs = []
df_eval = eval_single_experiment(df_cosmo_pv)
df_eval["Dataset"] = "Solar2015"
eval_dfs.append(df_eval)

df_eval = eval_single_experiment(df_open_pv)
df_eval["Dataset"] = "GermanSolarFarm"
eval_dfs.append(df_eval)

df_eval = pd.concat(eval_dfs, axis=0)

df_eval.set_index(["Dataset", "Model"], inplace=True)
# print(df_eval.T.LSTM.apply(float).dtypes)
print(df_eval.round(3).T.to_latex())
