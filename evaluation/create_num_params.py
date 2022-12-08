import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from ern.utils import get_eval_results


base_folder = "/home/scribbler/tmp/ern/results/"


def get_nparams(df):
    params, models = [], []
    for k, _df in df.groupby("ModelType", axis=0):
        if k != "BASELINE" and k != "LSTM":
            params.append(_df.NParametes.values[0])
        else:
            params.append(_df.NParametes.values.sum())

        models.append(k)
    df_res = pd.DataFrame({"ModelType": models, "NParametes": params})
    return df_res


for data_set_type, name in zip(
    ["wind_cosmo_old", "wind", "pv_cosmo_old", "pv"],
    ["Wind2015", "EuropeWindFarm", "Solar2015", "GermanSolarFarm"],
):

    if "wind" in data_set_type:
        data_type = "wind"
    else:
        data_type = "pv"

    df = get_eval_results(f"{base_folder}{data_set_type}", data_type=data_type)
    df_res = get_nparams(df)

    print("#", name)
    print()
    print(df_res.to_markdown())
    print()
    print()
