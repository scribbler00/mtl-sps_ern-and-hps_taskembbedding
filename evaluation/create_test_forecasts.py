import sys

sys.path.append("../../dies/")
sys.path.append("../dies/")
import pathlib
import argparse
import glob
import pandas as pd
import numpy as np
import torch
import pickle as pkl
import warnings
from sklearn.metrics import mean_squared_error as mse

from fastai.basic_train import Learner
from fastai.metrics import R2Score, RMSE
from fastai.basic_data import DatasetType


from dies.utils_pytorch import dev_to_np, np_to_dev
from dies.recurrent_models import LSTMModel

from ern.utils import get_preds, get_rmse, to_short_name
from ern.utils import (
    get_config,
    match_file_names,
    to_short_name,
    create_rmse_df_mtl,
    create_rmse_df_mlp,
    create_rmse_df_lstm,
)
from ern.utils_data import (
    create_databunch_mtl,
    create_databunch_mtl_mlp,
    create_databunch_recurrent_data,
)
from ern.utils_models import get_mlp_model, get_model


def create_forecasts(config, model_type, data_bunch, model_file, forecast_type="mtl"):
    device = "cpu"
    use_emb = True

    if "cosmo" in data_bunch.files:
        timesteps = 24 * 4
    elif "wind" in data_bunch.files:
        timesteps = 24
    else:
        timesteps = 24 // 3

    no_input_features = data_bunch.train_ds.x.shape[1]
    no_tasks = data_bunch.train_ds.y.shape[1]

    num_supaces = get_sn_config(config)

    dropout = config["dropout"]

    if forecast_type == "mtl":
        ann_model = get_model(
            model_type,
            no_input_features,
            config,
            device,
            no_tasks,
            use_emb,
            data_bunch.files,
            dropout,
            num_supaces,
        )
    elif model_type == "mlp":
        ann_model = create_mlp_model(no_input_features, config, device, use_emb)
    elif model_type == "lstm":
        ann_model = create_lstm_model(config, timesteps)

    learn = Learner(data_bunch, ann_model, loss_func=torch.nn.MSELoss())

    learn.model_dir = ""
    learn = learn.load(model_file.replace(".pth", ""), strict=True)

    df_res, df_for = create_eval_df(learn, model_type, forecast_type, data_bunch)

    if forecast_type == "mtl":
        df_res["ParkId"] = [to_short_name(f) for f in data_bunch.files]

    return df_res, df_for


def get_sn_config(config):
    if "num_supaces" in config.keys():
        num_supaces = config["num_supaces"]
    else:
        num_supaces = 1
    return num_supaces


def create_lstm_db(files, config, device):
    scale_data = False
    if "cosmo" in files[0]:
        scale_data = True
        timesteps = 24 * 4
    elif "wind" in files[0]:
        timesteps = 24
    else:
        timesteps = 24 // 3

    data_bunch = create_databunch_recurrent_data(
        file=files[0],
        config=config,
        device=device,
        timesteps=timesteps,
        scale_data=scale_data,
    )
    return data_bunch, timesteps


def create_eval_df(learn, model_type, forecast_type, data_bunch):
    y, y_hat = get_preds(learn, DatasetType.Test)
    y, y_hat = dev_to_np([y, y_hat])
    y_hat = np.clip(y_hat, 0, 1.05)

    if model_type == "mlp" and forecast_type == "mtl":
        # mlp mtl
        df_res, df_for = create_rmse_df_mlp(
            y, y_hat, dev_to_np(data_bunch.test_ds.cat[:, -1]), learn.data
        )
    elif model_type == "lstm":
        df_res, df_for = create_rmse_df_lstm(y, y_hat, data_bunch.files, data_bunch)
    else:
        # stl, mtl
        df_res, df_for = create_rmse_df_mtl(y, y_hat, data_bunch.files, learn.data)

    if forecast_type == "mtl":
        df_res["ModelType"] = model_type.upper()

    return df_res, df_for


def create_forecast_df(y, y_hat, data_bunch):
    pdfs = []
    for i in range(y.shape[1]):
        df_f = pd.DataFrame(
            {"Y": y[:, i], "Yhat": y_hat[:, i], "Time": data_bunch.test_ds.index}
        )
        df_f["ParkId"] = to_short_name(data_bunch.files[i])
        pdfs.append(df_f)
    pdfs = pd.concat(pdfs, axis=1)

    return pdfs


def create_lstm_model(config, timesteps):
    config["num_recurrent_layer"] = int(config["num_recurrent_layer"])

    ann_model = LSTMModel(
        timeseries_length=timesteps,
        n_features_hidden_state=config["n_features_hidden_state"],
        num_recurrent_layer=config["num_recurrent_layer"],
        output_dim=timesteps,
        dropout=config["dropout"],
    )
    return ann_model


def create_mlp_model(no_input_features, config, device, use_emb):
    from dies.utils import get_structure

    ann_structure = ann_structure = get_structure(
        no_input_features * config["size_multiplier"],
        config["percental_reduce"],
        11,
        final_outputs=[5, 1],
    )
    ann_model = get_mlp_model(ann_structure, device, no_input_features, use_emb=use_emb)

    return ann_model


def split_mtl_stl(files):
    stl_files = [f for f in files if ("wf" in f) or ("pv" in f)]
    mtl_files = [f for f in files if ("wf" not in f) and ("pv" not in f)]

    return stl_files, mtl_files


def get_forecast_type(mt):
    if mt == "mlp" or mt == "lstm":
        forecast_type = "stl"
    else:
        forecast_type = "mtl"
    return forecast_type


def get_model_type(cur_param_file):
    model_type = (
        to_short_name(cur_param_file)
        .replace("wind_", "")
        .replace("pv_", "")
        .replace("_config", "")
    ).split("_")[0]
    return model_type


def get_park_name(cur_param_file, data_files):
    park_name = to_short_name(cur_param_file).split("_")[-1]
    park_name = [match_file_names(park_name, data_files)]

    return park_name


def get_park_name_cosmo(cur_param_file, data_files):
    park_name = "_".join(to_short_name(cur_param_file).split("_")[-2:])
    park_name = [match_file_names(park_name, data_files)]

    return park_name


def prep_stl_df(dfs, mt):
    df = pd.concat(dfs, axis=0)
    if mt == "lstm":
        cur_model_type = "LSTM"
    else:
        cur_model_type = "BASELINE"
    df["ModelType"] = cur_model_type
    return df, cur_model_type


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        help="Path to the data prepared data folder.",
        default="./data/pv/",
    )

    parser.add_argument(
        "--output_folder",
        help="Path to the store results.",
        default="./results/test_forecasts/",
    )

    parser.add_argument(
        "--base_folder",
        help="Folder to the grid search parameters.",
        default="/home/scribbler/tmp/ern/results/pv/",
    )

    args = parser.parse_args()
    data_folder = args.data_folder
    base_folder = args.base_folder

    model_types = ["mtl", "mlp", "lstm"]

    if "wind" in data_folder.lower():
        data_type = "wind"
    else:
        data_type = "pv"

    # for mf in model_types:
    output_folder = args.output_folder + "/" + base_folder.split("/")[-2]
    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)
    pathlib.Path(output_folder + "/forecasts/").mkdir(exist_ok=True, parents=True)

    param_files = {
        mf: glob.glob(f"{base_folder}/{mf}/*config.csv") for mf in model_types
    }
    model_files = {mf: glob.glob(f"{base_folder}/{mf}/*.pth") for mf in model_types}
    data_files = glob.glob(f"{data_folder}/*.csv")

    for mt in model_types:
        print(mt)
        forecast_type = get_forecast_type(mt)

        cur_param_files = param_files[mt]
        cur_model_files = model_files[mt]
        dfs = []
        for cur_param_file in cur_param_files:

            cur_config = get_config(cur_param_file, include_rmse=False)

            model_type = get_model_type(cur_param_file)
            model_file = match_file_names(
                to_short_name(cur_param_file), cur_model_files
            )

            if forecast_type == "stl":
                if "cosmo" in data_files[0] or "pv" in data_files[0]:
                    park_name = get_park_name_cosmo(cur_param_file, data_files)
                else:
                    park_name = get_park_name(cur_param_file, data_files)

                park_short_name = to_short_name(park_name[0])

                if "lstm" in model_type:
                    data_bunch = pkl.load(
                        open(f"{data_folder}/dbs/{park_short_name}_lstm.pkl", "rb",)
                    )
                else:
                    data_bunch = pkl.load(
                        open(f"{data_folder}/dbs/{park_short_name}_stl.pkl", "rb",)
                    )

                df_res, df_for = create_forecasts(
                    cur_config,
                    model_type,
                    data_bunch,
                    model_file,
                    forecast_type=forecast_type,
                )
                df_res["ParkId"] = park_short_name
                dfs.append(df_res)
                df_for.to_csv(
                    f"{output_folder}/forecasts/{model_type}_{park_short_name}.csv",
                    sep=";",
                )
            else:
                if "mlp" in model_type:
                    data_bunch = pkl.load(open(f"{data_folder}/dbs/mtl_mlp.pkl", "rb"))
                else:
                    data_bunch = pkl.load(open(f"{data_folder}/dbs/mtl.pkl", "rb"))

                df_res, df_for = create_forecasts(
                    cur_config,
                    model_type,
                    data_bunch,
                    model_file,
                    forecast_type=forecast_type,
                )
                df_res.to_csv(f"{output_folder}/{model_type}.csv", sep=";")
                df_for.to_csv(f"{output_folder}/forecasts/{model_type}.csv", sep=";")

        if forecast_type == "stl" and len(dfs) > 0:
            df, cur_model_type = prep_stl_df(dfs, mt)
            df.to_csv(f"{output_folder}/{cur_model_type.lower()}.csv", sep=";")

