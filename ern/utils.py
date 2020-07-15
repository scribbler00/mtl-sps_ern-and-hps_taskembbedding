import sys

sys.path.append("../ern/")
sys.path.append("../dies/")
import copy
import torch
import numpy as np
import pandas as pd
from dies.utils import listify
from sklearn.metrics import mean_squared_error as mse
from torch.utils.data.dataloader import DataLoader
from fastai.basic_data import DataBunch
from fastai.basic_data import DatasetType
import glob


def to_short_name(file):
    return (
        file.split("/")[-1]
        .replace(".h5", "")
        .replace(".csv", "")
        .replace(".pkl", "")
        .replace(".pth", "")
        .replace("_config", "")
    )


def create_databunch(
    train_ds, val_ds, test_ds, batch_size, device,
):

    train_ds.to_device(device)
    tr = DataLoader(
        train_ds,
        batch_size,
        drop_last=True,
        shuffle=True,
        #   num_workers=6,
        pin_memory=False,
    )

    val_ds.to_device(device)
    val = DataLoader(val_ds, batch_size, pin_memory=False)

    if test_ds is not None:
        test_ds.to_device(device)
        test = DataLoader(test_ds, batch_size, pin_memory=False)
    else:
        test = None

    data_bunch = DataBunch(tr, val, test_dl=test)

    return data_bunch


def get_config(file, include_rmse=False):
    df = pd.read_csv(file, sep=",")
    min_rmse_idx = df.root_mean_squared_error.idxmin()
    relevant_cols = [c for c in df.columns if "config" in c]

    rename_cols = {c: c.replace("config/", "") for c in relevant_cols}

    if include_rmse:
        relevant_cols += ["root_mean_squared_error"]

    df = df[relevant_cols].loc[min_rmse_idx]
    df = df.rename(rename_cols)

    return df


def match_file_names(file_name, file_names):
    res = None
    file_name = to_short_name(file_name)
    for f in file_names:
        if file_name == to_short_name(f):
            res = f
            break

    return res


def get_preds(learn, data_type=DatasetType.Test):
    y_hats, y = learn.get_preds(data_type)
    y_hats = np.clip(y_hats, 0, 1.05)
    return y, y_hats


def get_rmse(learn, data_type=DatasetType.Test):
    y, y_hats = get_preds(learn, data_type=data_type)
    y_hats = np.clip(y_hats, 0, 1.05)
    e = mse(y, y_hats) ** 0.5

    return e


def get_ds_from_type(data_bunch, data_type):
    if data_type == DatasetType.Train:
        return data_bunch.train_ds
    elif data_type == DatasetType.Valid:
        return data_bunch.valid_ds
    elif data_type == DatasetType.Test:
        return data_bunch.test_ds


def create_rmse_df_lstm(y, y_hat, file, data_bunch, data_type=DatasetType.Test):
    res_rmses, park_ids = [], []
    pdfs = []
    ds = get_ds_from_type(data_bunch, data_type)

    y, y_hat = y.ravel(), y_hat.ravel()

    res_rmse = mse(y, y_hat) ** 0.5
    res_rmses.append(res_rmse)
    park_ids.append(file)

    df_f = pd.DataFrame({"Y": y, "Yhat": y_hat, "Time": ds.index})
    df_f["ParkId"] = to_short_name(file)
    pdfs.append(df_f)

    df_res = pd.DataFrame({"RMSE": res_rmses, "ParkId": park_ids})
    pdfs = pd.concat(pdfs, axis=0)

    return df_res, pdfs


def create_rmse_df_mtl(y, y_hat, files, data_bunch, data_type=DatasetType.Test):
    res_rmses, park_ids = [], []
    pdfs = []
    ds = get_ds_from_type(data_bunch, data_type)

    for i in range(y.shape[1]):
        res_rmse = mse(y[:, i], y_hat[:, i]) ** 0.5
        res_rmses.append(res_rmse)
        park_ids.append(files[i])

        df_f = pd.DataFrame({"Y": y[:, i], "Yhat": y_hat[:, i], "Time": ds.index})
        df_f["ParkId"] = to_short_name(data_bunch.files[i])
        pdfs.append(df_f)

    df_res = pd.DataFrame({"RMSE": res_rmses, "ParkId": park_ids})
    pdfs = pd.concat(pdfs, axis=0)

    return df_res, pdfs


def create_rmse_df_mlp(y, y_hat, park_ids, data_bunch, data_type=DatasetType.Test):
    cat_park_ids = park_ids.ravel()
    unique_park_ids = np.unique(park_ids)

    ds = get_ds_from_type(data_bunch, data_type)

    res_rmses, park_ids = [], []
    dfs = []
    for cur_park_id in unique_park_ids:
        mask = cat_park_ids == cur_park_id
        cy = y[mask]
        cyh = y_hat[mask]
        cid = ds.index[mask]

        df_f = pd.DataFrame({"Y": cy.ravel(), "Yhat": cyh.ravel(), "Time": cid})
        df_f["ParkId"] = to_short_name(data_bunch.files[cur_park_id])
        dfs.append(df_f)

        res_rmse = mse(cy, cyh) ** 0.5
        res_rmses.append(res_rmse)
        park_ids.append(cur_park_id)
    dfs = pd.concat(dfs, axis=0)
    df_res = pd.DataFrame({"RMSE": res_rmses, "ParkId": park_ids})
    return df_res, dfs


def get_test_results(test_folder):
    files = glob.glob(test_folder + f"/*.csv")

    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f, sep=";"))
    df = pd.concat(dfs, axis=0)

    return df


def get_eval_results(base_folder, data_type):

    forecast_folder = f"{base_folder}/mtl/"
    files = glob.glob(forecast_folder + f"/{data_type}*error.csv")

    forecast_folder = f"{base_folder}/lstm/"
    files = files + glob.glob(forecast_folder + f"/{data_type}*error.csv")

    forecast_folder = f"{base_folder}/mlp/"
    files = files + glob.glob(forecast_folder + f"/{data_type}*error.csv")

    dfs = []
    for f in files:
        dfs.append(pd.read_csv(f, sep=","))
    df = pd.concat(dfs, axis=0)

    return df

