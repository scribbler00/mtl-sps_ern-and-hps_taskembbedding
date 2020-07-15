import sys

sys.path.append("../../dies/")
sys.path.append("../dies/")
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from dies.data import (
    combine_datasets,
    create_databunch,
    ds_from_df,
    split_by_date,
    train_test_split_dataset,
    train_test_split_dataset_by_n_weeks,
    convert_data_to_recurrent_data,
    DatasetRecurrentData,
    scale_datasets,
)
from dies.utils import listify
from dies.utils_pytorch import dev_to_np, np_to_dev


def get_dataset(file, test_date="2016-01-01", park_id=None, num_samples_per_day=-1):
    df = pd.read_csv(file, sep=";")

    df.TimeUTC = pd.to_datetime(df.TimeUTC, infer_datetime_format=True, utc=True)
    df.set_index("TimeUTC", inplace=True)

    cat_cols = ["DayOfYear", "Hour"]
    x_cols = [c for c in df.columns if c not in ["PowerGeneration"] + cat_cols]
    y_cols = "PowerGeneration"

    if park_id is not None:
        df["ParkId"] = park_id
        cat_cols += ["ParkId"]

    if num_samples_per_day != -1:
        df = create_consistent_number_of_sampler_per_day(
            df, num_samples_per_day=num_samples_per_day
        )

    ds = ds_from_df(df, x_columns=x_cols, cat_columns=cat_cols, y_columns=y_cols,)

    ds_tr, ds_te = split_by_date(ds, test_date)

    return ds_tr, ds_te


def create_databunch_mtl_mlp(
    files, batch_size=512, device="cpu", test_date="2016-01-01", dim=1, scale_data=False
):
    ds_trs, ds_vals, ds_tes = [], [], []
    for idx, f in enumerate(files):
        ds_tr, ds_te = get_dataset(f, park_id=idx)
        ds_tr, ds_val = train_test_split_dataset_by_n_weeks(ds_tr)

        ds_trs.append(ds_tr)
        ds_vals.append(ds_val)
        ds_tes.append(ds_te)

    ds_trs = combine_datasets(ds_trs, dim=0)
    ds_vals = combine_datasets(ds_vals, dim=0)
    ds_tes = combine_datasets(ds_tes, dim=0)

    if scale_data:
        scale_datasets(ds_trs, [ds_vals, ds_tes], scaler=MinMaxScaler())

    data_bunch = create_databunch(
        ds_trs, ds_vals, test_ds=ds_tes, batch_size=int(batch_size), device=device
    )
    return data_bunch


def create_databunch_mtl(
    files, batch_size=512, device="cpu", test_date="2016-01-01", dim=1, scale_data=False
):
    ds_trs, ds_tes = [], []
    for f in files:
        ds_tr, ds_te = get_dataset(f)
        ds_trs.append(ds_tr)
        ds_tes.append(ds_te)

    ds_trs = combine_datasets(ds_trs, dim=1)
    ds_tes = combine_datasets(ds_tes, dim=1)

    ds_trs, ds_vals = train_test_split_dataset_by_n_weeks(ds_trs)

    if scale_data:
        scale_datasets(ds_trs, [ds_vals, ds_tes], scaler=MinMaxScaler())

    data_bunch = create_databunch(
        ds_trs, ds_vals, test_ds=ds_tes, batch_size=int(batch_size), device=device
    )
    return data_bunch


def create_databunch_recurrent_data(
    file, config, device, timesteps=24, scale_data=False
):
    ds_tr, ds_te = get_dataset(file, num_samples_per_day=timesteps)
    ds_tr, ds_val = train_test_split_dataset_by_n_weeks(ds_tr)

    if scale_data:
        scale_datasets(ds_tr, [ds_val, ds_te], scaler=MinMaxScaler())

    ds_tr, ds_val, ds_te = create_recurrent_ds(
        [ds_tr, ds_val, ds_te], timesteps=timesteps
    )

    data_bunch = create_databunch(
        ds_tr,
        ds_val,
        test_ds=ds_te,
        batch_size=int(config["batch_size"]),
        device=device,
    )
    return data_bunch


def create_recurrent_ds(datasets, timesteps=24):
    datasets = listify(datasets)
    new_datasets = []
    for ds in datasets:
        new_x = np_to_dev(
            convert_data_to_recurrent_data(dev_to_np(ds.x), timesteps=timesteps)
        )
        new_y = convert_data_to_recurrent_data(dev_to_np(ds.y), timesteps=timesteps)
        ds = DatasetRecurrentData(new_x, new_y)
        new_datasets.append(ds)

    return new_datasets


def create_consistent_number_of_sampler_per_day(df, num_samples_per_day=24):
    mask = df.resample("D").apply(len).PowerGeneration
    mask = (mask < num_samples_per_day) & (mask > 0)
    for i in range(len(mask)):
        if mask[i]:
            new_day = mask.index[i] + pd.DateOffset(days=1)
            new_day.hours = 0

            cur_mask = (df.index < mask.index[i]) | (df.index >= new_day)
            df = df[cur_mask]

    mask = df.resample("D").apply(len).PowerGeneration
    mask = (mask < num_samples_per_day) & (mask > 0)
    if mask.sum() != 0:
        raise ValueError("Wrong sample frequency.")
    return df


def get_data_with_intersected_timestamps(dfs, min_samples=23 * 30 * 24, round="1H"):
    file_names = list(dfs.keys())
    #  find intersecting timestamps
    index_intersection = None

    num_parks_included = 0
    relevant_parks = []
    for file_name in file_names:
        if dfs[file_name].shape[0] < min_samples:
            continue

        if index_intersection is None:
            index_intersection = dfs[file_name].index.round(freq=round)
        else:
            cur_df_index = dfs[file_name].index.round(freq=round)
            mask = cur_df_index.isin(index_intersection)
            index_intersection = cur_df_index[mask]

        relevant_parks.append(file_name)
        num_parks_included += 1

    # filter pased on intersecting timestamps
    intersected_dfs = []
    for file_name in relevant_parks:
        df = dfs[file_name]

        mask = df.index.isin(index_intersection)
        df = copy.copy(df[mask])

        mask = ~df.index.duplicated()
        df = df[mask]

        print(file_name, df.shape, len(np.unique(df.index.values)))
        intersected_dfs.append(df)

    return intersected_dfs, relevant_parks, index_intersection
