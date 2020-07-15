import sys

sys.path.append("../ern/")
sys.path.append("../..dies/dies/")

import pandas as pd
import numpy as np
import glob, os, argparse, copy, tqdm
from ern.shift_features import ShiftFeatures
from ern.utils import to_short_name
import pathlib
from ern.utils_data import (
    create_consistent_number_of_sampler_per_day,
    get_data_with_intersected_timestamps,
)


def get_df_pv(file):
    with pd.HDFStore(file, "r") as store:
        df = store["powerdata"]

    create_timestampindex(df)
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

    i = 0
    ni = []
    for idx in df.index:
        ni.append(idx + pd.DateOffset(hours=i))
        i += 2
    df.index = ni

    sf = ShiftFeatures(
        sliding_window_features=[
            "SolarRadiationDiffuse",
            "SolarRadiationDirect",
            "ClearSkyDirect",
            "ClearSkyDiffuse",
        ],
        eval_position="past",
        step_sizes=[1, 2, 3],
        drop_na=True,
    )

    df = sf.fit_transform(df)

    # df = create_consistent_number_of_sampler_per_day(df, num_samples_per_day=24 / 3)

    cols_to_drop = [
        c for c in df.columns if "sin" in c.lower() or "cos" in c.lower()
    ] + ["Season", "Month"]

    df.drop(cols_to_drop, axis=1, inplace=True)

    df["Hour"] = df.index.hour
    df["DayOfYear"] = df.index.dayofyear

    return df


def get_df_wind(file):
    with pd.HDFStore(file, "r") as store:
        df = store["powerdata"]

    create_timestampindex(df)
    df = df.apply(lambda x: x.fillna(x.mean()), axis=0)

    df.drop("ForecastingTime", inplace=True, axis=1)
    df["Hour"] = df.index.hour
    df["DayOfYear"] = df.index.dayofyear
    sf = ShiftFeatures(
        sliding_window_features=[
            "WindSpeed10m",
            "WindSpeed100m",
            "WindDirectionZonal100m",
            "WindDirectionMeridional100m",
        ],
        eval_position="past",
        step_sizes=[3, 2, 1],
        drop_na=True,
    )

    df = sf.fit_transform(df)

    # df = create_consistent_number_of_sampler_per_day(df)

    return df


def create_timestampindex(df):
    #  as the date is anonymoused crete a proper date
    df.TimeUTC = df.TimeUTC.apply(
        lambda x: x.replace("0000-", "2015-").replace("0001-", "2016-")
    )
    df.TimeUTC = pd.to_datetime(df.TimeUTC, infer_datetime_format=True, utc=True)
    df.set_index("TimeUTC", inplace=True)
    df.index = df.index.rename("TimeUTC")


def main(data_folder, data_type):
    files = glob.glob(data_folder + "/*.h5")
    output_folder = f"data/{data_type}/"
    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    get_data = {"wind": get_df_wind, "pv": get_df_pv}
    min_samples = 22.5 * 30 * 24

    if data_type == "pv":
        min_samples = 8 * 30 * 24

    dfs = {}
    for f in tqdm.tqdm(files):
        df = get_data[data_type](f)

        dfs[to_short_name(f)] = df

    (
        intersected_dfs,
        relevant_parks,
        index_intersection,
    ) = get_data_with_intersected_timestamps(dfs, min_samples=min_samples)

    for park_name, cur_park_df in zip(relevant_parks, intersected_dfs):
        cur_park_df.index = cur_park_df.index.rename("TimeUTC")
        cur_park_df.to_csv(f"{output_folder}{park_name}.csv", sep=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        help="The data folder.",
        default="~/data/prophesy-data/WindSandbox2015/",
    )

    args = parser.parse_args()
    data_folder = args.data_folder

    if "wind" in data_folder.lower():
        data_type = "wind"
    else:
        data_type = "pv"

    if "~" in data_folder:
        data_folder = os.path.expanduser(data_folder)

    main(data_folder, data_type)
