import sys, os

sys.path.append("../ern/")
sys.path.append("../..dies/dies/")
sys.path.append(os.path.expanduser("~/workspace/prophesy_code/"))
import pandas as pd
import numpy as np
import glob, argparse, copy, tqdm
from ern.shift_features import ShiftFeatures
from ern.utils import to_short_name
import pathlib
from ern.utils_data import (
    create_consistent_number_of_sampler_per_day,
    get_data_with_intersected_timestamps,
)
import prophesy
from prophesy.utils.utils import get_blacklist


def get_df(file):
    df = pd.read_csv(file, sep=";")

    create_timestampindex(df)

    df = create_consistent_number_of_sampler_per_day(df, num_samples_per_day=24 * 4)

    cols_to_drop = [c for c in df.columns if "sin" in c.lower() or "cos" in c.lower()]

    df.drop(cols_to_drop, inplace=True, axis=1)

    df["Hour"] = df.index.hour
    df["DayOfYear"] = df.index.dayofyear

    return df


def create_timestampindex(df):
    df.PredictionTimeUTC = pd.to_datetime(
        df.PredictionTimeUTC, infer_datetime_format=True, utc=True
    )

    df.rename({"PredictionTimeUTC": "TimeUTC"}, inplace=True)

    df.set_index("PredictionTimeUTC", inplace=True)


def main(data_folder, data_type):
    files = glob.glob(data_folder + "/*.csv")
    output_folder = f"data/{data_type}_cosmo/"
    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    bl = get_blacklist(data_type)

    min_samples = 22.5 * 30 * 24 * 3.565

    if "pv" in data_type:
        min_samples = 12 * 30 * 24 * 4 * 1.125

    dfs = {}
    for f in tqdm.tqdm(files):
        df = get_df(f)
        sn = to_short_name(f)
        if sn in bl:
            print("skipped")
            continue

        time_diff_min = (df.index[11] - df.index[10]).seconds / 60.0

        if time_diff_min != 15:
            print(
                f"WARNING: Skipping file due to time difference is {time_diff_min} instead of 15 mins."
            )
            return pd.DataFrame()

        dfs[sn] = df

    (
        intersected_dfs,
        relevant_parks,
        index_intersection,
    ) = get_data_with_intersected_timestamps(
        dfs, min_samples=min_samples, round="15Min"
    )
    # for anonymisation
    relevant_parks = [f"{data_type}_{i:02}" for i in range(len(relevant_parks))]
    for park_name, cur_park_df in zip(relevant_parks, intersected_dfs):
        cur_park_df.index = cur_park_df.index.rename("TimeUTC")
        cur_park_df.to_csv(f"{output_folder}{park_name}.csv", sep=";")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_folder",
        help="The data folder.",
        default="/home/scribbler/ownCloud/prophesy/AP2Analysis/ies_input_data/wind/day_ahead/",
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
