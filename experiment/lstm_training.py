import os

SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

import sys, argparse

sys.path.append("../ern/")
sys.path.append("../../dies/")
sys.path.append("../dies/")
import pandas as pd
import numpy as np
import random, glob, torch
from sklearn.metrics import mean_squared_error as mse

import ray
from ray import tune
from functools import partial
import pathlib

from torch import nn
import fastai
from fastai.basic_train import Learner
from fastai.metrics import mean_squared_error
from fastai.callbacks import OneCycleScheduler
from fastai.basic_data import DatasetType
from fastai.metrics import RMSE, R2Score

from dies.utils import listify
from dies.recurrent_models import LSTMModel
from ern.utils_data import create_databunch_recurrent_data
from ern.utils import get_rmse, to_short_name
from ern.utils_models import count_parameters, get_mlp_model, set_random_states


def train(
    config,
    model_type,
    file,
    data_type="wind",
    save_model=False,
    output_folder="./results/",
):
    device = "cpu"

    scale_data = False

    if "cosmo" in file:
        scale_data = True
        timesteps = 24 * 4
    elif "wind" in file:
        timesteps = 24
    else:
        timesteps = 24 // 3

    data_bunch = create_databunch_recurrent_data(
        file=file,
        config=config,
        device=device,
        timesteps=timesteps,
        scale_data=scale_data,
    )

    data_bunch.files = [file]

    # for some strange reason ray converts the data type, so make sure it is the correct one
    config["num_recurrent_layer"] = int(config["num_recurrent_layer"])
    print(config)

    ann_model = LSTMModel(
        timeseries_length=timesteps,
        n_features_hidden_state=config["n_features_hidden_state"],
        num_recurrent_layer=config["num_recurrent_layer"],
        output_dim=timesteps,
        dropout=config["dropout"],
    ).to(device)

    learn = Learner(
        data_bunch, ann_model, loss_func=torch.nn.MSELoss(), metrics=[RMSE(), R2Score()]
    )
    cb = OneCycleScheduler(learn, lr_max=0.01)

    if config["use_one_cycle"]:
        learn.fit(20, callbacks=[cb])

    learn.fit(config["epochs"], lr=config["lr"])

    if save_model:
        learn.path = pathlib.Path(f"{output_folder}/")
        learn.model_dir = ""
        learn.save(f"{data_type}_{model_type}_{to_short_name(file)}")
        cur_error = get_rmse(learn, data_type=DatasetType.Valid)
        df_res = pd.DataFrame(
            {
                "RMSE": [cur_error],
                "ModelType": ["LSTM"],
                "ParkId": [to_short_name(file)],
            }
        )
        num_params = count_parameters(learn.model)
        df_res["NParametes"] = num_params
        df_res.to_csv(
            f"{output_folder}/{data_type}_{model_type}_{to_short_name(file)}_error.csv"
        )
    else:
        cur_error = get_rmse(learn, data_type=DatasetType.Valid)
        tune.track.init()
        tune.track.log(root_mean_squared_error=cur_error)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_folder",
        help="Path to the data prepared data folder.",
        default="/home/scribbler/workspace/ern_and_te_for_mtl_regression_problems/data/wind/",
    )

    parser.add_argument(
        "--output_folder", help="Path to the store results.", default="./results/",
    )

    parser.add_argument(
        "--num_samples", help="Number of samples in random search.", default=1,
    )

    args = parser.parse_args()
    model_type = "lstm"
    data_folder = args.data_folder
    num_samples = int(args.num_samples)

    output_folder = args.output_folder + f"/{model_type}"
    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    if "wind" in data_folder.lower():
        data_type = "wind"
    else:
        data_type = "pv"

    config = {
        "lr": tune.choice([0.2, 0.1, 0.01, 0.001, 0.0001]),
        "use_one_cycle": tune.choice([True, False]),
        "wd": tune.choice([0.0, 0.05, 0.1, 0.2]),
        "epochs": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([8, 16, 32]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "num_recurrent_layer": tune.choice([1, 2, 4, 8, 10]),
        "n_features_hidden_state": tune.choice([5, 10, 20, 50, 100, 200]),
    }

    files = glob.glob(f"{args.data_folder}/*.csv")

    resources_per_trial = {"cpu": 1}

    for file in files:
        cur_short_name = to_short_name(file)

        def train_wrapper(config):
            train(
                config,
                model_type=model_type,
                file=file,
                data_type=data_type,
                save_model=False,
                output_folder=output_folder,
            )

        if os.path.isabs(output_folder):
            temp_folder = "/mnt/work/transfer/tmp/"
        else:
            temp_folder = "/tmp/"

        set_random_states(42)
        ray.init(num_cpus=SLURM_CPUS_PER_TASK, temp_dir=temp_folder)
        analysis = tune.run(
            train_wrapper,
            name=f"{data_type}_{model_type}_{cur_short_name}",
            config=config,
            verbose=1,
            num_samples=num_samples,
            resources_per_trial=resources_per_trial,
        )

        analysis.dataframe().to_csv(
            f"{output_folder}/{data_type}_{model_type}_{cur_short_name}_config.csv"
        )

        ray.shutdown()

        train(
            analysis.get_best_config(metric="root_mean_squared_error", mode="min"),
            model_type=model_type,
            file=file,
            data_type=data_type,
            save_model=True,
            output_folder=output_folder,
        )
