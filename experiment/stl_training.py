import os

SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)

import sys

sys.path.append("../ern/")
sys.path.append("../../dies/")
sys.path.append("../dies/")

import pandas as pd
import numpy as np
import random, glob, torch
from sklearn.metrics import mean_squared_error as mse
from fastai.basic_train import Learner
from fastai.metrics import mean_squared_error
from fastai.callbacks import OneCycleScheduler
from fastai.basic_data import DatasetType

from dies.utils import get_structure
from ern.utils_data import create_databunch_mtl
from ern.utils_models import count_parameters, get_mlp_model, set_random_states
import argparse
from fastai.metrics import RMSE, R2Score
from ray import tune
import ray
from functools import partial
import pathlib
from ern.utils import get_rmse, to_short_name


def train(
    config,
    model_type,
    file,
    data_type="wind",
    save_model=False,
    output_folder="./results/",
):
    device = "cpu"
    use_emb = True
    scale_data = False

    if "gefcom" in file or "cosmo" in file:
        scale_data = True

    data_bunch = create_databunch_mtl(
        [file], batch_size=config["batch_size"], scale_data=scale_data
    )
    data_bunch.files = [file]

    no_input_features = data_bunch.train_ds.x.shape[1]
    no_tasks = data_bunch.train_ds.y.shape[1]

    ann_structure = ann_structure = get_structure(
        no_input_features * config["size_multiplier"],
        config["percental_reduce"],
        11,
        final_outputs=[5, 1],
    )
    ann_model = get_mlp_model(
        ann_structure,
        device,
        no_input_features,
        use_emb=use_emb,
        dropout=config["dropout"],
    )

    learn = Learner(
        data_bunch, ann_model, loss_func=torch.nn.MSELoss(), metrics=[RMSE(), R2Score()]
    )
    cb = OneCycleScheduler(learn, lr_max=0.01)

    if config["use_one_cycle"]:
        learn.fit(20, callbacks=[cb])

    learn.fit(config["epochs"], lr=config["lr"], wd=config["wd"])

    if save_model:
        learn.path = pathlib.Path(f"{output_folder}/")
        learn.model_dir = ""
        learn.save(f"{data_type}_{model_type}_{to_short_name(file)}")
        cur_error = get_rmse(learn, data_type=DatasetType.Valid)
        df_res = pd.DataFrame(
            {
                "RMSE": [cur_error],
                "ModelType": ["BASELINE"],
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

    model_type = "mlp"
    args = parser.parse_args()
    data_folder = args.data_folder
    num_samples = int(args.num_samples)

    output_folder = args.output_folder + f"/{model_type}/"
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
        "batch_size": tune.choice([512, 1024, 2048]),
        "size_multiplier": tune.choice([10]),
        "percental_reduce": tune.choice([50]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    files = glob.glob(f"{args.data_folder}/*.csv")

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
