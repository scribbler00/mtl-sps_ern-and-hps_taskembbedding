import os

SLURM_CPUS_PER_TASK = os.getenv("SLURM_CPUS_PER_TASK")
if SLURM_CPUS_PER_TASK is None:
    SLURM_CPUS_PER_TASK = "4"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


SLURM_CPUS_PER_TASK = int(SLURM_CPUS_PER_TASK)
import torch

torch.set_num_threads(1)
import sys
import pickle as pkl

sys.path.append("../ern/")
sys.path.append("../../dies/")
sys.path.append("../dies/")
import argparse
import pandas as pd
import numpy as np
import random, glob, pathlib
from functools import partial

from fastai.basic_train import Learner
from fastai.metrics import mean_squared_error
from fastai.callbacks import OneCycleScheduler
from fastai.basic_data import DatasetType
from fastai.metrics import RMSE, R2Score
from fastai.layers import MSELossFlat

from ray import tune
import ray

from dies.utils import get_structure
from dies.utils_pytorch import dev_to_np
from ern.mtl_loss import MultiTaskLossWrapper
from ern.utils_data import create_databunch_mtl, create_databunch_mtl_mlp
from ern.utils_models import (
    get_cs_model,
    get_ern_model,
    get_hps_model,
    get_sn_model,
    get_mlp_model,
    set_random_states,
    get_model,
    count_parameters,
)

from ern.utils import (
    get_rmse,
    get_preds,
    create_rmse_df_mtl,
    create_rmse_df_mlp,
    to_short_name,
)


def train(
    config,
    model_type,
    data_folder,
    data_type="wind",
    save_model=False,
    output_folder="./results/",
):
    files = glob.glob(f"{data_folder}/*.csv")
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    use_emb = True
    scale_data = False

    if "gefcom" in data_folder or "cosmo" in data_folder:
        scale_data = True

    if model_type == "mlp":
        data_bunch = create_databunch_mtl_mlp(
            files, batch_size=config["batch_size"], scale_data=scale_data
        )
    else:
        data_bunch = create_databunch_mtl(
            files, batch_size=config["batch_size"], scale_data=scale_data
        )
    data_bunch.files = files

    no_input_features = data_bunch.train_ds.x.shape[1]
    no_tasks = data_bunch.train_ds.y.shape[1]

    ann_model = get_model(
        model_type,
        no_input_features,
        config,
        device,
        no_tasks,
        use_emb,
        files,
        dropout=config["dropout"],
        num_supaces=config["num_supaces"],
    ).to(device)

    learn = Learner(
        data_bunch, ann_model, loss_func=MSELossFlat(), metrics=[RMSE(), R2Score()],
    )
    cb = OneCycleScheduler(learn, lr_max=0.01)

    if config["use_one_cycle"]:
        learn.fit(20, callbacks=[cb])

    learn.fit(config["epochs"], lr=config["lr"], wd=config["wd"])

    if save_model:
        inference(learn, output_folder, data_type, model_type, data_bunch, files)
    else:
        ray_log(learn)


def ray_log(learn):
    cur_error = get_rmse(learn, data_type=DatasetType.Valid)
    tune.track.init()
    tune.track.log(root_mean_squared_error=cur_error)


def inference(learn, output_folder, data_type, model_type, data_bunch, files):

    learn.path = pathlib.Path(f"{output_folder}/")
    learn.model_dir = ""
    learn.save(f"{data_type}_{model_type}")
    y, y_hats = get_preds(learn, data_type=DatasetType.Valid)
    if model_type == "mlp":
        df_res, _ = create_rmse_df_mlp(
            y,
            y_hats,
            dev_to_np(data_bunch.valid_ds.cat[:, -1]),
            data_bunch,
            data_type=DatasetType.Valid,
        )
    else:
        df_res, _ = create_rmse_df_mtl(
            y, y_hats, files, data_bunch, data_type=DatasetType.Valid
        )
    df_res["ModelType"] = model_type.upper()
    df_res["ParkId"] = np.array([to_short_name(f) for f in files])
    num_params = count_parameters(learn.model)
    df_res["NParametes"] = num_params
    df_res.to_csv(f"{output_folder}/{data_type}_{model_type}_error.csv")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        help="One of the MTL models [HPS, ERN, SN, CS, MLP].",
        default="sn",
    )

    parser.add_argument(
        "--data_folder",
        help="Path to the data prepared data folder.",
        default="/home/scribbler/workspace/ern_and_te_for_mtl_regression_problems/data/wind/",
    )

    parser.add_argument(
        "--output_folder", help="Path to the store results.", default="./results/wind/",
    )

    parser.add_argument(
        "--num_samples", help="Number of samples in random search.", default=1,
    )

    args = parser.parse_args()
    model_type = args.model_type.lower()
    data_folder = args.data_folder
    num_samples = int(args.num_samples)

    output_folder = args.output_folder + "/mtl/"
    pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

    if "wind" in data_folder.lower():
        data_type = "wind"
    else:
        data_type = "pv"

    config = {
        "lr": tune.choice([0.2, 0.1, 0.01, 0.001, 0.0001]),
        "use_one_cycle": tune.choice([True, False]),
        "wd": tune.choice([0.0, 0.05, 0.1, 0.2]),
        "num_supaces": 1,  # only relevant for sluice network
        "epochs": tune.choice([50, 100, 200]),
        "batch_size": tune.choice([512, 1024, 2048]),
        "size_multiplier": tune.choice([10]),
        "percental_reduce": tune.choice([50]),
        "dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    if model_type == "sn":
        config["num_supaces"] = tune.choice([2, 4, 6])

    # @ray.remote(num_gpus=1)
    def train_wrapper(config):
        train(
            config,
            model_type=model_type,
            data_folder=data_folder,
            data_type=data_type,
            save_model=False,
            output_folder=output_folder,
        )

    if os.path.isabs(output_folder):
        temp_folder = "/mnt/work/transfer/tmp/"
    else:
        temp_folder = "/tmp/"

    # config = {
    #     "lr": 1e-3,
    #     "use_one_cycle": False,
    #     "wd": 0.0,
    #     "num_supaces": 1,  # only relevant for sluice network
    #     "epochs": 5,
    #     "batch_size": 2048,
    #     "size_multiplier": 30,
    #     "percental_reduce": 90,
    #     "dropout": 0.0,
    # }

    # train_wrapper(config)

    if torch.cuda.is_available():
        resources_per_trial = {"gpu": 0.5}
    else:
        resources_per_trial = {"cpu": 1}

    ray.init(
        num_cpus=SLURM_CPUS_PER_TASK, temp_dir=temp_folder,
    )
    analysis = tune.run(
        train_wrapper,
        name=f"{data_type}_{model_type}",
        config=config,
        verbose=1,
        num_samples=num_samples,
        resources_per_trial=resources_per_trial,
    )

    analysis.dataframe().to_csv(f"{output_folder}/{data_type}_{model_type}_config.csv")

    ray.shutdown()

    train(
        analysis.get_best_config(metric="root_mean_squared_error", mode="min"),
        model_type=model_type,
        data_folder=args.data_folder,
        data_type=data_type,
        save_model=True,
        output_folder=output_folder,
    )
