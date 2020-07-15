import glob
from ern.utils_data import (
    create_databunch_mtl,
    create_databunch_mtl_mlp,
    create_databunch_recurrent_data,
)
import sys
import pathlib
import pickle as pkl

sys.path.append("../ern/")
sys.path.append("../../dies/")
sys.path.append("../dies/")

from ern.utils import to_short_name


def create_db_pickles(files, data_folder):
    data_folder += "dbs/"
    pathlib.Path(data_folder).mkdir(exist_ok=True, parents=True)

    scale_data = False
    if "cosmo" in files[0]:
        scale_data = True
        timesteps = 24 * 4
    elif "wind" in files[0]:
        timesteps = 24
    else:
        timesteps = 24 // 3

    data_bunch = create_databunch_mtl_mlp(files, batch_size=256, scale_data=scale_data)
    data_bunch.files = files
    pkl.dump(data_bunch, open(f"{data_folder}mtl_mlp.pkl", "wb"))

    data_bunch = create_databunch_mtl(files, batch_size=256, scale_data=scale_data)
    data_bunch.files = files
    pkl.dump(data_bunch, open(f"{data_folder}mtl.pkl", "wb"))

    for file in files:
        data_bunch = create_databunch_recurrent_data(
            file=file,
            config={"batch_size": 16},
            device="cpu",
            timesteps=timesteps,
            scale_data=scale_data,
        )
        data_bunch.files = file
        pkl.dump(data_bunch, open(f"{data_folder}{to_short_name(file)}_lstm.pkl", "wb"))

        data_bunch = create_databunch_mtl([file], batch_size=256, scale_data=scale_data)
        data_bunch.files = file
        pkl.dump(data_bunch, open(f"{data_folder}{to_short_name(file)}_stl.pkl", "wb"))


data_folder = "./data/pv_cosmo_old/"
files = glob.glob(f"{data_folder}/*.csv")
create_db_pickles(files, data_folder)

data_folder = "./data/pv/"
files = glob.glob(f"{data_folder}/*.csv")
create_db_pickles(files, data_folder)

data_folder = "./data/wind_cosmo_old/"
files = glob.glob(f"{data_folder}/*.csv")
create_db_pickles(files, data_folder)

data_folder = "./data/wind/"
files = glob.glob(f"{data_folder}/*.csv")
create_db_pickles(files, data_folder)
