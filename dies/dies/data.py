import collections
import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import torch
from torch.utils.data.dataloader import DataLoader
from fastai.basic_data import DataBunch

from .utils_pytorch import np_to_dev, dev_to_np
from .utils import listify, np_int_dtypes


__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


class Dataset:
    """
    
    """

    def __init__(
        self,
        x,
        y,
        index=None,
        x_columns=[],
        y_columns=[],
        standardize_X=False,
        device="cpu",
    ):

        self.index = index
        self.x_columns = listify(x_columns)
        self.y_columns = listify(y_columns)

        # in case input and output are the same, rename them
        if collections.Counter(self.x_columns) == collections.Counter(self.y_columns):
            self.y_columns = [f"{c}_target" for c in y_columns]

        if isinstance(self.index, pd.DatetimeIndex):
            self.index = pd.to_datetime(
                self.index, infer_datetime_format=True, utc=True
            )

        self.standardize_X = standardize_X
        self.is_scaled = False
        self.scaler = None

        self.device = device

        self.x, self.y = self._to_tensor(x), self._to_tensor(y)

        if len(self.y.shape) == 1:
            self.y = np.reshape(self.y, (-1, 1))

        if self.standardize_X:
            self._scale_input(StandardScaler())

    @property
    def no_input_features(self):
        return self.x.shape[1]

    @property
    def columns(self):
        return self.x_columns, self.y_columns

    @property
    def no_output_features(self):
        return self.y.shape[1]

    @property
    def y_ranges(self):
        return get_y_ranges(self.y, self.device)

    def to_device(self, device):
        self.x = self.x.to(device)
        self.y = self.y.to(device)

    def to_np(self):
        return dev_to_np(self.x), dev_to_np(self.y)

    def _scale_input(self, scaler):
        x = dev_to_np(self.x)

        # replace with new scaling and invert previous one
        if self.is_scaled and self.scaler is not None:
            x = self.scaler.inverse_transform(x)

        try:
            x = scaler.transform(x)
        except NotFittedError as e:
            x = scaler.fit_transform(x)

        self.x = np_to_dev(x)
        self.scaler = scaler
        self.is_scaled = True

    def to_df(self, filter_columns=None, include_y=True, inverse_transform_input=False):

        X, y = self.to_np()

        if self.is_scaled and self.scaler is not None and inverse_transform_input:
            X = self.scaler.inverse_transform(X)

        return Dataset._create_df(
            X,
            None,
            y,
            self.index,
            filter_columns,
            include_y,
            self.x_columns,
            self.y_columns,
            None,
        )

    def scale_input(self, scaler=None):
        if scaler is None:
            scaler = StandardScaler()
        elif callable(scaler):
            scaler = scaler()
        self._scale_input(scaler)

    @staticmethod
    def _create_df(
        X, cat, y, index, filter_columns, include_y, x_columns, y_columns, cat_columns,
    ):
        index = listify(index)
        columns = listify(filter_columns)
        x_columns = listify(x_columns)
        y_columns = listify(y_columns)
        cat_columns = listify(cat_columns)

        df = pd.DataFrame(data=np.concatenate([X, y], axis=1))

        if len(index) == X.shape[0]:
            df.index = index

        if (len(x_columns) + len(y_columns)) == (X.shape[1] + y.shape[1]):
            df.columns = x_columns + y_columns

        if cat is not None:
            if len(cat_columns) == 0:
                cat_columns = [f"cat_{i}" for i in range(cat.shape[1])]
            for idx, c in enumerate(cat_columns):
                df[c] = cat[:, idx]

        if len(columns) > 0:
            df = df[columns]

        if not include_y:
            df = df.drop(y_columns, axis=1)

        df.sort_index(inplace=True)

        return df

    def _to_tensor(self, data):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            data = data.values

        if isinstance(data, np.ndarray):
            data = np_to_dev(data, self.device)

        return data

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def has_categorical_data(self):
        return False


class DatasetCategoricalData(Dataset):
    def __init__(
        self,
        x,
        cat,
        y,
        index=None,
        x_columns=[],
        cat_columns=[],
        y_columns=[],
        standardize_X=False,
        device="cpu",
    ):
        super().__init__(x, y, index, x_columns, y_columns, standardize_X, device)

        self.cat_columns = listify(cat_columns)

        self.cat = self._to_tensor(cat)

    @property
    def no_input_features(self):
        return self.x.shape[1], self.cat.shape[1]

    def __getitem__(self, item):
        return (self.x[item], self.cat[item]), self.y[item]

    def has_categorical_data(self):
        return True

    def to_np(self):
        return dev_to_np(self.x), dev_to_np(self.cat), dev_to_np(self.y)

    def to_device(self, device):
        super().to_device(device)
        self.cat = self.cat.to(device)

    def to_df(self, filter_columns=None, include_y=True, inverse_transform_input=False):
        X, cat, y = self.to_np()

        if self.is_scaled and self.scaler is not None and inverse_transform_input:
            X = self.scaler.inverse_transform(X)

        return Dataset._create_df(
            X,
            cat,
            y,
            self.index,
            filter_columns,
            include_y,
            self.x_columns,
            self.y_columns,
            self.cat_columns,
        )


def get_y_ranges(y, device):
    """
    Get the minimum and maximum values of the targets y.

    Parameters:
    ----------
    y: array
    target values.
    device: string
    the device for the tensor.
    """

    y_ranges = []
    for i in range(0, y.shape[1]):
        max_y = torch.max(y[:, i]) * 1.2
        min_y = torch.min(y[:, i]) * 1.2
        y_range = torch.tensor([min_y, max_y], device=device)
        y_ranges.append(y_range)

    return y_ranges


def combine_datasets(datasets, dim=0):
    datasets = listify(datasets)

    if len(datasets) >= 2:
        ds_merged = datasets[0]
        for i in range(1, len(datasets)):
            ds_merged = _combine_datasets(ds_merged, datasets[i], dim=dim)
    elif len(datasets) == 1:
        ds_merged = datasets[0]
    else:
        ds_merged = None

    return ds_merged


def _combine_datasets(ds1, ds2, dim=0):
    ds1 = copy.copy(ds1)
    ds2 = copy.copy(ds2)

    # combine test and training, as we don't need a testing set for the source task
    ds1.x = torch.cat([ds1.x, ds2.x], dim=dim)
    ds1.y = torch.cat([ds1.y, ds2.y], dim=dim)

    # in case of dim=1, a single index is sufficient
    if hasattr(ds1, "index") and hasattr(ds2, "index") and dim == 0:
        ds1.index = np.concatenate(
            [
                np.array(ds1.index, dtype="datetime64[ns]"),
                np.array(ds2.index, dtype="datetime64[ns]"),
            ]
        )

    # in case of dim=1, a single categorical data is sufficient
    if hasattr(ds1, "cat") and hasattr(ds2, "cat") and dim == 0:
        ds1.cat = torch.cat([ds1.cat, ds2.cat])

    return ds1


def ds_from_df_from_dtypes(df, y_columns, standardize_X=False):
    """
    Convert a pandas dataframe into a dataset with the respective d_types of the input features.

    Parameters:
    ----------
    df: pandas dataframe
        the dataframe used for the conversion.
    y_columns: string or list of strings
        name of the columns which are used as targets.
    standardize_X: bool, default=False
        if True, the features of the dataset are standardized.
    """

    y_columns = listify(y_columns)
    index = df.index

    x_cat_columns = df.drop(y_columns, axis=1).columns

    dtypes = df[x_cat_columns].dtypes
    dtypes = dict(zip(x_cat_columns, dtypes))

    x_columns = []
    cat_columns = []

    for cur_col_name in dtypes.keys():
        cur_dtype = dtypes[cur_col_name]

        if cur_dtype in np_int_dtypes:
            cat_columns.append(cur_col_name)
        else:
            x_columns.append(cur_col_name)

    x = df[x_columns].values
    cat = df[cat_columns].values
    y = df[y_columns].values

    if len(cat_columns) > 0:
        ds = DatasetCategoricalData(
            x=x,
            cat=cat,
            y=y,
            index=df.index,
            x_columns=x_columns,
            cat_columns=cat_columns,
            y_columns=y_columns,
            standardize_X=standardize_X,
        )
    else:
        ds = Dataset(
            x=x,
            y=y,
            index=df.index,
            x_columns=x_columns,
            y_columns=y_columns,
            standardize_X=standardize_X,
        )

    return ds


def ds_from_df(df, y_columns, cat_columns=[], x_columns=[]):
    """
    Convert a pandas dataframe into a dataset.

    Parameters:
    ----------
    df: pandas dataframe
        the dataframe used for the conversion.
    y_columns: string or list of strings
        name of the columns which are used as targets.
    cat_columns: list of strings
        names of the categorical features.
    x_columns: list of strings
        names of the non-categorical features. If empty, all features except the targets and categorical features are used. 
    """

    y_columns = listify(y_columns)
    index = df.index

    if len(x_columns) == 0:
        x_columns = df.drop(y_columns + cat_columns, axis=1).columns

    X = df[x_columns]
    cat = df[cat_columns]
    y = df[y_columns]

    if len(cat_columns) > 0:
        ds = DatasetCategoricalData(
            x=X,
            cat=cat,
            y=y,
            index=index,
            x_columns=x_columns,
            cat_columns=cat_columns,
            y_columns=y_columns,
        )
    else:
        ds = Dataset(x=X, y=y, index=index, x_columns=x_columns, y_columns=y_columns)

    return ds


def scale_datasets(train_dataset, other_datasets, scaler=None):
    """
    Scale datasets to the range of the training dataset.

    Parameters:
    ----------
    train_dataset: dataset
        the training dataset used for the scaling
    other_datasets: dataset or list of datasets
        the other datasets (validation, test) which are scaled according to the training dataset.
    scaler: scaler object, default=None
        type of scaler used for the scaling, e.g. MinMaxScaler.
    """

    other_datasets = listify(other_datasets)

    train_dataset.scale_input(scaler)

    for cur_dataset in other_datasets:
        cur_dataset.scale_input(train_dataset.scaler)


def _filter(dataset, indexes):
    dataset.x = dataset.x[indexes, :]
    dataset.y = dataset.y[indexes, :]

    dataset.index = dataset.index[indexes]

    if hasattr(dataset, "cat"):
        dataset.cat = dataset.cat[indexes, :]


def train_test_split_dataset_by_n_weeks(
    cur_dataset, every_n_weeks=4, for_n_weeks=1, skip_first=True
):
    start_date = np.min(cur_dataset.index)
    end_date = np.max(cur_dataset.index)

    sw = pd.date_range(start_date, end_date, freq=f"{every_n_weeks*7}D", normalize=True)
    ew = pd.date_range(
        start_date + pd.DateOffset(days=for_n_weeks * 7),
        end_date,
        freq=f"{(every_n_weeks)*7}D",
        normalize=True,
    )

    # in case we have a begining of a year, the first week might be rather short
    if skip_first:
        sw = sw[1:]
        ew = ew[1:]

    mask = np.zeros(len(cur_dataset.index), dtype=np.bool)

    for s, e in zip(sw, ew):

        mask = mask | ((cur_dataset.index > s) & (cur_dataset.index < e))

    indices = np.arange(len(cur_dataset))
    train_index, test_index = indices[~mask], indices[mask]

    tr_dataset, test_dataset = (
        copy.copy(cur_dataset),
        copy.copy(cur_dataset),
    )

    _filter(tr_dataset, train_index)
    _filter(test_dataset, test_index)

    return tr_dataset, test_dataset


def train_test_split_dataset_with_test_date(
    cur_dataset, test_date, train_size=0.8, random_state=42, shuffle=True,
):
    test_date = pd.to_datetime(test_date, utc=True)

    mask = cur_dataset.index < test_date
    indices = np.arange(len(cur_dataset))

    train_index, test_index = indices[mask], indices[~mask]

    tr_dataset, test_dataset = (
        copy.copy(cur_dataset),
        copy.copy(cur_dataset),
    )

    _filter(tr_dataset, train_index)
    _filter(test_dataset, test_index)

    tr_dataset, val_dataset, _ = train_test_split_dataset(
        cur_dataset=tr_dataset,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        include_test_dataset=False,
    )

    return tr_dataset, val_dataset, test_dataset


def train_test_split_dataset(
    cur_dataset,
    train_size=0.8,
    random_state=42,
    include_test_dataset=False,
    shuffle=True,
):
    indices = np.arange(len(cur_dataset))

    train_index, val_index = train_test_split(
        indices, train_size=train_size, shuffle=shuffle, random_state=random_state
    )

    tr_dataset, val_dataset, te_dataset = (
        copy.copy(cur_dataset),
        copy.copy(cur_dataset),
        copy.copy(cur_dataset),
    )

    # does the stuff in place, as we have copied it before
    _filter(tr_dataset, train_index)
    _filter(val_dataset, val_index)

    if include_test_dataset:
        indices = np.arange(len(tr_dataset))

        train_index, te_index = train_test_split(
            indices, train_size=train_size, shuffle=shuffle, random_state=random_state
        )
        _filter(tr_dataset, train_index)
        _filter(te_dataset, te_index)
    else:
        te_dataset = None

    return tr_dataset, val_dataset, te_dataset


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


class DatasetRecurrentData(Dataset):
    # assumes batch first is used in lstm
    def __getitem__(self, item):
        return self.x[item, :, :], self.y[item,]

    @property
    def no_input_features(self):
        return self.x.shape[1]

    @property
    def no_output_features(self):
        if len(self.y.shape) == 3:
            return self.y.shape[1]
        else:
            return 1

    def __len__(self):
        return self.x.shape[0]


def convert_data_to_recurrent_data(data, timesteps=24):
    conv = int(data.shape[0] / timesteps)
    n_features = data.shape[1]
    data_new = np.zeros((conv - 1, n_features, timesteps))

    for i, ix in enumerate(range(int(data.shape[0] / timesteps) - 1)):
        start = ix * timesteps
        end = (ix + 1) * timesteps
        # data_new[i, :, :] = data.swapaxes(0, 1).iloc[:, start:end]
        data_new[i, :, :] = data.swapaxes(0, 1)[:, start:end]

    if data_new.shape[1] == 1:
        data_new = data_new.reshape(-1, timesteps)

    return torch.from_numpy(data_new).type(torch.FloatTensor)


# class DatasetRecurrentWithCategoricalData(DatasetRecurrentData):
#     def __init__(self, x, cat, y, device="cpu"):
#         super().__init__(x, y, device)
#         self.cat = self._to_tensor(cat, type=torch.LongTensor)

#     def __getitem__(self, item):
#         return (self.x[:, item, :], self.cat[:, item]), self.y[:, item]

#     @property
#     def no_input_features(self):
#         return self.x.shape[2], self.cat.shape[1]

#     def has_categorical_data(self):
#         return True


def split_by_date(cur_dataset, split_date):
    split_date = pd.to_datetime(split_date, utc=True)

    mask = cur_dataset.index < split_date
    indices = np.arange(len(cur_dataset))

    train_index, test_index = indices[mask], indices[~mask]

    dataset_1, dataset_2 = (
        copy.copy(cur_dataset),
        copy.copy(cur_dataset),
    )

    _filter(dataset_1, train_index)
    _filter(dataset_2, test_index)

    return dataset_1, dataset_2
