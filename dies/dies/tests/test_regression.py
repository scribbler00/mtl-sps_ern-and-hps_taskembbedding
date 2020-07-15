import random
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error as mse
from sklearn import datasets

import unittest

import torch
from fastai.basic_train import Learner
from fastai.callbacks import OneCycleScheduler
from fastai.basic_data import DatasetType

from dies.data import (
    ds_from_df_from_dtypes,
    scale_datasets,
    create_databunch,
    ds_from_df,
)
from dies import data
from dies.mlp import MultiLayerPeceptron
from dies.embedding import Embedding
from dies.utils_pytorch import dev_to_np, xavier_init_uniform
from dies.autoencoder import Autoencoder

random_state = 0


def set_random_states():
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    random.seed(random_state)


def get_df():
    X, y, _ = datasets.make_regression(
        n_samples=50,
        n_features=2,
        bias=1000,
        n_informative=2,
        noise=10,
        coef=True,
        random_state=42,
    )

    df1 = pd.DataFrame(
        data=np.concatenate([X, y.reshape(-1, 1)], axis=1),
        columns=["feat1", "feat2", "target"],
    )
    cats = np.random.randint(low=0, high=10, size=(df1.shape[0], 2))
    df1["cat_1"] = cats[:, 0]
    df1["cat_2"] = cats[:, 1]

    index1 = pd.date_range("2000-01-01", "2000-06-01", periods=df1.shape[0])
    index1 = pd.to_datetime(index1, utc=True)
    df1.index = index1

    return df1


class TestMLP(unittest.TestCase):
    def setUp(self):
        n_features = 3
        device = "cpu"

        df = get_df()
        ds = ds_from_df_from_dtypes(df, "target")
        self.ds_tr, self.ds_val, _ = data.train_test_split_dataset(ds)
        self.db = create_databunch(
            self.ds_tr, self.ds_val, None, batch_size=40, device="cpu"
        )
        set_random_states()

    def test_simple_mlp(self):
        input_size = self.ds_tr.x.shape[1]
        df_tr = self.ds_tr.to_df()

        ann_model = MultiLayerPeceptron(
            input_size, ann_structure=[2, 1], embedding_module=None, dropout=0.1
        )
        ann_model.apply(xavier_init_uniform)

        learn = Learner(self.db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_init = mse(df_tr.target, y_hat)
        learn.fit(1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr.target, y_hat)

        self.assertLess(e_end, e_init)

    def test_mlp_with_yrange(self):
        input_size = self.ds_tr.x.shape[1]
        df_tr = self.ds_tr.to_df()

        y_ranges = self.ds_tr.y_ranges

        ann_model = MultiLayerPeceptron(
            input_size,
            ann_structure=[2, 1],
            embedding_module=None,
            dropout=0.1,
            y_ranges=y_ranges,
        )
        ann_model.apply(xavier_init_uniform)

        learn = Learner(self.db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_init = mse(df_tr.target, y_hat)
        learn.fit(1, lr=0.1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr.target, y_hat)

        self.assertLess(e_end, e_init)

    def test_simple_mlp_with_embedding(self):
        input_size = self.ds_tr.x.shape[1]
        df_tr = self.ds_tr.to_df()

        embedding_module = Embedding([11, 11], embedding_dropout=0.1)

        ann_model = MultiLayerPeceptron(
            input_size,
            ann_structure=[2, 1],
            embedding_module=embedding_module,
            dropout=0.1,
        )
        ann_model.apply(xavier_init_uniform)

        learn = Learner(self.db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_init = mse(df_tr.target, y_hat)
        learn.fit(1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr.target, y_hat)

        self.assertLess(e_end, e_init)

    def test_true(self):
        self.assertTrue(True)


class TestAE(unittest.TestCase):
    def setUp(self):
        n_features = 3
        device = "cpu"

        self.df = get_df()
        self.df.drop("target", axis=1, inplace=True)

        set_random_states()

    def test_simple_ae(self):
        cols = ["feat1", "feat2"]
        ds = ds_from_df(self.df, y_columns=cols, x_columns=cols)
        ds_tr, ds_val, _ = data.train_test_split_dataset(ds)
        db = create_databunch(ds_tr, ds_val, None, batch_size=40, device="cpu")
        df_tr = ds_tr.to_df()

        input_size = ds_tr.x.shape[1]
        print(input_size, ds_tr.y.shape[1])
        ann_structure = [10, 4, 1]

        ann_model = Autoencoder(input_size=input_size, ann_structure=ann_structure)
        ann_model.apply(xavier_init_uniform)

        learn = Learner(db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))

        target_cols = ["feat1_target", "feat2_target"]
        e_init = mse(df_tr[target_cols].values, y_hat)
        learn.fit(1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr[target_cols].values, y_hat)

        self.assertLess(e_end, e_init)

    def test_ae_with_yranges(self):
        cols = ["feat1", "feat2"]
        ds = ds_from_df(self.df, y_columns=cols, x_columns=cols)
        ds_tr, ds_val, _ = data.train_test_split_dataset(ds)
        db = create_databunch(ds_tr, ds_val, None, batch_size=40, device="cpu")
        df_tr = ds_tr.to_df()

        input_size = ds_tr.x.shape[1]
        print(input_size, ds_tr.y.shape[1])
        ann_structure = [10, 4, 1]

        y_ranges = ds_tr.y_ranges

        ann_model = Autoencoder(
            input_size=input_size, ann_structure=ann_structure, y_ranges=y_ranges
        )

        ann_model.apply(xavier_init_uniform)
        set_random_states()
        learn = Learner(db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))

        target_cols = ["feat1_target", "feat2_target"]
        e_init = mse(df_tr[target_cols].values, y_hat)
        learn.fit(1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr[target_cols].values, y_hat)

        self.assertLess(e_end, e_init)

    def test_ae_with_embedding_and_yrange(self):
        cols = ["feat1", "feat2"]
        ds = ds_from_df(
            self.df, y_columns=cols, x_columns=cols, cat_columns=["cat_1", "cat_2"]
        )
        ds_tr, ds_val, _ = data.train_test_split_dataset(ds)
        db = create_databunch(ds_tr, ds_val, None, batch_size=40, device="cpu")
        df_tr = ds_tr.to_df()

        y_ranges = ds_tr.y_ranges

        input_size = ds_tr.x.shape[1]
        print(input_size, ds_tr.y.shape[1])
        ann_structure = [10, 4, 1]

        embedding_module = Embedding([11, 11], embedding_dropout=0.1)
        ann_model = Autoencoder(
            input_size=input_size,
            ann_structure=ann_structure,
            embedding_module=embedding_module,
            embeding_position="start",
            y_ranges=y_ranges,
        )
        set_random_states()
        ann_model.apply(xavier_init_uniform)
        set_random_states()

        learn = Learner(db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))

        target_cols = ["feat1_target", "feat2_target"]
        e_init = mse(df_tr[target_cols].values, y_hat)
        learn.fit(1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr[target_cols].values, y_hat)

        # adds some small tolerance
        self.assertLess(
            e_end, e_init + 0.05,
        )

    def test_ae_with_embedding_at_start(self):
        cols = ["feat1", "feat2"]
        ds = ds_from_df(
            self.df, y_columns=cols, x_columns=cols, cat_columns=["cat_1", "cat_2"]
        )
        ds_tr, ds_val, _ = data.train_test_split_dataset(ds)
        db = create_databunch(ds_tr, ds_val, None, batch_size=40, device="cpu")
        df_tr = ds_tr.to_df()

        input_size = ds_tr.x.shape[1]
        print(input_size, ds_tr.y.shape[1])
        ann_structure = [10, 4, 1]

        embedding_module = Embedding([11, 11], embedding_dropout=0.1)
        ann_model = Autoencoder(
            input_size=input_size,
            ann_structure=ann_structure,
            embedding_module=embedding_module,
            embeding_position="start",
        )
        ann_model.apply(xavier_init_uniform)

        learn = Learner(db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))

        target_cols = ["feat1_target", "feat2_target"]
        e_init = mse(df_tr[target_cols].values, y_hat)
        learn.fit(1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr[target_cols].values, y_hat)

        self.assertLess(e_end, e_init)

    def test_ae_with_embedding_at_bottleneck(self):
        cols = ["feat1", "feat2"]
        ds = ds_from_df(
            self.df, y_columns=cols, x_columns=cols, cat_columns=["cat_1", "cat_2"]
        )
        ds_tr, ds_val, _ = data.train_test_split_dataset(ds)
        db = create_databunch(ds_tr, ds_val, None, batch_size=40, device="cpu")
        df_tr = ds_tr.to_df()

        input_size = ds_tr.x.shape[1]
        print(input_size, ds_tr.y.shape[1])
        ann_structure = [10, 4, 1]

        embedding_module = Embedding([11, 11], embedding_dropout=0.1)
        ann_model = Autoencoder(
            input_size=input_size,
            ann_structure=ann_structure,
            embedding_module=embedding_module,
            embeding_position="bottleneck",
        )
        ann_model.apply(xavier_init_uniform)

        learn = Learner(db, ann_model, loss_func=torch.nn.MSELoss())
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))

        target_cols = ["feat1_target", "feat2_target"]
        e_init = mse(df_tr[target_cols].values, y_hat)
        learn.fit(1, lr=0.1)
        y_hat, _ = dev_to_np(learn.get_preds(DatasetType.Train))
        e_end = mse(df_tr[target_cols].values, y_hat)

        self.assertLess(e_end, e_init)

    def test_true(self):
        self.assertTrue(True)
