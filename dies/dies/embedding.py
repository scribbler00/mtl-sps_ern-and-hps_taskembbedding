import torch
import torch.nn as nn
import pandas as pd
import numpy as np

__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


class Embedding(torch.nn.Module):
    def __init__(
        self, categorical_dimensions, embedding_dropout, embedding_dimensions=None
    ):
        super().__init__()
        """
        Parameters
        ----------
        categorical_dimensions: List with number of categorical
        values for each feature. Output size is calculated based on 
        `min(50, (x + 1) // 2)`. In case explicit dimensions
        are required use `embedding_dimensions`.

        embedding_dropout: Float
        The dropout to be used after the embedding layers.

        embedding_dimensions: List of two element tuples
        This list will contain a two element tuple for each
        categorical feature. The first element of a tuple will
        denote the number of unique values of the categorical
        feature. The second element will denote the embedding
        dimension to be used for that feature. If None, `categorical_dimensions`
        is used to determine the dimensions.

        """
        self.categorical_dimensions = categorical_dimensions

        if embedding_dimensions is None:
            self.embedding_dimensions = [
                (x, min(50, (x + 1) // 2)) for x in categorical_dimensions
            ]
        else:
            self.embedding_dimensions = embedding_dimensions

        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.embedding_dimensions]
        )

        self.no_of_embeddings = sum([y for _, y in self.embedding_dimensions])

        self.embedding_dropout_layer = nn.Dropout(embedding_dropout)

    @staticmethod
    def get_categorical_dimensions(data):
        if type(data) == pd.DataFrame:
            data = data.values

        df_tmp = pd.DataFrame(data=data)

        # print(data.shape)

        cat_dims = [
            int(df_tmp.loc[:, [i]].nunique()) + 1 for i in range(0, data.shape[1])
        ]

        return cat_dims

    def forward(self, categorical_data):

        x = torch.cat(
            [
                emb_layer(categorical_data[:, i])
                for i, emb_layer in enumerate(self.embedding_layers)
            ],
            1,
        )

        x = self.embedding_dropout_layer(x)

        return x
