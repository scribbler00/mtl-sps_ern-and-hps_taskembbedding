from itertools import chain
from collections import OrderedDict
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch as F
from .utils_pytorch import init_net, unfreeze_n_final_layer
from .embedding import Embedding

__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


class MultiLayerPeceptron(torch.nn.Module):
    def __init__(
        self,
        input_size,
        ann_structure,
        hidden_activation=nn.LeakyReLU(negative_slope=0.01),
        use_batch_norm=True,
        final_layer_activation=None,
        dropout=None,
        embedding_module=None,
        y_ranges=None,
    ):
        """
        
        Parameters
        ----------
         input_size: int
            The input size for continuos features . 

         ann_structure: list
            Sizes of the different layer.

         hidden_activation:
            The activation function to be used after each hidden layer.

         use_batch_norm: bool
            Whether to use batch norm or not after the actvation.
         final_layer_activation:
            The activation to use in the output layer.

         dropout: float
            Amount of dropout to use. 
            
         embedding_module: float
            Amount of dropout to use after embedding layer. 
        """
        super(MultiLayerPeceptron, self).__init__()
        ann_structure = copy.copy(ann_structure)

        # save number of potential y_ranges
        self.y_ranges = y_ranges

        self.embedding_module = embedding_module
        if self.embedding_module is not None:
            input_size = input_size + self.embedding_module.no_of_embeddings

        ann_structure.insert(0, input_size)
        self.input_size = input_size

        self.sequential = ann_structure

        self.sequential = [
            nn.Linear(x, y) for x, y in zip(self.sequential[0:-1], self.sequential[1:])
        ]

        self.sequential = init_net(
            self.sequential,
            hidden_activation,
            use_batch_norm,
            include_activation_final_layer=False,
            dropout=dropout,
            combine_to_sequential=True,
        )

        if final_layer_activation is not None:
            self.sequential.append(final_layer_activation)

        self.main = nn.Sequential(*self.sequential)

        self.n_gpu = torch.cuda.device_count()

    def forward(self, continuous_data, categorical_data=None):
        if self.embedding_module is not None:

            x = self.embedding_module(categorical_data)
            x = torch.cat([x, continuous_data], 1)
        else:
            x = continuous_data

        output = self.main(x)

        if self.y_ranges is not None:
            y_range = self.y_ranges[0]
            output = (y_range[1] - y_range[0]) * torch.sigmoid(output) + y_range[0]

        return output

    def unfreeze_layer(self, n, include_embedding=False):
        unfreeze_n_final_layer(self.main, n, include_embedding=include_embedding)
