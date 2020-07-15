import copy

import torch
import torch.nn as nn
from _collections import defaultdict
from dies.utils_pytorch import init_net
from dies.utils_pytorch import unfreeze_n_final_layer

__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


class HPSNetwork(nn.Module):
    """
    Hard-parameter-sharing-Network
    """

    def __init__(
        self,
        shared_layer_params,
        separate_layer_params,
        number_of_tasks,
        dropout=None,
        hidden_activation=nn.LeakyReLU(negative_slope=0.01),
        use_batch_norm=True,
        final_layer_activation=False,
        embedding_module=None,
        y_ranges=None,
    ):
        """

         shared_layer_params:
         separate_layer_params:
         number_of_tasks:
         hidden_activation:
         use_batch_norm:
         final_layer_activation:
        """
        super(HPSNetwork, self).__init__()

        shared_layer_params = copy.copy(shared_layer_params)
        separate_layer_params = copy.copy(separate_layer_params)

        self.embedding_module = embedding_module
        if self.embedding_module is not None:
            shared_layer_params[0] = (
                shared_layer_params[0] + self.embedding_module.no_of_embeddings
            )

        # save number of tasks
        self.number_of_tasks = number_of_tasks

        # save number of potential y_ranges
        self.y_ranges = y_ranges

        # add shared layer
        layers = [
            nn.Linear(dim1, dim2)
            for dim1, dim2 in zip(shared_layer_params[0:-1], shared_layer_params[1:])
        ]

        layers = init_net(
            layers,
            hidden_activation,
            use_batch_norm,
            include_activation_final_layer=final_layer_activation,
            dropout=dropout,
            combine_to_sequential=True,
        )

        self.shared_layers = nn.Sequential(*layers)

        # add separate layer
        self.separate_layers = nn.ModuleList()
        for j in range(self.number_of_tasks):
            layers = [
                nn.Linear(dim1, dim2)
                for dim1, dim2 in zip(
                    separate_layer_params[0:-1], separate_layer_params[1:]
                )
            ]
            layers = init_net(
                layers,
                hidden_activation,
                use_batch_norm,
                include_activation_final_layer=final_layer_activation,
                dropout=dropout,
                combine_to_sequential=True,
            )

            self.separate_layers.append(nn.Sequential(*layers))

    def forward(self, continuous_data, categorical_data=None):

        if self.embedding_module is not None:
            x = self.embedding_module(categorical_data)
            X = torch.cat([x, continuous_data], 1)
        else:
            X = continuous_data

        # propagation through shared layers
        X = self.shared_layers(X)

        # propagation through separate layers
        X_ = []
        for i in range(self.number_of_tasks):
            cur_task_output = self.separate_layers[i](X)
            if self.y_ranges is not None:
                y_range = self.y_ranges[i]
                #                 cur_task_output, indices = torch.sort(cur_task_output)
                cur_task_output = (y_range[1] - y_range[0]) * torch.sigmoid(
                    cur_task_output
                ) + y_range[0]

            X_.append(cur_task_output)

        # concat output of all tasks
        return torch.cat([x for x in X_], dim=1)

    def unfreeze_n_final_layer(self, n, include_embedding=True):
        # freeze_params(self.shared_layers, n, include_embedding=include_embedding)

        for sl in self.separate_layers:
            unfreeze_n_final_layer(
                sl, n, include_embedding=include_embedding,
            )

        num = max(0, n - len(self.separate_layers[0]))

        unfreeze_n_final_layer(
            self.shared_layers, num, include_embedding=include_embedding,
        )

        unfreeze_n_final_layer(
            self.embedding_module, -1, include_embedding=include_embedding,
        )

        return self


class CSNetwork(nn.Module):
    """
    Cross-stitch Network
    """

    def __init__(
        self,
        layer_params,
        number_of_tasks,
        hidden_activation=nn.LeakyReLU(negative_slope=0.01),
        use_batch_norm=True,
        final_layer_activation=False,
        dropout=None,
        embedding_module=None,
        cross_stitch_init_scheme="INBALANCED",
        num_supaces=1,
        as_sluice_nw=False,
    ):
        """

         layer_params:
         number_of_tasks:
         hidden_activation:
         use_batch_norm:
         final_layer_activation:
        """
        super(CSNetwork, self).__init__()

        self.embedding_module = embedding_module
        layer_params = copy.copy(layer_params)
        self.num_supaces = num_supaces
        self.as_sluice_nw = as_sluice_nw

        for idx, lp in enumerate(layer_params):
            layer_params[idx] = (
                [lp[0]] + [p * self.num_supaces for p in lp[1:-1]] + [lp[-1]]
            )
            additional_betasize = sum(layer_params[idx][1:-2])

        if self.embedding_module is not None:
            # shared_layer_params[0] = shared_layer_params[0] + self.embedding_module.no_of_embeddings
            new_params = []
            for params in layer_params:
                params = copy.copy(params)
                params[0] = params[0] + self.embedding_module.no_of_embeddings
                new_params.append(params)

            layer_params = new_params

        # save number of tasks
        self.number_of_tasks = number_of_tasks
        # save max layer
        self.max_layer = (
            max([len(task_layer_params) for task_layer_params in layer_params]) - 1
        )
        # save layer params
        self.layer_params = layer_params

        # create layer for each task
        self.layers = nn.ModuleList()
        for j in range(self.number_of_tasks):
            task_layers = [
                nn.Linear(dim1, dim2)
                for dim1, dim2 in zip(layer_params[j][0:-1], layer_params[j][1:])
            ]
            if self.as_sluice_nw:
                task_layers[-2] = nn.Linear(
                    layer_params[j][-3] + additional_betasize, layer_params[j][-2]
                )

            task_layers = init_net(
                task_layers,
                hidden_activation,
                use_batch_norm,
                include_activation_final_layer=final_layer_activation,
                combine_to_sequential=True,
                dropout=dropout,
            )

            self.layers.append(nn.ModuleList(task_layers))

        # create alpha units
        self.alpha_units = nn.ModuleList()
        for placement in range(self.max_layer - 1):
            # dimension of alpha unit is the sum of task layer param
            # dimension_alpha_unit = sum([layer_params[j][placement + 1] for j in range(self.number_of_tasks)])
            dimension_alpha_unit = self.number_of_tasks * self.num_supaces

            alpha_matrix = self.init_cross_stitch(
                cross_stitch_init_scheme, dimension_alpha_unit, number_of_tasks
            )

            self.alpha_units.append(AlphaUnit(alpha_matrix, self.num_supaces))

        self.betas = {}
        dimension_beta_unit = len(self.alpha_units)
        for n_task in range(self.number_of_tasks):
            beta_matrix = self.init_cross_stitch(
                cross_stitch_init_scheme, dimension_beta_unit, dimension_beta_unit
            )
            self.betas[n_task] = beta_matrix

    def init_cross_stitch(
        self, cross_stitch_init_scheme, dimension_alpha_unit, number_of_tasks
    ):
        if cross_stitch_init_scheme == "BALANCED":
            alpha_matrix = (
                torch.ones(
                    (dimension_alpha_unit, dimension_alpha_unit), requires_grad=True
                )
                / dimension_alpha_unit
            )
        else:
            alpha_matrix = torch.ones(
                (dimension_alpha_unit, dimension_alpha_unit), requires_grad=True
            )
            for i in range(dimension_alpha_unit):
                for j in range(dimension_alpha_unit):
                    if i == j:
                        alpha_matrix[i, j] = 0.9
                    else:
                        alpha_matrix[i, j] = (0.1) / number_of_tasks
        alpha_matrix = alpha_matrix.reshape(
            1, dimension_alpha_unit, dimension_alpha_unit
        )
        return alpha_matrix

    def forward(self, continuous_data, categorical_data=None):

        # split the data according to the first task layer parameter
        if self.embedding_module is not None:
            embedded_data = self.embedding_module(categorical_data)
            split_positions = [
                self.layer_params[j][0] - self.embedding_module.no_of_embeddings
                for j in range(self.number_of_tasks)
            ]

            X_ = list(torch.split(continuous_data, split_positions, dim=1))

            X_ = [torch.cat([embedded_data, cont_data], 1) for cont_data in X_]
        else:
            X_ = list(
                torch.split(
                    continuous_data,
                    [self.layer_params[j][0] for j in range(self.number_of_tasks)],
                    dim=1,
                )
            )

        results_alpha_units = defaultdict(list)
        for iteration in range(self.max_layer):
            # Layer-Propagation
            for i, task_layers in enumerate(self.layers):
                layer = task_layers[iteration]

                if (iteration == self.max_layer - 2) and self.as_sluice_nw:
                    _xhidden = torch.cat(results_alpha_units[i], axis=1)
                    _xtmp = torch.cat([X_[i], _xhidden], axis=1)
                    X_[i] = layer(_xtmp)
                else:
                    X_[i] = layer(X_[i])

            # Alpha-Unit calculation - if not the last layer
            if iteration != self.max_layer - 1:
                # Calculate Linear Combination with alpha unit
                X_ = self.alpha_units[iteration](X_)

                for id_task, x in enumerate(X_):
                    results_alpha_units[id_task] = results_alpha_units[id_task] + [x]

        return torch.cat([x for x in X_], dim=1)

    def unfreeze_layer(self, n):
        unfreeze_n_final_layer(self.layers, n)


class EmergingRelationNetwork(nn.Module):
    """
    Cross-stitch Network
    """

    def __init__(
        self,
        layer_params,
        number_of_tasks,
        hidden_activation=nn.LeakyReLU(negative_slope=0.01),
        use_batch_norm=True,
        final_layer_activation=False,
        dropout=None,
        embedding_module=None,
        cross_stitch_init_scheme="INBALANCED",
        include_skip_layer=True,
    ):
        """

         layer_params:
         number_of_tasks:
         hidden_activation:
         use_batch_norm:
         final_layer_activation:
        """
        super(EmergingRelationNetwork, self).__init__()
        self.include_skip_layer = include_skip_layer
        self.embedding_module = embedding_module
        layer_params = copy.copy(layer_params)

        if self.embedding_module is not None:
            # shared_layer_params[0] = shared_layer_params[0] + self.embedding_module.no_of_embeddings
            new_params = []
            for params in layer_params:
                params = copy.copy(params)
                params[0] = params[0] + self.embedding_module.no_of_embeddings
                new_params.append(params)

            layer_params = new_params

        # save number of tasks
        self.number_of_tasks = number_of_tasks
        # save max layer
        self.max_layer = (
            max([len(task_layer_params) for task_layer_params in layer_params]) - 1
        )
        # save layer params
        self.layer_params = layer_params

        # for idx, lp in enumerate(layer_params):
        self.size_skip_layer = sum(layer_params[0][1:-2])

        # create layer for each task
        self.layers = nn.ModuleList()
        for j in range(self.number_of_tasks):
            task_layers = [
                nn.Linear(dim1, dim2)
                for dim1, dim2 in zip(layer_params[j][0:-1], layer_params[j][1:])
            ]

            if self.include_skip_layer:
                task_layers[-2] = nn.Linear(
                    layer_params[j][-3] + self.size_skip_layer, layer_params[j][-2]
                )

            task_layers = init_net(
                task_layers,
                hidden_activation,
                use_batch_norm,
                include_activation_final_layer=final_layer_activation,
                combine_to_sequential=True,
                dropout=dropout,
            )

            self.layers.append(nn.ModuleList(task_layers))

        # create alpha units
        self.alpha_units = nn.ModuleList()
        for placement in range(self.max_layer - 1):
            # dimension of alpha unit is the sum of task layer param
            dimension_alpha_unit = sum(
                [layer_params[j][placement + 1] for j in range(self.number_of_tasks)]
            )

            #  assumes that the layers size is equal accross all tasks
            alpha_matrix = self.init_cross_stitch(
                cross_stitch_init_scheme,
                dimension_alpha_unit,
                number_of_tasks,
                layer_params[j][placement + 1],
            )
            self.alpha_units.append(CombiningUnit(alpha_matrix))

    def init_cross_stitch(
        self,
        cross_stitch_init_scheme,
        dimension_alpha_unit,
        number_of_tasks,
        layer_size,
    ):
        if cross_stitch_init_scheme == "BALANCED":
            # alpha_matrix = (torch.ones((dimension_alpha_unit, dimension_alpha_unit),
            #                            requires_grad=True) / dimension_alpha_unit)
            alpha_matrix = torch.diag(
                torch.ones(dimension_alpha_unit, requires_grad=True)
            )
        else:
            # alpha_matrix = (torch.ones((dimension_alpha_unit, dimension_alpha_unit), requires_grad=True))
            alpha_matrix = torch.diag(
                torch.zeros(dimension_alpha_unit, requires_grad=True)
            )
            am_shape = alpha_matrix.shape

            alpha_matrix += 0.1 / (am_shape[0] * am_shape[1] - am_shape[0])

            for i in range(dimension_alpha_unit):
                alpha_matrix[i, i] = 0.9

        return alpha_matrix

    def forward(self, continuous_data, categorical_data=None):

        # split the data according to the first task layer parameter
        if self.embedding_module is not None:
            embedded_data = self.embedding_module(categorical_data)
            split_positions = [
                self.layer_params[j][0] - self.embedding_module.no_of_embeddings
                for j in range(self.number_of_tasks)
            ]

            X_ = list(torch.split(continuous_data, split_positions, dim=1))

            X_ = [torch.cat([embedded_data, cont_data], 1) for cont_data in X_]
        else:
            X_ = list(
                torch.split(
                    continuous_data,
                    [self.layer_params[j][0] for j in range(self.number_of_tasks)],
                    dim=1,
                )
            )

        results_combining_units = defaultdict(list)
        for iteration in range(self.max_layer):
            # Layer-Propagation
            for i, task_layers in enumerate(self.layers):
                layer = task_layers[iteration]

                if (iteration == self.max_layer - 2) and self.include_skip_layer:
                    _xhidden = torch.cat(results_combining_units[i], axis=1)
                    _xtmp = torch.cat([X_[i], _xhidden], axis=1)
                    X_[i] = layer(_xtmp)
                else:
                    X_[i] = layer(X_[i])

            # Alpha-Unit calculation - if not the last layer
            if iteration != self.max_layer - 1:
                # Calculate Linear Combination with alpha unit
                X_ = self.alpha_units[iteration](X_)
                # Split data after alpha unit calculation according to the task layer parameter
                X_ = list(
                    torch.split(
                        X_,
                        [
                            self.layer_params[j][iteration + 1]
                            for j in range(self.number_of_tasks)
                        ],
                        dim=1,
                    )
                )

                for id_task, x in enumerate(X_):
                    results_combining_units[id_task] = results_combining_units[
                        id_task
                    ] + [x]

        return torch.cat([x for x in X_], dim=1)

    def unfreeze_layer(self, n):
        unfreeze_n_final_layer(self.layers, n)


class AlphaUnit(nn.Module):
    def __init__(self, matrix, num_subspaces):
        """
        Alpha-Unit for Cross Stitch Network
        """
        super(AlphaUnit, self).__init__()
        self.matrix = nn.Parameter(matrix)
        self.num_subspaces = num_subspaces

    def forward(self, input_all):
        """

        :param input_all: Expects list with n_task entries. Each element is of size batch_size X hidden_dimension
        :return:
        """
        cat_input = torch.stack(input_all)
        s = cat_input.shape
        # reshape to batchsize X numtask*numsupspaces X hiddendim/numsubspaces
        cat_input = cat_input.reshape(
            s[1], s[0] * self.num_subspaces, int(s[2] / self.num_subspaces)
        )

        res = self.matrix @ cat_input
        res = res.reshape(len(input_all), -1, input_all[0].shape[1])

        res = [res[i] for i in range(len(input_all))]

        return res


class CombiningUnit(nn.Module):
    def __init__(self, matrix):
        """
        Alpha-Unit for Cross Stitch Network
        """
        super(CombiningUnit, self).__init__()
        self.matrix = nn.Parameter(matrix)

    def forward(self, input_all):
        cat_input = torch.cat(input_all, dim=1)
        # Linear Combination: (Samples x dim(A) + dim(B)) * (dim(A) + dim(B) x dim(A) + dim(B)) matrix
        return torch.mm(cat_input, self.matrix)
