from itertools import chain
from collections import OrderedDict
import copy
import numpy as np
import torch
import torch.nn as nn
import torch as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from .mlp import Embedding


from .utils_pytorch import init_net, unfreeze_n_final_layer
from dies.mlp import MultiLayerPeceptron


__author__ = "Jens Schreiber"
__copyright__ = "Copyright 2017, Intelligent Embedded Systems, Universität Kassel"
__status__ = "Prototype"


class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        ann_structure,
        hidden_activation=nn.LeakyReLU(negative_slope=0.1),
        use_batch_norm=True,
        final_layer_activation=None,
        dropout=None,
        embedding_module=None,
        embeding_position="start",
        y_ranges=None,
    ):
        """Class to create an autoencoder network, based on fully connected neural networks
        Args:
            input_size (int): size of the input layer, according to the number of features
            ann_structure (list of ints): the following network structure
            hidden_activation (torch.nn.Module, optional): Activation function used after each hidden layer. Defaults to nn.LeakyReLU(negative_slope=0.1).
            use_batch_norm (bool, optional): Select if batchnorm layer are included after each hidden layer. Defaults to True.
            final_layer_activation (torch.nn.Module, optional): Activation funciton used after the final layer. Defaults to None.
            dropout (float, optional): dropout probability after each hidden layer. Defaults to None.
            embedding_module (dies.embedding.Embedding, optional): Embedding module for categorical features. Defaults to None.
            embeding_position (str, optional): Position at which the output of the embedding layer is included, Either "start" or "bottleneck". Defaults to "start".
            y_ranges ([type], optional): [description]. Defaults to None.

        Raises:
            ValueError: [description]
            Warning: [description]
        """

        super(Autoencoder, self).__init__()

        ann_structure = copy.copy(ann_structure)

        self.latent_size = ann_structure[-1]

        # save number of potential y_ranges
        self.y_ranges = y_ranges

        self.output_size = input_size

        self.embedding_module = embedding_module
        self.embeding_position = embeding_position

        if (
            self.embedding_module is not None
            and self.embeding_position != "start"
            and self.embeding_position != "bottleneck"
        ):
            raise ValueError("embeding_position must be start or bottleneck")

        if self.embedding_module is not None and self.embeding_position == "start":
            input_size = input_size + self.embedding_module.no_of_embeddings

        ann_structure.insert(0, input_size)
        self.input_size = input_size

        self.network_structure_encoder = ann_structure.copy()
        self.network_structure_decoder = [ls for ls in reversed(ann_structure)]
        self.network_structure_decoder[-1] = self.output_size

        if self.network_structure_encoder[-1] != self.network_structure_decoder[0]:
            raise Warning(
                "Last element of encoder should match first element of decoder."
                + "But is {} and {}".format(
                    self.network_structure_encoder, self.network_structure_decoder
                )
            )

        if self.embedding_module is not None and self.embeding_position == "bottleneck":
            self.network_structure_decoder[0] = (
                self.network_structure_decoder[0]
                + self.embedding_module.no_of_embeddings
            )

        self.network_structure_encoder = [
            nn.Linear(x, y)
            for x, y in zip(
                self.network_structure_encoder[0:-1], self.network_structure_encoder[1:]
            )
        ]
        self.network_structure_decoder = [
            nn.Linear(x, y)
            for x, y in zip(
                self.network_structure_decoder[0:-1], self.network_structure_decoder[1:]
            )
        ]

        self.encoder = init_net(
            self.network_structure_encoder,
            hidden_activation,
            use_batch_norm,
            dropout=dropout,
            combine_to_sequential=True,
        )
        self.decoder = init_net(
            self.network_structure_decoder,
            hidden_activation,
            use_batch_norm,
            include_activation_final_layer=False,
            dropout=dropout,
            combine_to_sequential=True,
        )

        if final_layer_activation is not None:
            self.decoder.append(final_layer_activation)

        #  keep self.main for compability with older models
        self.main = self.encoder + self.decoder
        self.main = nn.Sequential(*self.main)

        self.encoder_ = nn.Sequential(*self.encoder)
        self.decoder_ = nn.Sequential(*self.decoder)

        self.n_gpu = torch.cuda.device_count()

    def _encode(self, continuous_data, categorical_data=None):
        if self.embeding_position == "start":
            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, continuous_data], 1)
        else:
            x = continuous_data

        x = self.encoder_(x)

        if self.embeding_position == "bottleneck":
            categorical_data = self.embedding_module(categorical_data)
            x = torch.cat([categorical_data, x], 1)

        return x

    def forward(self, continuous_data, categorical_data=None):
        if self.embedding_module is None:
            x = self.main(continuous_data)
        else:
            x = self._encode(continuous_data, categorical_data)

            x = self.decoder_(x)

        if self.y_ranges is not None:
            for idx, y_range in enumerate(self.y_ranges):
                x[:, idx] = (y_range[1] - y_range[0]) * torch.sigmoid(
                    x[:, idx]
                ) + y_range[0]

        return x

    def unfreeze_layer(self, n, include_embedding=False, only_encoder=True):
        if only_encoder:
            #  make sure everything is turned off
            unfreeze_n_final_layer(self.main, 0, include_embedding=include_embedding)
            unfreeze_n_final_layer(
                self.encoder_, n, include_embedding=include_embedding
            )
        else:
            unfreeze_n_final_layer(self.main, n, include_embedding=include_embedding)


class VariationalAutoencoder(nn.Module):
    def __init__(
        self,
        input_size,
        ann_structure,
        hidden_activation=nn.LeakyReLU(negative_slope=0.1),
        use_batch_norm=False,
        dropout=None,
        final_layer_activation=None,
    ):
        """[summary]

        Args:
            input_size ([type]): [description]
            ann_structure ([type]): [description]
            hidden_activation ([type], optional): [description]. Defaults to nn.LeakyReLU(negative_slope=0.1).
            use_batch_norm (bool, optional): [description]. Defaults to False.
            dropout ([type], optional): [description]. Defaults to None.
            final_layer_activation ([type], optional): [description]. Defaults to None.
        """

        super(VariationalAutoencoder, self).__init__()
        ann_structure = copy.copy(ann_structure)

        self.input_size = input_size
        ann_structure.insert(0, input_size)
        representation_size = ann_structure[-1]
        del ann_structure[-1]

        self.network_structure_encoder = ann_structure.copy()
        self.network_structure_decoder = [ls for ls in reversed(ann_structure)]

        self.network_structure_encoder = [
            nn.Linear(x, y)
            for x, y in zip(
                self.network_structure_encoder[0:-1], self.network_structure_encoder[1:]
            )
        ]
        self.network_structure_decoder = [
            nn.Linear(x, y)
            for x, y in zip(
                self.network_structure_decoder[0:-1], self.network_structure_decoder[1:]
            )
        ]

        self.encoder = init_net(
            self.network_structure_encoder,
            hidden_activation,
            use_batch_norm,
            dropout=dropout,
        )
        self.decoder = init_net(
            self.network_structure_decoder,
            hidden_activation,
            use_batch_norm,
            include_activation_final_layer=False,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

        self.en_mu = nn.Linear(ann_structure[-1], representation_size)
        self.en_std = nn.Linear(ann_structure[-1], representation_size)

        self.de1 = nn.Linear(representation_size, ann_structure[-1])
        self.representation_size = representation_size

    def encode(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        #         h1 = self.act_func(self.en1(x))
        h1 = self.encoder(x)

        return self.en_mu(h1), self.en_std(h1)

    def decode(self, z):
        """Decode a batch of latent variables"""
        h2 = self.de1(z)
        return self.decoder(h2)

    def reparam(self, mu, logvar):
        """Reparameterisation trick to sample z values. 
        This is stochastic during training,  and returns the mode during evaluation."""

        if self.training:
            # convert logarithmic variance to standard deviation representation
            std = logvar.mul(0.5).exp_()
            # create normal distribution as large as the data
            eps = Variable(std.data.new(std.size()).normal_())
            # scale by learned mean and standard deviation
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        mu, logvar = self.encode(x)  # .view(-1, self.input_size))
        z = self.reparam(mu, logvar)

        return self.decode(z)

    def get_posteriors(self, x):

        return self.encode(x)

    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""

        mu, logvar = self.encode(x)
        return self.reparam(mu, logvar)

    def unfreeze_layer(self, n, include_embedding=False):
        unfreeze_n_final_layer(self.decoder, n, include_embedding=include_embedding)

        if n > len(self.decoder):
            unfreeze_n_final_layer(self.en_mu, 1, include_embedding=include_embedding)
            unfreeze_n_final_layer(self.en_std, 1, include_embedding=include_embedding)
            # -1 due to mu,std
            unfreeze_n_final_layer(
                self.encoder,
                n - len(self.decoder) - 1,
                include_embedding=include_embedding,
            )


class VAEReconstructionLoss(nn.Module):
    def __init__(self, model):
        super(VAEReconstructionLoss, self).__init__()
        self.reconstruction_function = torch.nn.MSELoss()
        self.model = model

    def forward(self, x_hat, x):
        recon_x = x_hat

        mu, logvar = self.model.get_posteriors(x)
        # how well do input x and output recon_x agree?

        generation_loss = self.reconstruction_function(recon_x, x)
        # KLD is Kullback–Leibler divergence -- how much does one learned
        # distribution deviate from another, in this specific case the
        # learned distribution from the unit Gaussian

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= x.shape[0] * x.shape[1]

        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return generation_loss + KLD


class ConvolutionalAutoEncoder(torch.nn.Module):
    def __init__(self, cnn_structure, kernel_size, padding=False, debug=False):
        """[summary]
        
        Arguments:
            cnn_structure {Array} -- Structure of the encoder side of the CNN, e.g. [16,12,8,4]
            kernel_size {int} -- [description]
        
        Keyword Arguments:
            adjust_groups {bool} -- [description] (default: {False})
            debug {bool} -- [description] (default: {False})
        """

        super(ConvolutionalAutoEncoder, self).__init__()
        self.debug = debug

        self.padding = kernel_size // 2 if padding else 0

        self.encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "cnn_layer{}_{}".format(input_size, output_size),
                        nn.Conv1d(
                            input_size, output_size, kernel_size, padding=self.padding
                        ),
                    )
                    for input_size, output_size in zip(
                        cnn_structure[0:-1], cnn_structure[1:]
                    )
                ]
            )
        )
        self.decoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "deconv_layer{}_{}".format(input_size, output_size),
                        nn.ConvTranspose1d(
                            input_size, output_size, kernel_size, padding=self.padding
                        ),
                    )
                    for input_size, output_size in zip(
                        reversed(cnn_structure[1:]), reversed(cnn_structure[0:-1])
                    )
                ]
            )
        )

    def get_encoder(self, input):
        return self.encoder(input)

    def shape_print(self, name, output):
        if self.debug:
            print("{}: {}".format(name, output.shape))

    def forward(self, input):
        encoder_out = self.encoder(input)
        return self.decoder(encoder_out)


class CnnMSELoss(torch.nn.MSELoss):
    def __init__(self):
        super(CnnMSELoss, self).__init__(None, None, "mean")

    def forward(self, input, target):
        return torch.mean(
            torch.sqrt(torch.mean(torch.mean(torch.pow((target - input), 2), 2), 0))
        )


class AEMLP(torch.nn.Module):
    def __init__(self, autoencoder, mlp):
        """ Combines an autoencoder and and MLP.
            First encodes the input data and then forwards it to an MLP to create the final output.

        Args:
            autoencoder (Autoencoder): The autoencoder to encode the input data.
            mlp (MultiLayerPeceptron): The MLP to create the response.
        """
        super(AEMLP, self).__init__()
        self.autoencoder = autoencoder
        self.mlp = mlp

        if not isinstance(self.autoencoder, Autoencoder):
            raise TypeError
        if not isinstance(self.mlp, MultiLayerPeceptron):
            raise TypeError

    def forward(self, x, cat=None):
        x = self.autoencoder._encode(x, cat)
        x = self.mlp(x)

        return x

    def unfreeze_layer(self, n, include_embedding=False):
        self.mlp.unfreeze_layer(n, include_embedding=False)
        n = np.max([0, n - len(self.mlp.main)])
        self.autoencoder.unfreeze_layer(n, include_embedding=True, only_encoder=True)
