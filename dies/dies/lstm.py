import torch
import torch.nn as nn
import torch as F


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x, cat_data=None):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)
        if cat_data is not None:
            cat_data_reshaped = cat_data.contiguous().view(
                -1, cat_data.size(-1)
            )  # (samples * timesteps, input_size)
        else:
            cat_data_reshaped = None

        y = self.module(x_reshape, cat_data_reshaped)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(
                x.size(0), -1, y.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class LstmModule(nn.Module):
    def __init__(self, lstm_input_size, num_layers, hidden_dim, output_dim):
        super(LstmModule, self).__init__()

        self.lstm = nn.LSTM(lstm_input_size, hidden_dim, num_layers, batch_first=True)

        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, lstm_hidden = self.lstm(x)

        x = self.linear(lstm_out)

        return x.view(x.shape[0], x.shape[1])


# based on fast ai course v2, see https://github.com/fastai/course-v3/blob/master/nbs/dl2/07_batchnorm.ipynb
class GeneralRelu(nn.Module):
    def __init__(self, leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak, self.sub, self.maxv = leak, sub, maxv

    def forward(self, x):
        x = F.leaky_relu(x, self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None:
            x.sub_(self.sub)
        if self.maxv is not None:
            x.clamp_max_(self.maxv)
        return x
