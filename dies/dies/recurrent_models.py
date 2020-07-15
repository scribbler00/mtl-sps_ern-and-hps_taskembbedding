from torch import nn
import torch


class LSTMModel(nn.Module):
    # based on https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_lstm_neuralnetwork/
    def __init__(
        self,
        timeseries_length,
        n_features_hidden_state,
        num_recurrent_layer,
        output_dim,
        dropout=0.0,
    ):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = n_features_hidden_state

        # Number of hidden layers
        self.layer_dim = num_recurrent_layer

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(
            input_size=timeseries_length,
            hidden_size=n_features_hidden_state,
            num_layers=num_recurrent_layer,
            batch_first=True,
            dropout=0,
        )

        # Readout layer
        self.fc = nn.Linear(n_features_hidden_state, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out
