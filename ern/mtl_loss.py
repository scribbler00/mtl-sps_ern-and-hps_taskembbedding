from fastai.layers import MSELossFlat
import torch
from torch import nn


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, input, target):

        mse = MSELossFlat()

        loss = 0
        for i in range(self.task_num):
            cur_pred, cur_target = input[i, :], target[i, :]
            cur_loss = mse(cur_pred, cur_target)
            precision = torch.exp(-self.log_vars[i])
            cur_loss = precision * cur_loss + self.log_vars[i]

            loss += cur_loss

        return loss

