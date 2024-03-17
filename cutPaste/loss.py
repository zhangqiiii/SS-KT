import torch
import torch.nn.functional as F


class ContrastLoss(torch.nn.Module):
    def __init__(self):
        super(ContrastLoss, self).__init__()

    def forward(self, predict, label):
        """
        :param predict: predict in [0,1]
        :param label:
        """
        changed = predict[label > 0].mean()
        unchanged = predict[label == 0].mean()
        res = unchanged - changed
        return res
