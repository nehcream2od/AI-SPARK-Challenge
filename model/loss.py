import torch
import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mean_squared_error(output, target):
    return F.mse_loss(output, target)


def binary_cross_entropy_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = self.loss_fn.reduction  # mean, sum, etc..

    def forward(self, pred, true):
        bceloss = self.loss_fn(pred, true)

        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss = alpha_factor * modulating_factor * bceloss

        if self.reduction == "mean":
            return loss.mean()

        elif self.reduction == "sum":
            return loss.sum()

        else:  # 'none'
            return loss
